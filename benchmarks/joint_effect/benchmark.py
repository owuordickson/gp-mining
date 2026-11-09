"""Command-line entry point for the joint-effect benchmark."""

import argparse
import csv
import json
import platform
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from tools.archive import BASE_URL, DATASETS, archive_split
from tools.evaluation import Protocol, evaluate_split
from tools.synthetic import synthetic_split

HERE = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="experiment", required=True)
    uea = modes.add_parser("uea", help="Use the published UEA archive splits")
    uea.add_argument("--datasets", choices=DATASETS, nargs="+", default=list(DATASETS))
    uea.add_argument("--data-dir", type=Path, default=HERE / "data")
    uea.add_argument(
        "--download", action="store_true", help="Fetch missing official ZIPs"
    )
    uea.add_argument("--seed", type=int, default=0, help="CV and bootstrap seed")

    synthetic = modes.add_parser("synthetic", help="Use independently generated cases")
    synthetic.add_argument(
        "--regimes",
        choices=("shared", "timing"),
        nargs="+",
        default=["shared", "timing"],
    )
    synthetic.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    synthetic.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    synthetic.add_argument("--train-size", type=int, default=120)
    synthetic.add_argument("--test-size", type=int, default=80)

    for mode in (uea, synthetic):
        mode.add_argument("--output", type=Path, help="A new directory for reports")
        mode.add_argument("--max-patterns", type=int, default=24)
        mode.add_argument("--min-support", type=float, default=0.4)
        mode.add_argument("--folds", type=int, default=3)
        mode.add_argument("--c-grid", type=float, nargs="+", default=[0.1, 1.0, 10.0])
        mode.add_argument("--bootstrap", type=int, default=1000)
    return parser


def _cases(args):
    if args.experiment == "uea":
        for dataset in dict.fromkeys(args.datasets):
            yield (
                {
                    "dataset": dataset,
                    "noise": None,
                    "seed": args.seed,
                    "source_url": f"{BASE_URL}/{dataset}.zip",
                },
                archive_split(dataset, args.data_dir, download=args.download),
            )
    else:
        for regime in dict.fromkeys(args.regimes):
            for noise in dict.fromkeys(args.noise):
                for seed in dict.fromkeys(args.seeds):
                    yield (
                        {"dataset": regime, "noise": noise, "seed": seed},
                        synthetic_split(
                            regime,
                            seed=seed,
                            noise=noise,
                            train_size=args.train_size,
                            test_size=args.test_size,
                        ),
                    )


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict], *, paired: bool = False) -> list[dict]:
    keys = (
        ("dataset", "noise", "method", "baseline")
        if paired
        else ("dataset", "noise", "method")
    )
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)

    summary = []
    for group, records in groups.items():
        row = dict(zip(keys, group, strict=True))
        row["runs"] = len(records)
        for metric in ("balanced_accuracy", "macro_f1"):
            values = [record[metric] for record in records]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
        summary.append(row)
    return summary


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "scikit-learn", "so4gp", "torch"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "source checkout"
    return versions


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    protocol = Protocol(
        max_patterns=args.max_patterns,
        min_support=args.min_support,
        folds=args.folds,
        c_grid=tuple(args.c_grid),
        bootstrap=args.bootstrap,
    )
    output = args.output or (
        HERE / "results" / f"{args.experiment}-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}"
    )
    if output.exists():
        parser.error(f"Output already exists: {output}")
    metrics, contrasts, runs = [], [], []
    for info, split in _cases(args):
        run = evaluate_split(*split, seed=info["seed"], protocol=protocol)
        common = {k: info[k] for k in ("dataset", "noise", "seed")}
        metrics.extend({**common, **row} for row in run.pop("metrics"))
        contrasts.extend({**common, **row} for row in run.pop("contrasts"))
        runs.append({**info, **run})
        print(
            f"{info['dataset']} seed={info['seed']} noise={info['noise']} "
            f"patterns={len(run['patterns'])} elapsed={run['total_seconds']:.2f}s",
            flush=True,
        )

    output.mkdir(parents=True)
    _csv(output / "metrics.csv", metrics)
    _csv(output / "paired_contrasts.csv", contrasts)
    _csv(output / "summary.csv", _summarize(metrics))
    _csv(output / "paired_summary.csv", _summarize(contrasts, paired=True))
    manifest = {
        "experiment": args.experiment,
        "arguments": vars(args),
        "protocol": asdict(protocol),
        "versions": _versions(),
        "runs": runs,
    }
    (output / "run.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Reports written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
