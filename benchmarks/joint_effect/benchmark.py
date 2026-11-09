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
from uea.tools.evaluation import Protocol, evaluate_split
from synthetic import suite as synthetic_suite
from uea.tools import suite as uea_suite

HERE = Path(__file__).resolve().parent
SUITES = {"uea": uea_suite, "synthetic": synthetic_suite}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="experiment", required=True)

    for name, suite in SUITES.items():
        mode = modes.add_parser(name, help=suite.__doc__)
        suite.add_arguments(mode)
        mode.add_argument("--output", type=Path, help="A new directory for reports")
        mode.add_argument("--max-patterns", type=int, default=24)
        mode.add_argument("--min-support", type=float, default=0.4)
        mode.add_argument("--folds", type=int, default=3)
        mode.add_argument("--c-grid", type=float, nargs="+", default=[0.1, 1.0, 10.0])
        mode.add_argument("--bootstrap", type=int, default=1000)
    return parser


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
        HERE
        / args.experiment
        / "results"
        / f"{args.experiment}-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}"
    )
    if output.exists():
        parser.error(f"Output already exists: {output}")
    metrics, contrasts, runs = [], [], []
    for info, split in SUITES[args.experiment].cases(args):
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
