"""Run the joint-effect benchmark from an IDE or the command line."""

import argparse
import csv
import json
import platform
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    # Use this checkout's SO4GP even if the interpreter has an older install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

if __package__:
    from .synthetic.tools import data_tools as synthetic_data_tools
    from .uea.tools import data_tools as uea_data_tools
    from .uea.tools.evaluation import Protocol, evaluate_split
else:
    from synthetic.tools import data_tools as synthetic_data_tools
    from uea.tools import data_tools as uea_data_tools
    from uea.tools.evaluation import Protocol, evaluate_split

PARENT_PATH = Path(__file__).resolve().parent
EXPERIMENT = {"uea": uea_data_tools, "synthetic": synthetic_data_tools}

# These settings apply when the IDE's Run button passes no arguments.
RUN_EXPERIMENT = "uea"
RUN_OPTIONS = {
    "uea": ("--datasets", "ERing"),
    "synthetic": (
        "--seeds",
        "0",
        "--noise",
        "0.5",
        "--train-size",
        "32",
        "--test-size",
        "24",
        "--max-patterns",
        "8",
        "--folds",
        "2",
        "--bootstrap",
        "100",
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="experiment", required=True)

    for name, exp_tools in EXPERIMENT.items():
        mode = modes.add_parser(name, help=exp_tools.__doc__)
        exp_tools.add_arguments(mode)  # Updates and adds to mode specific (uea or synthetic) arguments
        mode.add_argument("--output", type=Path, help="A new directory for reports")
        mode.add_argument("--max-patterns", type=int, default=24)
        mode.add_argument("--min-support", type=float, default=Protocol.min_support)
        mode.add_argument(
            "--folds",
            type=int,
            default=Protocol.folds,
            help="Maximum number of stratified cross-validation folds",
        )
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
        ("dataset_name", "noise_type", "method", "baseline")
        if paired
        else ("dataset_name", "noise_type", "method")
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
    from so4gp import __version__ as so4gp_version

    versions = {"python": platform.python_version(), "so4gp": so4gp_version}
    for package in ("numpy", "scikit-learn", "torch"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "source checkout"
    return versions


def main(argv: list[str] | None = None) -> int:
    """Run explicit arguments or use the settings in RUN_EXPERIMENT and RUN_OPTIONS."""
    if argv is None:
        argv = [RUN_EXPERIMENT]#, *RUN_OPTIONS.get(RUN_EXPERIMENT, ())]
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
            PARENT_PATH
            / args.experiment
            / "results"
            / f"{args.experiment}-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}"
    )
    if output.exists():
        parser.error(f"Output already exists: {output}")
    metrics, contrasts, runs = [], [], []
    for metadata, train_test_data in EXPERIMENT[args.experiment].fetch_data(args):
        run = evaluate_split(*train_test_data, seed=metadata["random_seed"], protocol=protocol)
        common = {k: metadata[k] for k in ("dataset_name", "noise_type", "random_seed")}
        metrics.extend({**common, **row} for row in run.pop("metrics"))
        contrasts.extend({**common, **row} for row in run.pop("contrasts"))
        runs.append({**metadata, **run})
        print(
            f"{metadata['dataset_name']} seed={metadata['random_seed']} noise={metadata['noise_type']} "
            f"folds={run['cv_folds']} patterns={len(run['patterns'])} "
            f"elapsed={run['total_seconds']:.2f}s",
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
        "protocol": {**asdict(protocol), "pattern_miner": "SO4GP GRAANK (Apriori)"},
        "versions": _versions(),
        "runs": runs,
    }
    (output / "run.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Reports written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
