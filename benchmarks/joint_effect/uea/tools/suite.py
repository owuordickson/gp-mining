"""Use the published UEA archive splits."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from .archive import BASE_URL, DATASETS, archive_split


def add_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--datasets", choices=DATASETS, nargs="+", default=list(DATASETS)
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    parser.add_argument(
        "--download", action="store_true", help="Fetch missing official ZIPs"
    )
    parser.add_argument("--seed", type=int, default=0, help="CV and bootstrap seed")


def cases(args: Namespace):
    for dataset in dict.fromkeys(args.datasets):
        info = {
            "dataset": dataset,
            "noise": None,
            "seed": args.seed,
            "source_url": f"{BASE_URL}/{dataset}.zip",
        }
        yield info, archive_split(dataset, args.data_dir, download=args.download)
