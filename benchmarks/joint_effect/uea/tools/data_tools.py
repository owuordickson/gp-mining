"""Use the published UEA archive splits."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from .download import BASE_URL, DATASETS, load_splits


def add_arguments(parser: ArgumentParser) -> None:
    """Register dataset and data-access options on the UEA command parser.

    Modify parser in place with the supported dataset choices, cache
    directory, optional download flag, and CV/bootstrap seed. Resolve
    the default cache directory relative to this suite's location.
    """
    parser.add_argument(
        "--datasets", choices=DATASETS, nargs="+", default=list(DATASETS)
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data"
    )
    parser.add_argument(
        "--download", action="store_true", help="Fetch missing official ZIPs"
    )
    parser.add_argument("--seed", type=int, default=0, help="CV and bootstrap seed")


def fetch_data(args: Namespace):
    """Yield each requested UEA split with its reporting metadata.

    Remove duplicate dataset names while preserving their first
    occurrence, then call load_splits for each name. Each yielded
    pair contains dataset, seed, noise, and source-URL metadata plus
    the four training/test arrays expected by evaluate_split.
    """
    for dataset in dict.fromkeys(args.datasets):
        metadata = {
            "dataset_name": dataset,
            "noise_level": None,
            "random_seed": args.seed,
            "source_url": f"{BASE_URL}/{dataset}.zip",
        }
        train_test_data = load_splits(dataset, args.data_dir, download=args.download)
        yield metadata, train_test_data
