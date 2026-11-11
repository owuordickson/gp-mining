"""Use independently generated cases."""

from argparse import ArgumentParser, Namespace
from itertools import product

from .generator import generate_splits


def add_arguments(parser: ArgumentParser) -> None:
    """Register synthetic generation options on the command parser.

    Modify parser in place with regime, noise, and seed lists plus
    training and test sizes. The suite combines these lists into
    the requested experiment grid.
    """
    parser.add_argument(
        "--regimes",
        choices=("shared", "timing"),
        nargs="+",
        default=["shared", "timing"],
    )
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--train-size", type=int, default=120)
    parser.add_argument("--test-size", type=int, default=80)


def fetch_data(args: Namespace):
    """Yield synthetic splits for each distinct experiment setting.

    Deduplicate regimes, noise levels, and seeds in their input order
    and iterate over their Cartesian product. Call generate_splits
    for each combination and yield its dataset/noise/seed metadata
    together with the four training/test arrays.
    """
    for regime, noise_level, random_seed in product(
        dict.fromkeys(args.regimes),
        dict.fromkeys(args.noise),
        dict.fromkeys(args.seeds),
    ):
        metadata = {
            "dataset_name": regime,
            "noise_level": noise_level,
            "random_seed": random_seed,
        }
        train_test_data = generate_splits(
            regime,
            seed=random_seed,
            noise=noise_level,
            train_size=args.train_size,
            test_size=args.test_size,
        )
        yield metadata, train_test_data
