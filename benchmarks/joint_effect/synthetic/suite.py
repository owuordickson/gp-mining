"""Use independently generated cases."""

from argparse import ArgumentParser, Namespace
from itertools import product

from .generator import synthetic_split


def add_arguments(parser: ArgumentParser) -> None:
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


def cases(args: Namespace):
    for regime, noise, seed in product(
        dict.fromkeys(args.regimes),
        dict.fromkeys(args.noise),
        dict.fromkeys(args.seeds),
    ):
        info = {"dataset": regime, "noise": noise, "seed": seed}
        split = synthetic_split(
            regime,
            seed=seed,
            noise=noise,
            train_size=args.train_size,
            test_size=args.test_size,
        )
        yield info, split
