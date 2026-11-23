"""Isolated GRAANK worker for the optimization benchmark.

The runner sends one JSON object on standard input and receives one JSON result.
The dense worker selects the pinned SO4GP source before importing it.
"""

from __future__ import annotations

import io
import json
import random
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

SEARCH_TYPES = ("apriori", "pso", "ga", "hc", "random")
DEFAULT_ITERATIONS = 10
DEFAULT_SUPPORT = 0.2
EMPTY_SEARCH_ERRORS = ("Search space is empty!", "Pairwise matrices not available!")


def _load_graank(method, source_path):
    expected_source = None
    if method == "dense":
        if not source_path:
            raise ValueError("Dense GRAANK requires the pinned source_path")
        path = Path(source_path).resolve()
        expected_source = path / "so4gp" / "__init__.py"
        if not expected_source.is_file():
            raise ValueError(f"No dense SO4GP package at {path}")
        if "so4gp" in sys.modules:
            raise RuntimeError("Dense GRAANK requires a fresh worker process")
        sys.path.insert(0, str(path))
    elif source_path is not None:
        raise ValueError("A source_path override is only supported for dense GRAANK")

    import so4gp
    from so4gp.algorithms import GRAANK
    from so4gp.gradual_patterns import PairwiseMatrix

    actual = Path(so4gp.__file__).resolve()
    if expected_source is not None and actual != expected_source:
        raise RuntimeError(
            f"Expected to import SO4GP from {expected_source}, got {actual}"
        )
    bitmap_field = "bin_mat" if method == "dense" else "packed_bin_mat"
    if bitmap_field not in PairwiseMatrix.__dataclass_fields__:
        raise RuntimeError(f"SO4GP at {actual} does not provide {method} bitmaps")
    return GRAANK, str(actual)


def _mining_engine(miner):
    """Read the public engine property in either SO4GP revision."""
    engine = getattr(miner, "mine_engine", None)
    return engine if engine is not None else miner.mining_engine


def run_experiment(config: dict) -> dict:
    """Execute one requested experiment and return a JSON-serializable record."""
    if config["experiment"] != "graank":
        raise ValueError(f"Unknown experiment: {config['experiment']}")
    method = config["method"]
    if method not in ("dense", "packed"):
        raise ValueError(f"Unknown GRAANK bitmap method: {method}")
    search_type = config.get("search_type", "apriori")
    if search_type not in SEARCH_TYPES:
        raise ValueError(f"Unknown GRAANK search type: {search_type}")
    if method == "dense" and search_type != "apriori":
        raise ValueError("Dense GRAANK supports only APRIORI")
    if "data_csv" in config:
        data = pd.read_csv(io.StringIO(config["data_csv"]))
    else:
        data = pd.read_csv(config["data_path"])

    graank, source = _load_graank(method, config.get("source_path"))
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    support = float(config.get("support", DEFAULT_SUPPORT))
    iteration_limit = (
        config.get("apriori_level")
        if search_type == "apriori"
        else config.get("iterations", DEFAULT_ITERATIONS)
    )
    start = time.perf_counter()
    miner = graank(data.copy(), min_sup=support, eq=False)
    try:
        result_json = miner.discover(
            search_type=search_type,
            max_iteration=iteration_limit,
            target_col=config.get("target"),
            compute_descriptors=False,
        )
    except (AttributeError, ValueError) as exc:
        # Some releases of GA and PSO pass the ValueError to the JSON encoder.
        error = (
            exc.obj if isinstance(exc, AttributeError) and exc.name == "tolist" else exc
        )
        if not isinstance(error, ValueError) or str(error) not in EMPTY_SEARCH_ERRORS:
            raise
        result_json = json.dumps({"Error": str(error)})
    elapsed = time.perf_counter() - start
    engine = _mining_engine(miner)
    result = json.loads(result_json)
    invalid_count = result.get("Invalid Count")
    extra = {}
    if "Error" in result:
        error = result["Error"]
        if engine.valid_bins is None and error in EMPTY_SEARCH_ERRORS:
            # Too few gradual items passed support filtering to form a pattern.
            extra["empty_search_space"] = True
            result["Empty search reason"] = result.pop("Error")
            invalid_count = 0
        else:
            raise RuntimeError(str(error))
    patterns = [
        {"items": pattern.to_string(), "reported_support": float(pattern.support)}
        for pattern in engine.gradual_patterns or []
    ]
    return {
        "elapsed_seconds": elapsed,
        "source_path": source,
        "patterns": patterns,
        "pattern_count": len(patterns),
        "valid_pattern_count": len(patterns),
        "invalid_pattern_count": (
            int(invalid_count) if invalid_count is not None else None
        ),
        "unknown_pattern_count": 0,
        "algorithm_result": result,
        **extra,
    }


def main():
    config = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        result = run_experiment(config)
    json.dump(result, sys.stdout, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
