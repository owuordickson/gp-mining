"""Isolated workers for the optimization benchmark.

The runner sends one JSON object on standard input and receives one JSON result.
Source selection happens before importing SO4GP, so historical and current
packages must always run in different interpreter processes.
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


def canonical_pattern(items):
    """Normalize ordering and global sign reversal of gradual items."""
    normal = tuple(sorted(str(item) for item in items))
    swapped = tuple(
        sorted(item[:-1] + ("-" if item[-1] == "+" else "+") for item in normal)
    )
    return min(normal, swapped)


def search_iteration_limit(config, method):
    """APRIORI depth and stochastic iteration counts have different meanings."""
    return (
        config.get("apriori_level") if method == "apriori" else config.get("iterations")
    )


def _kernel(data, repeats, method):
    numeric = data.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2 or len(numeric) < 2:
        raise ValueError("kernel requires two numeric columns and two rows")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if method not in ("dense", "packed"):
        raise ValueError(f"unknown kernel method: {method}")
    values = numeric.iloc[:, :2].to_numpy(dtype=float)
    n = len(values)
    start = time.perf_counter()
    left = values[:, 0, None] < values[None, :, 0]
    right = values[:, 1, None] < values[None, :, 1]
    dense_build = time.perf_counter() - start
    start = time.perf_counter()
    left_packed = np.packbits(left.ravel())
    right_packed = np.packbits(right.ravel())
    pack_seconds = time.perf_counter() - start
    # Validate one intersection of each representation outside the measured loop.
    dense = np.logical_and(left, right)
    packed = np.bitwise_and(left_packed, right_packed)
    reconstructed = np.unpackbits(packed, count=n * n).reshape(n, n).astype(bool)
    dense_count = int(np.count_nonzero(dense))
    packed_count = int(np.count_nonzero(reconstructed))
    if dense_count != packed_count or not np.array_equal(dense, reconstructed):
        raise ValueError("dense and packed intersections do not agree")
    del dense, packed, reconstructed
    start = time.perf_counter()
    if method == "dense":
        for _ in range(repeats):
            dense = np.logical_and(left, right)
            np.count_nonzero(dense)
    else:
        for _ in range(repeats):
            packed = np.bitwise_and(left_packed, right_packed)
            if hasattr(np, "bitwise_count"):
                np.bitwise_count(packed).sum(dtype=np.uint64)
            else:
                np.unpackbits(packed).sum(dtype=np.uint64)
    elapsed = time.perf_counter() - start
    pair_count = n * (n - 1) / 2
    return {
        "equal": True,
        "dense_count": dense_count,
        "packed_count": packed_count,
        "intersection_support": dense_count / pair_count,
        "dense_bytes": int(left.nbytes + right.nbytes),
        "packed_bytes": int(left_packed.nbytes + right_packed.nbytes),
        "dense_build_seconds": dense_build,
        "pack_seconds": pack_seconds,
        f"{method}_intersection_seconds": elapsed,
        "repeats": repeats,
    }


def _select_source(source_path):
    path = (
        Path(source_path).resolve()
        if source_path
        else Path(__file__).resolve().parents[3] / "src"
    )
    if not (path / "so4gp" / "__init__.py").is_file():
        raise ValueError(f"no SO4GP package at {path}")
    if "so4gp" in sys.modules:
        actual = Path(sys.modules["so4gp"].__file__).resolve()
        if not actual.is_relative_to(path):
            raise RuntimeError("SO4GP was already imported from a different source")
    sys.path.insert(0, str(path))
    import so4gp

    actual = Path(so4gp.__file__).resolve()
    if not actual.is_relative_to(path):
        raise RuntimeError(f"incorrect SO4GP import: {actual}")
    return str(actual)


def _mining_engine(miner):
    """Read the public engine property in either SO4GP revision."""
    engine = getattr(miner, "mine_engine", None)
    return engine if engine is not None else miner.mining_engine


def _patterns(engine, data, min_support):
    """Collect canonical patterns and recompute strict full support on input."""
    result = []
    n = len(data)
    for pattern in engine.gradual_patterns or []:
        raw = [str(item.to_string()) for item in pattern.gradual_items]
        items = canonical_pattern(raw)
        entry = {"items": list(items), "reported_support": float(pattern.support)}
        try:
            comparisons = np.ones((n, n), dtype=bool)
            for item in raw:
                column, sign = int(item[:-1]), item[-1]
                values = pd.to_numeric(data.iloc[:, column]).to_numpy(dtype=float)
                comparisons &= (
                    (values[:, None] < values[None, :])
                    if sign == "+"
                    else (values[:, None] > values[None, :])
                )
            entry["recomputed_support"] = int(comparisons.sum()) / (n * (n - 1) / 2)
            entry["valid"] = entry["recomputed_support"] + 1e-12 >= min_support
        except ValueError, IndexError, TypeError:
            entry["recomputed_support"] = None
            entry["valid"] = None
        result.append(entry)
    return result


def run_experiment(config: dict) -> dict:
    """Execute one requested experiment and return a JSON-serializable record."""
    if "data_csv" in config:
        data = pd.read_csv(io.StringIO(config["data_csv"]))
    else:
        data = pd.read_csv(config["data_path"])
    experiment = config["experiment"]
    if experiment == "kernel":
        method = config.get("method", "dense")
        result = _kernel(data, int(config.get("repeats", 10)), method)
        result["elapsed_seconds"] = result[f"{method}_intersection_seconds"]
        return result

    source = _select_source(config.get("source_path"))
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    support = float(config.get("support", 0.5))
    method = config.get("method", "apriori")

    if experiment == "graank":
        from so4gp.algorithms.graank import GRAANK

        start = time.perf_counter()
        miner = GRAANK(data.copy(), min_sup=support, eq=False)
        result_json = miner.discover(
            search_type=method,
            max_iteration=search_iteration_limit(config, method),
            target_col=config.get("target"),
            compute_descriptors=False,
        )
        elapsed = time.perf_counter() - start
        engine = _mining_engine(miner)
        extra = {}
    else:
        raise ValueError(f"unknown experiment: {experiment}")
    result = json.loads(result_json)
    if "Error" in result:
        error = result["Error"]
        if (
            experiment == "graank"
            and engine.valid_bins is None
            and error in ("Search space is empty!", "Pairwise matrices not available!")
        ):
            # Too few gradual items passed support filtering to form a pattern.
            extra["empty_search_space"] = True
            result["Empty search reason"] = result.pop("Error")
        else:
            raise RuntimeError(str(error))
    patterns = _patterns(engine, data, support)
    if experiment == "graank":
        extra.update(
            {
                "valid_pattern_count": sum(p.get("valid") is True for p in patterns),
                "invalid_pattern_count": sum(p.get("valid") is False for p in patterns),
                "unknown_pattern_count": sum(p.get("valid") is None for p in patterns),
            }
        )
    return {
        "elapsed_seconds": elapsed,
        "source_path": source,
        "patterns": patterns,
        "pattern_count": len(patterns),
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
