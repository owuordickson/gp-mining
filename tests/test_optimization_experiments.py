import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks/optimization/tools/experiments.py"
)
EMPTY_SEARCH_CSV = "a,b\n2,3\n1,4\n4,1\n3,2\n"
spec = importlib.util.spec_from_file_location("optimization_experiments", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
THREAD_ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def test_worker_accepts_in_memory_csv_via_stdin():
    config = {
        "experiment": "kernel",
        "method": "packed",
        "data_csv": "time,a,b\n2020-01-01,1,2\n2020-01-02,3,1\n2020-01-03,2,3\n",
    }
    process = subprocess.run(
        [sys.executable, str(PATH)],
        input=json.dumps(config),
        text=True,
        capture_output=True,
        check=True,
        env=THREAD_ENV,
    )
    result = json.loads(process.stdout)
    assert result["equal"] is True
    assert result["dense_count"] == result["packed_count"] == 1
    assert result["packed_bytes"] == 4


def test_kernel_agrees_with_dense_for_partial_byte(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 3, 2], "b": [2, 1, 3]}).to_csv(path, index=False)
    result = module.run_experiment({"experiment": "kernel", "data_path": str(path)})
    assert result["equal"] is True
    assert result["dense_count"] == result["packed_count"] == 1
    assert result["intersection_support"] == pytest.approx(1 / 3)
    assert result["dense_bytes"] == 18
    assert result["packed_bytes"] == 4


@pytest.mark.parametrize("method,other", [("dense", "packed"), ("packed", "dense")])
def test_kernel_reports_only_requested_timing(tmp_path, method, other):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 3, 2], "b": [2, 1, 3]}).to_csv(path, index=False)
    result = module.run_experiment(
        {"experiment": "kernel", "data_path": str(path), "method": method, "repeats": 3}
    )
    assert result["elapsed_seconds"] == result[f"{method}_intersection_seconds"]
    assert f"{other}_intersection_seconds" not in result


def test_kernel_rejects_incorrect_packed_intersection(tmp_path, monkeypatch):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 3, 2], "b": [2, 1, 3]}).to_csv(path, index=False)
    original_packbits = np.packbits
    monkeypatch.setattr(
        np,
        "packbits",
        lambda *args, **kwargs: np.zeros_like(original_packbits(*args, **kwargs)),
    )
    with pytest.raises(ValueError, match="intersection"):
        module.run_experiment(
            {"experiment": "kernel", "data_path": str(path), "method": "packed"}
        )


def test_pattern_validity_uses_recomputed_support():
    from so4gp.gradual_patterns import GI, GP

    def pattern(items, reported):
        gp = GP()
        for col, sign in items:
            gp.add_gradual_item(GI(col, sign))
        gp.support = reported
        return gp

    engine = SimpleNamespace(
        gradual_patterns=[
            pattern([(0, "+"), (1, "+")], 0.01),
            pattern([(0, "+"), (1, "-")], 1.0),
            pattern([(0, "+"), (5, "+")], 1.0),
        ]
    )
    rows = module._patterns(engine, pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}), 0.7)
    assert [row["valid"] for row in rows] == [True, False, None]


def test_canonical_global_inverse():
    assert module.canonical_pattern(["1+", "2-"]) == module.canonical_pattern(
        ["1-", "2+"]
    )
    assert module.canonical_pattern(["2-", "1+"]) == ("1+", "2-")


def test_max_lag_mapping():
    for n, lag in [(10, 2), (64, 10), (12, 1)]:
        rep = module.representativity_for_lag(n, lag)
        assert n - int(rep * n) - 1 == lag
    with pytest.raises(ValueError):
        module.representativity_for_lag(5, 4)


def test_apriori_depth_is_not_the_heuristic_iteration_budget():
    assert module.search_iteration_limit({"iterations": 5}, "apriori") is None
    assert (
        module.search_iteration_limit({"iterations": 5, "apriori_level": 2}, "apriori")
        == 2
    )
    assert module.search_iteration_limit({"iterations": 5}, "aco") == 5


def test_current_graank_worker_returns_parseable_result(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": range(12), "b": range(12, 0, -1), "c": range(12)}).to_csv(
        path, index=False
    )
    config = {
        "experiment": "graank",
        "data_path": str(path),
        "support": 0.7,
        "method": "apriori",
        "target": 0,
    }
    process = subprocess.run(
        [sys.executable, str(PATH)],
        input=json.dumps(config),
        text=True,
        capture_output=True,
        check=True,
        env=THREAD_ENV,
    )
    result = json.loads(process.stdout)
    assert result["pattern_count"] > 0
    assert result["patterns"][0]["recomputed_support"] == pytest.approx(1.0)
    assert result["source_path"].endswith("so4gp\\__init__.py") or result[
        "source_path"
    ].endswith("so4gp/__init__.py")


@pytest.mark.parametrize("method", ["ga", "pso"])
def test_graank_serializes_missing_bitmap_diagnostic(method):
    from io import StringIO

    from so4gp.algorithms.graank import GRAANK

    miner = GRAANK(pd.read_csv(StringIO(EMPTY_SEARCH_CSV)), min_sup=0.7)
    result = json.loads(miner.discover(search_type=method, compute_descriptors=False))
    assert result["Error"] == "Pairwise matrices not available!"
    assert result["Patterns"] == []


@pytest.mark.parametrize("method", ["apriori", "aco", "ga", "pso", "hc", "random"])
def test_graank_worker_completes_when_no_item_pairs_meet_support(method):
    result = module.run_experiment(
        {
            "experiment": "graank",
            "method": method,
            "data_csv": EMPTY_SEARCH_CSV,
            "support": 0.7,
            "target": 0,
        }
    )
    assert result["empty_search_space"] is True
    assert result["pattern_count"] == 0
    assert result["patterns"] == []
    assert "Error" not in result["algorithm_result"]
    assert result["elapsed_seconds"] > 0


@pytest.mark.parametrize("method", ["all", "ami", "random"])
def test_temporal_worker_accepts_lag_and_reports_method(tmp_path, method):
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=12),
            "a": range(12),
            "b": range(12, 0, -1),
            "c": range(12),
        }
    ).to_csv(path, index=False)
    config = {
        "experiment": "temporal",
        "method": method,
        "data_path": str(path),
        "support": 0.7,
        "target": 1,
        "lag": 2,
        "seed": 17,
    }
    process = subprocess.run(
        [sys.executable, str(PATH)],
        input=json.dumps(config),
        text=True,
        capture_output=True,
        check=True,
        timeout=90,
        env=THREAD_ENV,
    )
    result = json.loads(process.stdout)
    assert result["eligible_lags"] == [1, 2]
    assert result["evaluation_mode"] is False
    if method == "random":
        assert result["transformation_steps"]
    elif method == "ami":
        assert result["transformation_steps"] is None
