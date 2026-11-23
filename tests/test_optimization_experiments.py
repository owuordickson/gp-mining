import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

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


@pytest.mark.parametrize("method", ["dense", "packed"])
def test_graank_worker_returns_patterns_from_selected_source(tmp_path, method):
    repo = PATH.parents[3]
    config = {
        "experiment": "graank",
        "method": method,
        "data_csv": "a,b,c\n" + "".join(f"{i},{12 - i},{i}\n" for i in range(12)),
        "support": 0.7,
        "target": 0,
    }
    if method == "dense":
        revision = "affc7860397002fba26af96dcb2f0461f2cdda9d"
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
            cwd=repo,
            capture_output=True,
        )
        if exists.returncode:
            pytest.skip("Pinned dense revision is missing from local Git history")
        archive = subprocess.check_output(
            ["git", "archive", revision, "src/so4gp"], cwd=repo
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
            tree.extractall(tmp_path, filter="data")
        config["source_path"] = str(tmp_path / "src")
        expected_source = tmp_path / "src" / "so4gp" / "__init__.py"
    else:
        expected_source = Path(
            subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "from importlib.util import find_spec; "
                    "print(find_spec('so4gp').origin)",
                ],
                text=True,
                env=THREAD_ENV,
            ).strip()
        )
    process = subprocess.run(
        [sys.executable, str(PATH)],
        input=json.dumps(config),
        text=True,
        capture_output=True,
        timeout=90,
        env=THREAD_ENV,
    )
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["algorithm_result"]["Algorithm"] == "GRAANK"
    assert result["valid_pattern_count"] == result["pattern_count"] > 0
    assert result["invalid_pattern_count"] == int(
        result["algorithm_result"]["Invalid Count"]
    )
    assert result["patterns"][0]["reported_support"] == pytest.approx(1.0)
    assert Path(result["source_path"]) == expected_source.resolve()


def test_packed_worker_refuses_a_source_override(tmp_path):
    with pytest.raises(ValueError, match="source_path"):
        module.run_experiment(
            {
                "experiment": "graank",
                "method": "packed",
                "data_csv": EMPTY_SEARCH_CSV,
                "source_path": str(tmp_path),
            }
        )


def test_graank_worker_propagates_invalid_target():
    with pytest.raises(ValueError, match="Target column does not exist"):
        module.run_experiment(
            {
                "experiment": "graank",
                "method": "packed",
                "data_csv": EMPTY_SEARCH_CSV,
                "support": 0.7,
                "target": 2,
            }
        )


@pytest.mark.parametrize(
    ("search_type", "algorithm"),
    [
        ("pso", "PSO-GRAANK"),
        ("ga", "GA-GRAANK"),
        ("hc", "LS-GRAANK"),
        ("random", "RS-GRAANK"),
    ],
)
def test_packed_worker_runs_selected_search_with_its_iteration_budget(
    search_type, algorithm
):
    result = module.run_experiment(
        {
            "experiment": "graank",
            "method": "packed",
            "search_type": search_type,
            "data_csv": "a,b,c\n" + "".join(f"{i},{12 - i},{i}\n" for i in range(12)),
            "support": 0.7,
            "target": 0,
            "iterations": 2,
            "apriori_level": 5,
        }
    )
    assert result["algorithm_result"]["Algorithm"] == algorithm
    assert int(result["algorithm_result"]["Number of iterations"]) == 2


def test_dense_worker_rejects_heuristic_search():
    with pytest.raises(ValueError, match="Dense GRAANK supports only APRIORI"):
        module.run_experiment(
            {
                "experiment": "graank",
                "method": "dense",
                "search_type": "pso",
                "data_csv": EMPTY_SEARCH_CSV,
            }
        )


@pytest.mark.parametrize("search_type", ["apriori", "pso", "ga", "hc", "random"])
def test_graank_worker_completes_when_no_item_pairs_meet_support(search_type):
    result = module.run_experiment(
        {
            "experiment": "graank",
            "method": "packed",
            "search_type": search_type,
            "data_csv": EMPTY_SEARCH_CSV,
            "support": 0.7,
            "target": 0,
        }
    )
    assert result["empty_search_space"] is True
    assert result["valid_pattern_count"] == result["invalid_pattern_count"] == 0
    assert result["patterns"] == []
    assert "Error" not in result["algorithm_result"]
    assert result["elapsed_seconds"] > 0
