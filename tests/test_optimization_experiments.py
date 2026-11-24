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
EMPTY_SEARCH_CSV = "a,b\n0,0\n0,1\n1,0\n1,1\n"
PATTERN_CSV = "a,b,c\n" + "".join(f"{i},{12 - i},{i}\n" for i in range(12))
spec = importlib.util.spec_from_file_location("optimization_experiments", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
THREAD_ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


@pytest.fixture(scope="session")
def dense_source(tmp_path_factory):
    repo = PATH.parents[3]
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
    directory = tmp_path_factory.mktemp("dense-so4gp")
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        tree.extractall(directory, filter="data")
    return directory / "src"


def _run_worker(config):
    process = subprocess.run(
        [sys.executable, str(PATH)],
        input=json.dumps(config),
        text=True,
        capture_output=True,
        timeout=90,
        env=THREAD_ENV,
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


def _run_graank(config, request):
    if config["method"] == "packed":
        return module.run_experiment(config)
    source = request.getfixturevalue("dense_source")
    result = _run_worker({**config, "source_path": str(source)})
    assert Path(result["source_path"]) == source.resolve() / "so4gp" / "__init__.py"
    return result


@pytest.mark.parametrize("method", ["dense", "packed"])
def test_graank_worker_returns_patterns_from_selected_source(request, method):
    config = {
        "experiment": "graank",
        "method": method,
        "data_csv": PATTERN_CSV,
        "support": 0.7,
        "target": 0,
    }
    if method == "dense":
        source = request.getfixturevalue("dense_source")
        config["source_path"] = str(source)
        expected_source = source / "so4gp" / "__init__.py"
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
    result = _run_worker(config)
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


@pytest.mark.parametrize("method", ["dense", "packed"])
@pytest.mark.parametrize(
    ("search_type", "algorithm"),
    [
        ("pso", "PSO-GRAANK"),
        ("ga", "GA-GRAANK"),
        ("hc", "LS-GRAANK"),
        ("random", "RS-GRAANK"),
    ],
)
def test_graank_worker_runs_selected_search_with_its_iteration_budget(
    request, method, search_type, algorithm
):
    config = {
        "experiment": "graank",
        "method": method,
        "search_type": search_type,
        "data_csv": PATTERN_CSV,
        "support": 0.7,
        "target": 0,
        "iterations": 2,
        "apriori_level": 5,
    }
    result = _run_graank(config, request)
    assert result["algorithm_result"]["Algorithm"] == algorithm
    assert int(result["algorithm_result"]["Number of iterations"]) == 2


@pytest.mark.parametrize("method", ["dense", "packed"])
@pytest.mark.parametrize("search_type", ["pso", "ga"])
def test_graank_worker_handles_wide_candidate_positions(request, method, search_type):
    columns = range(64)
    data_csv = "\n".join(
        [",".join(f"col{i}" for i in columns)]
        + [",".join(str(row + i) for i in columns) for row in range(24)]
    )
    result = _run_graank(
        {
            "experiment": "graank",
            "method": method,
            "search_type": search_type,
            "data_csv": data_csv,
            "support": 0.7,
            "target": 0,
            "iterations": 2,
        },
        request,
    )
    assert int(result["algorithm_result"]["Number of iterations"]) == 2
    assert result["invalid_pattern_count"] is not None


@pytest.mark.parametrize("method", ["dense", "packed"])
@pytest.mark.parametrize("search_type", ["apriori", "pso", "ga", "hc", "random"])
def test_graank_worker_completes_when_no_item_pairs_meet_support(
    request, method, search_type
):
    config = {
        "experiment": "graank",
        "method": method,
        "search_type": search_type,
        "data_csv": EMPTY_SEARCH_CSV,
        "support": 0.7,
        "target": 0,
    }
    result = _run_graank(config, request)
    assert result["empty_search_space"] is True
    assert result["valid_pattern_count"] == result["invalid_pattern_count"] == 0
    assert result["patterns"] == []
    assert "Error" not in result["algorithm_result"]
    assert result["elapsed_seconds"] > 0
