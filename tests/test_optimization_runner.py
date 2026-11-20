"""Checks for isolated benchmark worker execution."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def load_runner():
    path = Path(__file__).resolve().parents[1] / "benchmarks/optimization/benchmark.py"
    spec = importlib.util.spec_from_file_location("optimization_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_timeout_is_not_a_success(tmp_path):
    runner = load_runner()
    script = tmp_path / "slow.py"
    script.write_text("import time; time.sleep(10)", encoding="utf-8")
    result = runner.run_child({}, timeout=0.1, command=[sys.executable, str(script)])
    assert result["status"] == "timeout"
    assert "elapsed_seconds" not in result


def test_worker_error_preserves_diagnostic(tmp_path):
    runner = load_runner()
    script = tmp_path / "bad.py"
    script.write_text("raise ValueError('bad input')", encoding="utf-8")
    result = runner.run_child({}, timeout=5, command=[sys.executable, str(script)])
    assert result["status"] == "error"
    assert "bad input" in result["error"]


@pytest.mark.parametrize("launch", ["script", "module"])
def test_run_uses_data_directory_without_writing_prepared_files(tmp_path, launch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source = data_dir / "breast_cancer.csv"
    source.write_text(
        "Age,Glucose,Classification\n20,80,1\n30,90,2\n40,70,1\n50,100,2\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    output = tmp_path / "results"
    runner = (
        Path(__file__).resolve().parents[1] / "benchmarks/optimization/benchmark.py"
    )
    if launch == "script":
        command = [sys.executable, str(runner)]
        cwd = runner.parent  # PyCharm uses the script's directory by default.
    else:
        command = [sys.executable, "-m", "benchmarks.optimization.benchmark"]
        cwd = runner.parents[2]
    result = subprocess.run(
        [
            *command,
            "--profile",
            "quick",
            "--datasets",
            "BCR",
            "--experiment",
            "kernel",
            "--data-dir",
            str(data_dir),
            "--output",
            str(output),
            "--rows",
            "4",
            "--features",
            "2",
            "--warmups",
            "0",
            "--timeout",
            "30",
        ],
        text=True,
        capture_output=True,
        timeout=90,
        cwd=cwd,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert source.read_bytes() == original
    assert sorted(path.name for path in data_dir.iterdir()) == ["breast_cancer.csv"]
    assert not list(output.glob("*-prepared.csv"))
    assert not (output / "report.md").exists()
    assert (output / "trials.jsonl").is_file()
