"""Checks for isolated benchmark worker execution."""

import importlib.util
import sys
from pathlib import Path


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
