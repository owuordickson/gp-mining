"""Checks for isolated benchmark worker execution."""

import csv
import importlib.util
import json
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
    benchmark = load_runner()
    revision = subprocess.run(
        ["git", "cat-file", "-e", f"{benchmark.DENSE_REVISION}^{{commit}}"],
        cwd=benchmark.ROOT,
        capture_output=True,
    )
    if revision.returncode:
        pytest.skip("Pinned dense revision is missing from local Git history")

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
            "graank",
            "--search-types",
            "apriori",
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
            "60",
        ],
        text=True,
        capture_output=True,
        timeout=150,
        cwd=cwd,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert source.read_bytes() == original
    assert sorted(path.name for path in data_dir.iterdir()) == ["breast_cancer.csv"]
    assert not list(output.glob("*-prepared.csv"))
    assert not (output / "report.md").exists()
    records = [
        json.loads(line)
        for line in (output / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert {record["method"] for record in records} == {"dense", "packed"}
    assert {record["search_type"] for record in records} == {"apriori"}
    assert len({record["source_path"] for record in records}) == 2
    assert len({record["prepared_sha256"] for record in records}) == 1


def test_reports_refresh_at_dataset_boundaries_and_on_resume(tmp_path, monkeypatch):
    runner = load_runner()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "breast_cancer.csv").write_text(
        "Age,Glucose\n20,80\n30,90\n40,70\n50,100\n", encoding="utf-8"
    )
    (data_dir / "air_quality.csv").write_text(
        "Date,Time,CO(GT),T\n3/10/2004,00:00:00,2,11\n3/11/2004,00:00:00,3,12\n",
        encoding="utf-8",
    )
    output = tmp_path / "results"
    summary = output / "summary.csv"
    args = [
        "--profile",
        "quick",
        "--datasets",
        "BCR",
        "AQY",
        "--data-dir",
        str(data_dir),
        "--output",
        str(output),
        "--warmups",
        "0",
    ]

    def summarized_trials():
        with summary.open(encoding="utf-8", newline="") as stream:
            return {
                (row["dataset"], row["search_type"], row["method"])
                for row in csv.DictReader(stream)
            }

    attempts = []
    variants = {
        ("apriori", "dense"),
        ("apriori", "packed"),
        ("pso", "packed"),
        ("ga", "packed"),
        ("hc", "packed"),
        ("random", "packed"),
    }
    bcr_trials = {("BCR", *variant) for variant in variants}
    first_aqy = len(variants)

    def pause_during_second_dataset(config, *args, **kwargs):
        variant = (config.get("search_type", "apriori"), config["method"])
        attempts.append(variant)
        assert ("source_path" in config) == (variant == ("apriori", "dense"))
        if len(attempts) == first_aqy + 2:
            raise KeyboardInterrupt
        if len(attempts) == first_aqy + 3:
            assert summarized_trials() == bcr_trials | {("AQY", *attempts[first_aqy])}
        return {"status": "ok"}

    monkeypatch.setattr(runner, "_dense_source", lambda: str(tmp_path))
    monkeypatch.setattr(runner, "run_child", pause_during_second_dataset)
    with pytest.raises(KeyboardInterrupt):
        runner.main(args)

    assert summarized_trials() == bcr_trials
    assert (output / "paired_summary.csv").is_file()
    summary.unlink()  # Simulate a summary lost during an interruption.

    assert runner.main([*args, "--resume"]) == 0
    assert len(attempts) == 2 * len(variants) + 1
    assert attempts[first_aqy + 1] == attempts[first_aqy + 2]
    assert summarized_trials() == bcr_trials | {
        ("AQY", *variant) for variant in variants
    }
    records = [
        json.loads(line)
        for line in (output / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len({record["trial_id"] for record in records}) == len(records) == 12
