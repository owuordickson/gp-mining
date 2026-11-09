"""Content-addressed records for benchmark inputs, source, and results."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys

from .integrity import BENCHMARK_ROOT, artifact_key, artifact_path

SOURCE_FILES = (
    "benchmark.py",
    "tools/tools.py",
    "tools/integrity.py",
    "tools/baseline_methods.py",
    "tools/scoring_tools.py",
    "tools/provenance.py",
    "0_generate_datasets.py",
    "rescore.py",
    "verify_setup.py",
    "requirements.txt",
    "run_pair.py",
    "environment_probe.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with artifact_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    return {"filepath": artifact_key(path), "sha256": sha256(path)}


def environment_record():
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": dict(
            sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
        ),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "mode": "cuda" if torch.cuda.is_available() else "cpu",
    }


def source_record():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=BENCHMARK_ROOT, text=True
    ).strip()
    return {"commit": commit, "files": [file_record(path) for path in SOURCE_FILES]}


def write_run_record(run_path, cfg, command, selection, environment):
    run_path = artifact_path(run_path)
    record = {
        "schema_version": 1,
        "run_id": artifact_key(run_path),
        "status": "executed",
        "command": command,
        "source": source_record(),
        "environment": environment,
        "selection": selection,
        "inputs": {"labels": file_record(cfg.label_path), "timeseries": file_record(cfg.data_path)},
        "artifacts": {
            name: file_record(run_path / name)
            for name in ("config.yaml", "preds.p", "runtime.csv", "scoring.csv")
        },
        "scorer": file_record("tools/scoring_tools.py"),
        "metrics": {
            "AUROC": "pooled off-diagonal",
            "Individual AUROC": "mean per graph, excluding single-class graphs",
        },
    }
    (run_path / "provenance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
