"""Actual imports, versions, and compute mode for the supported pip profile."""

import importlib
import json

from tools.provenance import BENCHMARK_ROOT, environment_record

if __name__ == "__main__":
    modules = (
        "numpy",
        "pandas",
        "torch",
        "so4gp",
        "statsmodels.tsa.api",
        "hydra",
        "omegaconf",
        "networkx",
        "sklearn",
        "matplotlib",
        "tabulate",
    )
    for name in modules:
        importlib.import_module(name)
        print(f"import {name}: OK", flush=True)
    record = environment_record()
    output = BENCHMARK_ROOT / "evidence/review-232/environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
