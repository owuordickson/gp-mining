"""Execute only the agreed confounder_3 T-GRAANK/VAR pair with frozen inputs."""

import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from tools.integrity import BENCHMARK_ROOT, MANIFEST_PATH, secure_load_pickle
from tools.provenance import environment_record, file_record, source_record


def main():
    evidence = BENCHMARK_ROOT / "evidence/review-232"
    frozen = evidence / "frozen/confounder_3/east.p"
    if (evidence / "pair.json").exists():
        raise FileExistsError(
            "This evidence identity already has a pair; choose a new identity for a rerun"
        )
    graphs = secure_load_pickle("datasets/confounder_3/east.p")
    regenerated = secure_load_pickle("evidence/review-232/generated/confounder_3/east.p")
    original_topology = [(sorted(g.nodes), sorted(g.edges)) for g in graphs]
    generated_topology = [(sorted(g.nodes), sorted(g.edges)) for g in regenerated]
    if original_topology != generated_topology:
        raise ValueError("Generated and frozen confounder selections differ")
    frozen.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BENCHMARK_ROOT / "datasets/confounder_3/east.p", frozen)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    frozen_record = file_record(frozen)
    matches = [
        entry
        for entry in manifest["data_runs"]
        if entry.get("filepath") == frozen_record["filepath"]
    ]
    if matches:
        if len(matches) != 1 or matches[0]["sha256"] != frozen_record["sha256"]:
            raise ValueError("Frozen pair identity has an inconsistent or ambiguous manifest entry")
    else:
        manifest["data_runs"].append(
            {
                **frozen_record,
                "status": "frozen-byte-copy",
                "source": "datasets/confounder_3/east.p",
            }
        )
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    environment = environment_record()
    if environment["mode"] != "cpu":
        raise ValueError("This evidence profile is explicitly CPU; use a separate identity for GPU")
    records = []
    for method in ("var", "tgraank"):
        command = [
            sys.executable,
            "benchmark.py",
            f"method={method}",
            "label_path=evidence/review-232/frozen/confounder_3/east.p",
            "data_path=product/rivers_ts_east_germany.csv",
            "restrict_to=-1",
            "dt_preprocess=True",
            "data_preprocess.resolution=6h",
            "data_preprocess.normalize=True",
            "data_preprocess.interpolate=True",
            "data_preprocess.subsample=1",
            "data_preprocess.subset_year=False",
            "data_preprocess.subset_month=False",
            "data_preprocess.remove_trailing_nans_early=False",
            "method.max_lag=3",
            "method.var_absolute_values=True",
            "method.map_to_summary_graph=max",
            "remove_diagonal=True",
            "save_full_out=True",
        ]
        before = set((BENCHMARK_ROOT / f"results/{method}_confounder_3").glob("*/provenance.json"))
        print(
            f"Executing {method}: all {len(graphs)} frozen observable-confounder graphs, CPU",
            flush=True,
        )
        result = subprocess.run(command, cwd=BENCHMARK_ROOT, capture_output=True, text=True, check=False)
        (evidence / f"{method}-run.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            print(result.stdout + result.stderr, flush=True)
            raise RuntimeError(f"{method} failed with exit {result.returncode}")
        after = set((BENCHMARK_ROOT / f"results/{method}_confounder_3").glob("*/provenance.json"))
        created = after - before
        if len(created) != 1:
            raise ValueError("Expected exactly one new attached run record")
        path = created.pop()
        record = json.loads(path.read_text(encoding="utf-8"))
        record["command"] = [".venv_review/Scripts/python.exe", *command[1:]]
        record["environment_record"] = file_record(evidence / "environment.json")
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        scoring = pd.read_csv(path.parent / "scoring.csv", index_col=0).iloc[:, 0]
        record_summary = {
            "run_id": record["run_id"],
            "record": file_record(path),
            "pooled_auroc": float(scoring.loc["AUROC"]),
            "mean_per_graph_auroc": float(scoring.loc["Individual AUROC"]),
            "runtime_seconds": float(pd.read_csv(path.parent / "runtime.csv")["runtime"].iloc[0]),
        }
        records.append(record_summary)
        print(json.dumps(record_summary, indent=2), flush=True)
    left, right = [
        json.loads((BENCHMARK_ROOT / rec["record"]["filepath"]).read_text()) for rec in records
    ]
    if left["inputs"] != right["inputs"] or left["selection"] != right["selection"]:
        raise ValueError("The pair did not use identical frozen inputs and sample selection")
    for rec in records:
        predictions = secure_load_pickle(
            json.loads((BENCHMARK_ROOT / rec["record"]["filepath"]).read_text())["artifacts"][
                "preds.p"
            ]["filepath"],
            json.loads((BENCHMARK_ROOT / rec["record"]["filepath"]).read_text())["artifacts"][
                "preds.p"
            ]["sha256"],
        )
        if not np.isfinite(np.asarray(predictions)).all():
            raise ValueError("Pair predictions must be finite")
    pair = {
        "scope": "One confounder_3 river-network graph-recovery comparison; no forecast/superiority claim",
        "command": ".venv_review/Scripts/python.exe run_pair.py",
        "source": source_record(),
        "environment": file_record(evidence / "environment.json"),
        "generation_matches_frozen_topology_and_order": True,
        "sample_count": len(graphs),
        "shared_inputs": left["inputs"],
        "shared_selection": left["selection"],
        "runs": records,
    }
    (evidence / "pair.json").write_text(json.dumps(pair, indent=2) + "\n", encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["benchmark_runs"].extend(
        {"run_id": rec["run_id"], "status": "executed", "record": rec["record"]} for rec in records
    )
    manifest["paired_comparison"] = file_record(evidence / "pair.json")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        "Verified: both methods used identical input hashes, ordered graphs, time windows, and preprocessing.",
        flush=True,
    )


if __name__ == "__main__":
    main()
