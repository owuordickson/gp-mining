"""Checksum-aware rescoring of recorded predictions without rerunning a method."""

import argparse
import io
import json

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from tools.integrity import artifact_key, artifact_path, secure_load_pickle, verified_bytes
from tools.provenance import environment_record, file_record, source_record
from tools.scoring_tools import score
from tools.tools import graph_to_label_tensor


def rescore(run, output, labels_override=None):
    run = artifact_path(run)
    output = artifact_path(output)
    if output.exists():
        raise FileExistsError(f"Rescore output must have a new identity: {output}")
    record = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    # Verify all recorded inputs/artifacts before any pickle deserialization.
    for group in ("inputs", "artifacts"):
        for name, entry in record[group].items():
            if group == "inputs" and name == "labels" and labels_override is not None:
                # The recovered selection replaces the unavailable archived label file.
                # It is independently manifest-verified below before deserialization.
                continue
            verified_bytes(entry["filepath"], entry["sha256"])
    artifacts = record["artifacts"]
    for name, entry in artifacts.items():
        if artifact_key(entry["filepath"]) != artifact_key(run / name):
            raise ValueError(f"Artifact is not attached to this run: {name}")
    config = OmegaConf.load(
        io.StringIO(
            verified_bytes(
                artifacts["config.yaml"]["filepath"], artifacts["config.yaml"]["sha256"]
            ).decode()
        )
    )
    label_entry = record["inputs"]["labels"]
    if artifact_key(config.label_path) != artifact_key(label_entry["filepath"]):
        raise ValueError("Config and provenance label paths differ")
    if labels_override is not None:
        label_entry = file_record(labels_override)
        # An override must have its own tracked integrity-manifest entry.
        graphs = secure_load_pickle(labels_override)
    else:
        graphs = secure_load_pickle(label_entry["filepath"], label_entry["sha256"])
    if config.restrict_to >= 0:
        graphs = graphs[config.restrict_to : config.restrict_to + 1]
    labels = [graph_to_label_tensor(graph) for graph in graphs]
    pred_entry = artifacts["preds.p"]
    predictions = secure_load_pickle(pred_entry["filepath"], pred_entry["sha256"])
    if labels_override is not None:
        if len(predictions) != len(graphs):
            raise ValueError("Recovered selection size differs from historical predictions")
        for prediction, graph in zip(predictions, graphs, strict=True):
            expected_nodes = [str(node) for node in sorted(graph.nodes)]
            if not isinstance(prediction, pd.DataFrame) or (
                list(map(str, prediction.columns)) != expected_nodes
                or list(map(str, prediction.index)) != expected_nodes
            ):
                raise ValueError(
                    "Recovered labels do not match prediction node identities and order"
                )
    result = score(
        predictions, labels, remove_autoregressive=config.remove_diagonal, name=config.method.name
    )
    previous = pd.read_csv(run / "scoring.csv", index_col=0).iloc[:, 0]
    differences = result.iloc[:, 0] - previous.reindex(result.index)
    output.mkdir(parents=True)
    result.to_csv(output / "scoring.csv")
    differences.rename("rescored_minus_stored").to_csv(output / "score_differences.csv")
    report = {
        "run": artifact_key(run),
        "provenance_scope": record.get("provenance_scope", "Recorded executed run"),
        "scored_labels": label_entry,
        "labels_override": labels_override is not None,
        "run_provenance": file_record(run / "provenance.json"),
        "scorer": file_record("tools/scoring_tools.py"),
        "source": source_record(),
        "environment": environment_record(),
        "pooled_auroc": float(result.loc["AUROC"].iloc[0]),
        "mean_per_graph_auroc": float(result.loc["Individual AUROC"].iloc[0]),
        "max_absolute_stored_score_difference": float(np.abs(differences).max()),
        "artifacts": [
            file_record(output / name) for name in ("scoring.csv", "score_differences.csv")
        ],
    }
    (output / "rescore.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "run",
                    "pooled_auroc",
                    "mean_per_graph_auroc",
                    "max_absolute_stored_score_difference",
                )
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True, help="New benchmark-relative output directory")
    parser.add_argument(
        "--labels", help="Verified reconstructed labels; node order must match predictions"
    )
    args = parser.parse_args()
    rescore(args.run, args.output, args.labels)
