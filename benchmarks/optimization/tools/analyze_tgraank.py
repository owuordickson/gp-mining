"""Score the rainfall/EVI trials with the temporal performance notebook's protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from so4gp import GI, TGP, TimeDelay

METHODS = {
    "all": "TGRAANK",
    "ami": "TGRAANK-AMI",
    "random": "TGRAANK-AMI (Random)",
}
COUNTS = ("tp", "fp", "fn", "tn")
METRICS = ("accuracy", "precision", "recall", "specificity", "f1")
SUPPORT_THRESHOLD = 0.5


def _restore_pattern(record):
    pattern = TGP()
    pattern.target_gradual_item = GI.from_string(record["items"][0])
    pattern.support = record["reported_support"]
    for item in record["temporal_items"]:
        pattern.add_temporal_gradual_item(
            GI.from_string(item["item"]),
            TimeDelay(item["lag_seconds"], item["lag_support"]),
        )
    return pattern


def confusion_counts(rainfall, evi):
    """Match FTGPs once in mining order using the notebook's similarity rule."""
    counts = dict.fromkeys(COUNTS, 0)
    labels = {
        (True, True): "tp",
        (True, False): "fp",
        (False, True): "fn",
        (False, False): "tn",
    }
    unmatched_evi = [_restore_pattern(record) for record in evi]
    for record in rainfall:
        pattern = _restore_pattern(record)
        match = next(
            (
                index
                for index, candidate in enumerate(unmatched_evi)
                if pattern.is_similar_to(candidate)
            ),
            None,
        )
        evi_support = unmatched_evi.pop(match).support if match is not None else 0
        rainfall_hit = pattern.support >= SUPPORT_THRESHOLD
        evi_hit = evi_support >= SUPPORT_THRESHOLD
        counts[labels[rainfall_hit, evi_hit]] += 1
    for pattern in unmatched_evi:
        counts[labels[False, pattern.support >= SUPPORT_THRESHOLD]] += 1
    return counts


def score(records):
    """Return per-site counts and scores plus their means and sample deviations."""
    trials = {}
    targets = set()
    for trial in records:
        if trial.get("experiment") != "temporal":
            continue
        seed, site = trial["seed"], trial["target_site"]
        key = seed, site, trial["method"], trial["dataset"]
        if trial["status"] != "ok" or key in trials:
            raise ValueError(f"Scoring requires one successful trial for {key}")
        trials[key] = trial
        targets.update((seed, name.split(" (")[0]) for name in trial["columns"][1:])
    if not trials:
        raise ValueError("No temporal trial records found")

    rows = []
    for seed, site in sorted(targets):
        for method, algorithm in METHODS.items():
            rain = trials.get((seed, site, method, "KE_RAIN"))
            evi = trials.get((seed, site, method, "KE_EVI"))
            if rain is None or evi is None:
                raise ValueError(
                    f"Missing rainfall/EVI pair for {algorithm}, {site}, {seed}"
                )
            if method != "all" and (
                not rain["transformation_steps"]
                or rain["transformation_steps"] != evi["transformation_steps"]
            ):
                raise ValueError(
                    f"EVI did not reuse rainfall {method} steps for {site}, {seed}"
                )
            rows.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "site": site,
                    **confusion_counts(rain["patterns"], evi["patterns"]),
                }
            )

    scores = pd.DataFrame(rows).sort_values(["algorithm", "seed", "site"])
    tp, fp, fn, tn = (scores[name] for name in COUNTS)
    scores["accuracy"] = (tp + tn) / (tp + fp + fn + tn)
    scores["precision"] = tp / (tp + fp)
    scores["recall"] = tp / (tp + fn)
    scores["specificity"] = tn / (tn + fp)
    scores["f1"] = (
        2
        * scores["precision"]
        * scores["recall"]
        / (scores["precision"] + scores["recall"])
    )
    summary = scores.groupby("algorithm")[list(COUNTS + METRICS)].agg(["mean", "std"])
    summary.columns = [f"{field}_{stat}" for field, stat in summary.columns]
    return scores, summary.reset_index()


def _plot_confusion_matrix(summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_methods = len(summary)
    fig, axes = plt.subplots(
        1, n_methods, figsize=(4 * n_methods, 4), layout="constrained"
    )
    fig.suptitle("Mean confusion counts per site", fontsize=14)
    limit = summary[[f"{name}_mean" for name in COUNTS]].max().max()
    for axis, row in zip(axes, summary.itertuples(index=False), strict=True):
        counts = [[row.tp_mean, row.fn_mean], [row.fp_mean, row.tn_mean]]
        heatmap = axis.imshow(counts, cmap="Blues", vmin=0, vmax=max(1, limit))
        for y, values in enumerate(counts):
            for x, value in enumerate(values):
                color = "white" if value > limit * 0.6 else "#102a43"
                axis.text(x, y, f"{value:.2f}", ha="center", va="center", color=color)
        axis.set_title(row.algorithm)
        axis.set_xticks([0, 1], ["Positive", "Negative"])
        axis.set_yticks([0, 1], ["Positive", "Negative"])
        axis.set_xlabel("Rainfall prediction")
        axis.set_ylabel("EVI reference")
    fig.colorbar(heatmap, ax=axes, label="Mean pattern count", shrink=0.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "tgraank-mi",
    )
    output = parser.parse_args(argv).results
    with (output / "trials.jsonl").open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    scores, summary = score(records)
    scores.to_csv(output / "confusion_matrix.csv", index=False)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    _plot_confusion_matrix(summary, output / "confusion_matrix.png")


if __name__ == "__main__":
    main()
