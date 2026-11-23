"""Summarize optimization trials without treating censored trials as timings."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

GROUP_FIELDS = ("dataset", "experiment", "support", "target", "rows", "features")
SUMMARY_FIELDS = (*GROUP_FIELDS, "search_type", "method")
PAIR_FIELDS = ("seed", "repeat")
INPUT_FIELDS = ("prepared_sha256",)
ARCHIVED_BASELINES = {
    "graank": (None, "apriori"),
    "temporal": (None, "all"),
    "kernel": (None, "dense"),
    "version": (None, "legacy"),
}
SUCCESS = {"ok", "success"}
PATTERN_COUNT_FIELDS = (
    "valid_pattern_count",
    "invalid_pattern_count",
    "unknown_pattern_count",
)


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _group(record):
    return tuple(record.get(field) for field in GROUP_FIELDS)


def _search_variant(record):
    method = record.get("method")
    search_type = record.get("search_type")
    if (
        search_type is None
        and record.get("experiment") == "graank"
        and method in ("dense", "packed")
    ):
        search_type = "apriori"
    return search_type, method


def _percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _interval(values):
    """Deterministic percentile bootstrap interval for a paired median."""
    if len(values) < 2:
        return None, None
    rng = random.Random(0)
    medians = [
        statistics.median(rng.choices(values, k=len(values))) for _ in range(2000)
    ]
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def summarize(records):
    """Return per-workload, per-method counts and successful-trial medians."""
    groups = defaultdict(list)
    for record in records:
        groups[(*_group(record), *_search_variant(record))].append(record)
    result = []
    for key, trials in groups.items():
        statuses = Counter(str(row.get("status", "error")) for row in trials)
        completed = [row for row in trials if row.get("status") in SUCCESS]
        elapsed = [
            row["elapsed_seconds"]
            for row in completed
            if _finite(row.get("elapsed_seconds")) and row["elapsed_seconds"] >= 0
        ]
        memory = [
            row["peak_rss_bytes"]
            for row in completed
            if _finite(row.get("peak_rss_bytes")) and row["peak_rss_bytes"] >= 0
        ]
        peak_rss_median = statistics.median(memory) if memory else None
        pattern_totals = {}
        for field in PATTERN_COUNT_FIELDS:
            counts = [trial.get(field) for trial in completed]
            pattern_totals[f"{field}_total"] = (
                sum(counts)
                if counts and all(type(count) is int and count >= 0 for count in counts)
                else None
            )
        median = statistics.median(elapsed) if elapsed else None
        row = dict(zip(SUMMARY_FIELDS, key))
        row.update(
            {
                "attempts": len(trials),
                "completed": len(completed),
                "timeout": statuses["timeout"],
                "memory_limit": statuses["memory_limit"],
                "error": statuses["error"],
                "input_error": statuses["input_error"],
                "other_failures": sum(
                    count
                    for status, count in statuses.items()
                    if status
                    not in SUCCESS | {"timeout", "memory_limit", "error", "input_error"}
                ),
                "elapsed_median_seconds": median,
                "elapsed_mad_seconds": statistics.median(
                    abs(value - median) for value in elapsed
                )
                if elapsed
                else None,
                "peak_rss_median_bytes": peak_rss_median,
                "peak_rss_median_mib": peak_rss_median / 1024**2
                if peak_rss_median is not None
                else None,
                **pattern_totals,
            }
        )
        result.append(row)
    return sorted(
        result,
        key=lambda item: tuple(str(item.get(field, "")) for field in SUMMARY_FIELDS),
    )


def _patterns(record):
    patterns = record.get("patterns")
    if patterns is None or not isinstance(patterns, list):
        return None
    bitmap_comparison = record.get("experiment") == "graank" and record.get(
        "method"
    ) in ("dense", "packed")
    identities = []
    for pattern in patterns:
        if isinstance(pattern, dict):
            if pattern.get("valid") is False:
                continue
            if (
                record.get("experiment") == "graank"
                and "valid" in pattern
                and pattern["valid"] is None
            ):
                continue
            if "items" in pattern:
                pattern = pattern["items"]
        if bitmap_comparison and isinstance(pattern, list):
            inverse = [
                item[:-1] + ("-" if item[-1] == "+" else "+") for item in pattern
            ]
            pattern = min(sorted(pattern), sorted(inverse))
        identities.append(
            json.dumps(pattern, sort_keys=True, separators=(",", ":"), default=str)
        )
    return set(identities)


def paired_summaries(records):
    """Compare methods using completed, uniquely identified matching trials only."""
    groups = defaultdict(list)
    for record in records:
        groups[_group(record)].append(record)
    result = []
    for key, trials in groups.items():
        experiment = dict(zip(GROUP_FIELDS, key)).get("experiment")
        variants = defaultdict(list)
        for row in trials:
            variants[_search_variant(row)].append(row)
        for variant, candidate_trials in variants.items():
            search_type, method = variant
            if experiment == "graank" and search_type is not None:
                baseline = (
                    "apriori",
                    "dense" if search_type == "apriori" else "packed",
                )
            else:
                baseline = ARCHIVED_BASELINES.get(experiment)
            if baseline is None or baseline == variant or method is None:
                continue
            candidates = defaultdict(list)
            references = defaultdict(list)
            for row in candidate_trials:
                candidates[tuple(row.get(field) for field in PAIR_FIELDS)].append(row)
            for row in variants.get(baseline, []):
                references[tuple(row.get(field) for field in PAIR_FIELDS)].append(row)
            ratios, memory_ratios, overlaps, recalls = [], [], [], []
            pair_ids = set(candidates) | set(references)
            for pair in pair_ids:
                left, right = references[pair], candidates[pair]
                if len(left) != 1 or len(right) != 1:
                    continue
                ref, current = left[0], right[0]
                if (
                    ref.get("status") not in SUCCESS
                    or current.get("status") not in SUCCESS
                ):
                    continue
                if any(ref.get(field) != current.get(field) for field in INPUT_FIELDS):
                    continue
                old, new = ref.get("elapsed_seconds"), current.get("elapsed_seconds")
                if not (_finite(old) and old >= 0 and _finite(new) and new > 0):
                    continue
                ratios.append(old / new)
                old_mem, new_mem = (
                    ref.get("peak_rss_bytes"),
                    current.get("peak_rss_bytes"),
                )
                if (
                    experiment != "kernel"
                    and _finite(old_mem)
                    and old_mem >= 0
                    and _finite(new_mem)
                    and new_mem > 0
                ):
                    memory_ratios.append(old_mem / new_mem)
                ref_patterns, new_patterns = _patterns(ref), _patterns(current)
                if ref_patterns is not None and new_patterns is not None:
                    union = ref_patterns | new_patterns
                    overlaps.append(
                        len(ref_patterns & new_patterns) / len(union) if union else 1.0
                    )
                    if ref_patterns:
                        recalls.append(
                            len(ref_patterns & new_patterns) / len(ref_patterns)
                        )
            lower, upper = _interval(ratios)
            row = dict(zip(GROUP_FIELDS, key))
            row.update(
                {
                    "baseline_search_type": baseline[0],
                    "baseline": baseline[1],
                    "search_type": search_type,
                    "method": method,
                    "completed_pairs": len(ratios),
                    "unmatched_or_censored_pairs": len(pair_ids) - len(ratios),
                    "speedup_median": statistics.median(ratios) if ratios else None,
                    "speedup_ci_low": lower,
                    "speedup_ci_high": upper,
                    "memory_ratio_median": statistics.median(memory_ratios)
                    if memory_ratios
                    else None,
                    "itemset_jaccard_median": statistics.median(overlaps)
                    if overlaps
                    else None,
                    "apriori_output_recall": statistics.median(recalls)
                    if recalls and experiment == "graank"
                    else None,
                }
            )
            result.append(row)
    return sorted(
        result,
        key=lambda item: tuple(str(item.get(field, "")) for field in SUMMARY_FIELDS),
    )


def _write_csv(path, rows, default_fields):
    fields = list(dict.fromkeys(field for row in rows for field in row)) or list(
        default_fields
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_quality_plot(path, summary, paired):
    points = []
    for comparison in paired:
        recall = comparison.get("apriori_output_recall")
        if not _finite(recall):
            continue
        key = tuple(comparison.get(field) for field in GROUP_FIELDS)
        variant = _search_variant(comparison)
        matching = [
            row
            for row in summary
            if _group(row) == key and _search_variant(row) == variant
        ]
        if matching and _finite(matching[0]["elapsed_median_seconds"]):
            points.append(
                (
                    matching[0]["elapsed_median_seconds"],
                    recall,
                    comparison["dataset"],
                    "/".join(value for value in variant if value is not None),
                )
            )
    if not points:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axis = plt.subplots(figsize=(7, 5))
    for seconds, recall, dataset, method in points:
        axis.scatter(seconds, recall)
        axis.annotate(
            f"{dataset}: {method}",
            (seconds, recall),
            xytext=(4, 4),
            textcoords="offset points",
        )
    axis.set_xlabel("Median elapsed time (seconds)")
    axis.set_ylabel("APRIORI output recall")
    axis.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_reports(output_dir, records):
    """Write machine-readable summaries and an optional quality plot."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(records)
    paired = paired_summaries(records)
    data = {"summary": summary, "paired": paired}
    _write_csv(
        output / "summary.csv",
        summary,
        (
            *SUMMARY_FIELDS,
            "attempts",
            "completed",
            "peak_rss_median_bytes",
            "peak_rss_median_mib",
            *(f"{field}_total" for field in PATTERN_COUNT_FIELDS),
        ),
    )
    _write_csv(
        output / "paired_summary.csv",
        paired,
        (
            *SUMMARY_FIELDS,
            "baseline_search_type",
            "baseline",
            "completed_pairs",
            "speedup_median",
        ),
    )
    _write_quality_plot(output / "time_vs_pattern_recovery.png", summary, paired)
    return data
