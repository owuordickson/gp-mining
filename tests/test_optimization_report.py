"""Result summaries must not turn failed or incomparable runs into speedups."""

import csv
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "optimization_report",
    Path(__file__).resolve().parents[1] / "benchmarks/optimization/tools/report.py",
)


def load_report():
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module


def trial(method, seed, elapsed, status="ok", **extra):
    return {
        "dataset": "BCR",
        "experiment": "temporal",
        "method": method,
        "seed": seed,
        "status": status,
        "elapsed_seconds": elapsed,
        "prepared_sha256": "abc",
        "support": 0.7,
        "target": "Glucose",
        **extra,
    }


def test_only_comparable_successful_pairs_contribute_to_speedup():
    report = load_report()
    data = [
        trial("all", 1, 10),
        trial("ami", 1, 2),
        trial("all", 2, 20),
        trial("ami", 2, None, "timeout"),
        trial("ami", 3, 1),
        trial("all", 3, 4, prepared_sha256="different"),
    ]
    paired = report.paired_summaries(data)
    result = next(row for row in paired if row["method"] == "ami")
    assert result["completed_pairs"] == 1
    assert result["speedup_median"] == 5
    assert result["unmatched_or_censored_pairs"] == 2


def test_summary_reports_failures_without_treating_timeouts_as_zero():
    report = load_report()
    data = [
        trial("ami", 0, 4),
        trial("ami", 1, None, "timeout"),
        trial("ami", 2, None, "error"),
    ]
    summary = report.summarize(data)[0]
    assert summary["attempts"] == 3
    assert summary["completed"] == 1
    assert summary["timeout"] == 1
    assert summary["error"] == 1
    assert summary["elapsed_median_seconds"] == 4


def test_summary_csv_includes_validity_totals_and_peak_memory_in_mib(tmp_path):
    def graank_trial(method, seed, valid, invalid, unknown, memory, status="ok"):
        return trial(
            method,
            seed,
            1,
            status,
            experiment="graank",
            valid_pattern_count=valid,
            invalid_pattern_count=invalid,
            unknown_pattern_count=unknown,
            peak_rss_bytes=memory,
        )

    data = [
        graank_trial("ga", 0, 4, 1, 0, 1048576),
        graank_trial("ga", 1, 6, 2, 1, 4194304),
        graank_trial("ga", 2, 99, 99, 99, 99999999, status="timeout"),
        graank_trial("pso", 0, 0, 0, 0, None),
        trial("ami", 0, 1, pattern_count=6),
    ]
    load_report().write_reports(tmp_path, data)
    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = {row["method"]: row for row in csv.DictReader(stream)}

    assert rows["ga"]["valid_pattern_count_total"] == "10"
    assert rows["ga"]["invalid_pattern_count_total"] == "3"
    assert rows["ga"]["unknown_pattern_count_total"] == "1"
    assert float(rows["ga"]["peak_rss_median_bytes"]) == 2621440
    assert float(rows["ga"]["peak_rss_median_mib"]) == 2.5
    assert rows["pso"]["valid_pattern_count_total"] == "0"
    assert rows["pso"]["peak_rss_median_mib"] == ""
    assert rows["ami"]["invalid_pattern_count_total"] == ""


def test_pattern_agreement_uses_items_not_reported_support():
    report = load_report()
    data = [
        trial("all", 0, 8, patterns=[{"items": ["0+", "1-"], "reported_support": 0.7}]),
        trial("ami", 0, 2, patterns=[{"items": ["0+", "1-"], "reported_support": 0.8}]),
    ]
    assert report.paired_summaries(data)[0]["itemset_jaccard_median"] == 1.0


def test_pattern_recovery_excludes_invalid_and_explicitly_unknown_validity():
    report = load_report()
    reference = [
        {"items": ["0+", "1+"], "valid": True},
        {"items": ["0+", "2+"], "valid": True},
        {"items": ["1+", "2+"], "valid": True},
    ]
    heuristic = [
        {"items": ["0+", "1+"], "valid": False},
        {"items": ["0+", "2+"], "valid": None},
        {"items": ["1+", "2+"]},
    ]
    data = [
        {**trial("apriori", 0, 8, patterns=reference), "experiment": "graank"},
        {**trial("aco", 0, 2, patterns=heuristic), "experiment": "graank"},
    ]
    comparison = report.paired_summaries(data)[0]
    assert comparison["itemset_jaccard_median"] == 1 / 3
    assert comparison["apriori_output_recall"] == 1 / 3


def test_temporal_itemset_overlap_retains_unknown_validity():
    report = load_report()
    patterns = [{"items": ["0+", "1+"], "valid": None}]
    data = [
        trial("all", 0, 8, patterns=patterns),
        trial("ami", 0, 2, patterns=patterns),
    ]
    assert report.paired_summaries(data)[0]["itemset_jaccard_median"] == 1


def test_kernel_memory_ratio_is_not_reported_as_representation_memory():
    report = load_report()
    data = [
        {**trial("dense", 0, 8, peak_rss_bytes=80), "experiment": "kernel"},
        {**trial("packed", 0, 2, peak_rss_bytes=40), "experiment": "kernel"},
    ]
    result = report.paired_summaries(data)[0]
    assert result["speedup_median"] == 4
    assert result["memory_ratio_median"] is None


def test_packed_graank_is_paired_against_dense():
    data = [
        trial(
            "dense",
            0,
            8,
            experiment="graank",
            peak_rss_bytes=80,
            patterns=[{"items": ["1+", "2-"]}],
        ),
        trial(
            "packed",
            0,
            2,
            experiment="graank",
            peak_rss_bytes=40,
            patterns=[{"items": ["2+", "1-"]}],
        ),
    ]
    (result,) = load_report().paired_summaries(data)
    assert result["baseline"] == "dense"
    assert result["method"] == "packed"
    assert result["completed_pairs"] == 1
    assert result["speedup_median"] == 4
    assert result["memory_ratio_median"] == 2
    assert result["itemset_jaccard_median"] == result["apriori_output_recall"] == 1


def test_graank_groups_search_types_and_uses_matching_bitmap_baselines():
    data = [
        trial(
            method,
            0,
            elapsed,
            experiment="graank",
            search_type=search_type,
            peak_rss_bytes=memory,
        )
        for search_type, method, elapsed, memory in [
            ("apriori", "dense", 8, 80),
            ("apriori", "packed", 4, 40),
            ("ga", "packed", 2, 50),
            ("pso", "packed", 1, 20),
        ]
    ]
    report = load_report()
    summary = report.summarize(data)
    assert len(summary) == 4
    assert all(row["completed"] == 1 for row in summary)
    paired = {row["search_type"]: row for row in report.paired_summaries(data)}
    assert {
        (search, row["baseline_search_type"], row["baseline"], row["method"])
        for search, row in paired.items()
    } == {
        ("apriori", "apriori", "dense", "packed"),
        ("ga", "apriori", "packed", "packed"),
        ("pso", "apriori", "packed", "packed"),
    }
    assert paired["apriori"]["speedup_median"] == 2
    assert paired["ga"]["speedup_median"] == 2
    assert paired["ga"]["memory_ratio_median"] == 0.8
    assert paired["pso"]["speedup_median"] == 4


@pytest.mark.parametrize("search_type", ["pso", "ga", "hc", "random"])
def test_graank_reports_bitmap_and_search_comparisons_separately(search_type):
    apriori_patterns = [{"items": ["0+", "1-"]}, {"items": ["0+", "2+"]}]
    data = [
        trial(
            method,
            0,
            elapsed,
            experiment="graank",
            search_type=search,
            peak_rss_bytes=memory,
            patterns=patterns,
        )
        for search, method, elapsed, memory, patterns in [
            ("apriori", "dense", 80, 640, apriori_patterns),
            ("apriori", "packed", 16, 256, apriori_patterns),
            (search_type, "dense", 20, 320, [{"items": ["0+", "1-"]}]),
            (search_type, "packed", 2, 64, [{"items": ["1+", "0-"]}]),
        ]
    ]

    comparisons = {
        (
            row["baseline_search_type"],
            row["baseline"],
            row["search_type"],
            row["method"],
        ): (
            row["speedup_median"],
            row["memory_ratio_median"],
            row["itemset_jaccard_median"],
            row["apriori_output_recall"],
        )
        for row in load_report().paired_summaries(data)
    }

    assert comparisons == {
        ("apriori", "dense", "apriori", "packed"): (5, 2.5, 1, 1),
        ("apriori", "dense", search_type, "dense"): (4, 2, 0.5, 0.5),
        ("apriori", "packed", search_type, "packed"): (8, 4, 0.5, 0.5),
        (search_type, "dense", search_type, "packed"): (10, 5, 1, None),
    }


def test_graank_bitmap_pair_retains_censoring_without_apriori():
    data = [
        trial("dense", 0, None, "timeout", experiment="graank", search_type="ga"),
        trial("packed", 0, 2, experiment="graank", search_type="ga"),
        trial("dense", 1, 8, experiment="graank", search_type="ga"),
        trial("packed", 1, 4, experiment="graank", search_type="ga"),
        trial("dense", 2, 9, experiment="graank", search_type="ga"),
    ]
    (result,) = [
        row
        for row in load_report().paired_summaries(data)
        if row["baseline_search_type"] == "ga"
    ]
    assert result["completed_pairs"] == 1
    assert result["unmatched_or_censored_pairs"] == 2
    assert result["speedup_median"] == 2


def test_empty_and_nonempty_reports_are_valid_files(tmp_path):
    report = load_report()
    result = report.write_reports(tmp_path, [])
    assert result == {"summary": [], "paired": []}
    assert not (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.csv").is_file()
    assert not (tmp_path / "report.md").exists()
    data = [trial("all", 0, 8), trial("ami", 0, 2)]
    report.write_reports(tmp_path, data)
    with (tmp_path / "paired_summary.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream))[0]["speedup_median"] == "4.0"
    assert not (tmp_path / "report.md").exists()
    assert not (tmp_path / "summary.json").exists()


def test_search_quality_plot_uses_completed_reference_patterns(tmp_path):
    pytest.importorskip("matplotlib")
    report = load_report()
    data = [
        {
            **trial("apriori", 0, 8, patterns=[{"items": ["0+", "1-"]}]),
            "experiment": "graank",
        },
        {
            **trial("aco", 0, 2, patterns=[{"items": ["0+", "1-"]}]),
            "experiment": "graank",
        },
    ]
    report.write_reports(tmp_path, data)
    assert (tmp_path / "time_vs_pattern_recovery.png").stat().st_size > 0
