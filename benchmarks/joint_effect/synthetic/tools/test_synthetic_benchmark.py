import numpy as np
import pytest


def test_baseline_preserves_raw_values_in_one_row_per_recording():
    """Keep raw readings in observation/channel order, including strided inputs."""
    from benchmarks.joint_effect.uea.tools.features import build_features

    recordings = np.array(
        [
            [[1, 10], [2, 20], [3, 30]],
            [[4, 40], [5, 50], [6, 60]],
        ],
        dtype=np.float32,
    )[:, ::-1, :]
    original = recordings.copy()

    features = build_features(recordings, max_patterns=2, min_support=0).matrices
    np.testing.assert_array_equal(
        features["baseline"],
        [[3, 30, 2, 20, 1, 10], [6, 60, 5, 50, 4, 40]],
    )
    np.testing.assert_array_equal(recordings, original)


@pytest.mark.parametrize(
    "method,expected",
    [
        (
            "baseline_gcs",
            [
                1,
                1,
                2,
                2,
                4,
                4,
                3,
                0,
                0.5,
                2,
                2 / 3,
                2 / 9,
                4 / 3,
                1 / 3,
                2,
                1,
                0.25,
                2.5,
            ],
        ),
        (
            "baseline_standard",
            [
                1,
                1,
                2,
                2,
                4,
                4,
                3,
                0,
                2.5,
                1.75,
                np.sqrt(1.25),
                np.sqrt(2.1875),
                2.4,
                -0.3,
                3.5 / np.sqrt(43.75),
            ],
        ),
    ],
)
def test_baseline_combinations_append_features_to_the_raw_recording(method, expected):
    """Keep raw values first and append the hand-calculated descriptor block."""
    from benchmarks.joint_effect.uea.tools.features import build_features

    recording = np.array([[[1, 1], [2, 2], [4, 4], [3, 0]]], dtype=float)
    features = build_features(recording, max_patterns=2, min_support=0).matrices
    np.testing.assert_allclose(features[method], [expected])


def test_composed_features_are_separate_and_ablate_every_pattern():
    """Check independent feature blocks and remove a descriptor from every GP.

    Mine two patterns in a four-observation recording. Compare descriptors
    with literal pair-set calculations and check every ablation's columns.
    """
    from benchmarks.joint_effect.uea.tools.features import (
        DESCRIPTORS,
        build_features,
    )

    x = np.array([[[1, 1], [2, 2], [4, 4], [3, 0]]], dtype=float)
    features = build_features(x, max_patterns=2, min_support=0).matrices
    assert features["gi"].shape == (1, 20)
    assert features["gp"].shape == (1, 4)
    assert features["gcs"].shape == (1, 10)
    np.testing.assert_allclose(features["gp"][0, ::2], [0.667, 0.333])
    assert np.all((features["gp"][0, 1::2] >= 0) & (features["gp"][0, 1::2] <= 1))

    expected = np.array([[[3 / 6, 2, 2 / 3, 2 / 9, 4 / 3], [2 / 6, 2, 1, 0.25, 2.5]]])
    np.testing.assert_allclose(features["gcs"], expected.reshape(1, -1))
    for column, name in enumerate(DESCRIPTORS):
        np.testing.assert_allclose(
            features[f"gcs_no_{name}"],
            np.delete(expected, column, axis=2).reshape(1, -1),
        )


def test_reverse_only_support_and_empty_descriptors():
    """Check full support and empty forward sets using decreasing channels."""
    from benchmarks.joint_effect.uea.tools.features import build_features

    x = np.array([[[3, 3], [2, 2], [1, 1]]], dtype=float)
    features = build_features(x, max_patterns=1, min_support=0).matrices
    np.testing.assert_array_equal(features["gp"], [[1, 1, 0, 0]])
    # The inverse-only retained GP has three isolated observations.
    # The conflicting candidate has zero support, so its unused slot is zero.
    np.testing.assert_array_equal(features["gcs"], [[0, 3, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert all(np.isfinite(matrix).all() for matrix in features.values())


def test_build_features_mines_each_recording_with_graank(monkeypatch):
    """Wrap real GRAANK discovery to verify each input recording is preserved."""
    from benchmarks.joint_effect.uea.tools.features import build_features
    from so4gp.algorithms import GRAANK

    recordings = np.array(
        [
            [[1, 3], [2, 4], [3, 1], [4, 2], [5, 5]],
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
        ],
        dtype=float,
    )
    discover = GRAANK.discover
    observed = []

    def observe(miner, *args, **kwargs):
        """Run the miner and record the numerical data held by its engine."""
        result = discover(miner, *args, **kwargs)
        observed.append(miner.mining_engine.data.copy())
        return result

    monkeypatch.setattr(GRAANK, "discover", observe)
    features = build_features(recordings, max_patterns=2, min_support=0.2)
    np.testing.assert_array_equal(np.stack(observed), recordings)
    assert features.matrices["gp"].shape == (2, 4)
    assert features.matrices["gcs"].shape == (2, 10)


def test_timing_layout_preserves_support_confidence_and_density():
    """Compare two layouts of the same generated rows through build_features.

    The layouts preserve ordinary GP scores and directional density while
    changing structural GCS descriptors.
    """
    from benchmarks.joint_effect.synthetic.tools.generator import _timing_case
    from benchmarks.joint_effect.uea.tools.features import build_features

    recordings = np.stack(
        [_timing_case(label, np.random.default_rng(7), 0.5) for label in (0, 1)]
    )
    features = build_features(recordings, max_patterns=4, min_support=0.2).matrices
    np.testing.assert_array_equal(features["gp"][0], features["gp"][1])
    np.testing.assert_array_equal(features["gcs"][0, ::5], features["gcs"][1, ::5])
    assert np.any(features["gcs"][0] != features["gcs"][1])


def test_feature_building_keeps_fold_and_test_recordings_separate(monkeypatch):
    """Record build_features inputs during tuning and final evaluation.

    Check that each CV preparation uses disjoint training/validation sets
    and that the final preparation uses the original train/test split.
    """
    from benchmarks.joint_effect.uea.tools import evaluation
    from benchmarks.joint_effect.uea.tools.features import METHODS, build_features

    train = np.random.default_rng(5).normal(size=(12, 8, 3))
    train[:, 0, 0] = np.arange(12)
    test = np.random.default_rng(6).normal(size=(6, 8, 3))
    test[:, 0, 0] = np.arange(20, 26)
    seen = []

    def observe(recordings, **kwargs):
        """Record input identifiers and delegate to the shared feature builder."""
        seen.append(set(recordings[:, 0, 0]))
        return build_features(recordings, **kwargs)

    monkeypatch.setattr(evaluation, "build_features", observe)
    result = evaluation.evaluate_split(
        train,
        np.arange(12) % 2,
        test,
        np.arange(6) % 2,
        seed=1,
        config=evaluation.BenchmarkConfig(max_patterns=4, folds=3, bootstrap=10),
    )
    assert result["cv_folds"] == len(result["cv"]) == 3
    assert len(seen) == 8
    for position in range(0, 6, 2):
        training, validation = seen[position : position + 2]
        assert len(training) == 8 and len(validation) == 4
        assert training.isdisjoint(validation)
        assert training | validation == set(range(12))
    assert seen[-2] == set(range(12))
    assert seen[-1] == set(range(20, 26))
    assert [row["method"] for row in result["metrics"]] == list(METHODS)
    baseline = next(row for row in result["metrics"] if row["method"] == "baseline")
    assert baseline["features"] == 24
    feature_counts = {row["method"]: row["features"] for row in result["metrics"]}
    assert feature_counts["baseline_gcs"] == 54
    assert feature_counts["baseline_standard"] == 36
    scores = {row["method"]: row for row in result["metrics"]}
    contrasts = {(row["method"], row["baseline"]): row for row in result["contrasts"]}
    for method in ("baseline_gcs", "baseline_standard"):
        for metric in ("balanced_accuracy", "macro_f1"):
            assert contrasts[method, "baseline"][metric] == pytest.approx(
                scores[method][metric] - scores["baseline"][metric]
            )
    assert all(0 <= row["macro_f1"] <= 1 for row in result["metrics"])


def test_build_features_filters_support_per_recording_and_pads():
    """Apply the support threshold within each recording and pad unused slots."""
    from benchmarks.joint_effect.uea.tools.features import build_features

    recordings = np.array(
        [
            [[1, 3], [2, 4], [3, 1], [4, 2], [5, 5]],
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
        ],
        dtype=float,
    )
    features = build_features(recordings, max_patterns=3, min_support=0.4).matrices
    np.testing.assert_allclose(features["gp"][:, 0], [0.6, 1])
    np.testing.assert_array_equal(features["gp"][:, 2:], np.zeros((2, 2)))
    np.testing.assert_array_equal(features["gcs"][:, 5:], np.zeros((2, 5)))


def test_gi_columns_keep_channel_identity_after_pruning():
    """Verify that pruning a constant channel leaves its signed GI slots empty."""
    from benchmarks.joint_effect.uea.tools.features import build_features

    increasing = np.arange(4, dtype=float)
    recording = np.column_stack((increasing, increasing, increasing))
    pruned = recording.copy()
    pruned[:, 0] = 1
    features = (
        build_features(np.stack((pruned, recording)), max_patterns=2, min_support=0.2)
        .matrices["gi"]
        .reshape(2, 6, 5)
    )
    np.testing.assert_array_equal(features[0, :2], np.zeros((2, 5)))
    np.testing.assert_array_equal(features[0, 2:], features[1, 2:])


@pytest.mark.parametrize("min_support", [0.2, 1.0])
def test_recordings_with_no_mined_patterns_are_zero_padded(min_support):
    """Handle constant channels and a threshold that admits no GP."""
    from benchmarks.joint_effect.uea.tools.features import build_features

    recording = np.tile(np.arange(3, dtype=float)[:, None], (1, 2))
    if min_support == 0.2:
        recording[:] = 1
    features = build_features(
        recording[None], max_patterns=2, min_support=min_support
    ).matrices
    np.testing.assert_array_equal(features["gp"], np.zeros((1, 4)))
    np.testing.assert_array_equal(features["gcs"], np.zeros((1, 10)))
    assert all(np.isfinite(matrix).all() for matrix in features.values())


@pytest.mark.parametrize(
    "max_patterns,min_support", [(0, 0.2), (-1, 0.2), (2, -0.1), (2, 1.1), (2, np.nan)]
)
def test_build_features_rejects_invalid_limits(max_patterns, min_support):
    """Reject invalid pattern limits and support thresholds at the entry point."""
    from benchmarks.joint_effect.uea.tools.features import build_features

    recording = np.tile(np.arange(3, dtype=float)[None, :, None], (1, 1, 2))
    with pytest.raises(ValueError):
        build_features(recording, max_patterns=max_patterns, min_support=min_support)


def test_synthetic_runner_uses_build_features(tmp_path, monkeypatch):
    """Run both synthetic regimes through the shared feature builder.

    Inspect matrix widths at each extraction call and check the manifest
    from a small explicit experiment with the configured train/test sizes.
    """
    import json

    from benchmarks.joint_effect import benchmark
    from benchmarks.joint_effect.uea.tools import evaluation
    from benchmarks.joint_effect.uea.tools.features import DESCRIPTORS, build_features

    observed = []

    def observe(recordings, **kwargs):
        """Extract features and retain their dimensions for integration checks."""
        features = build_features(recordings, **kwargs)
        observed.append((features.matrices["gp"].shape, features.matrices["gcs"].shape))
        return features

    monkeypatch.setattr(benchmark, "PARENT_PATH", tmp_path)
    monkeypatch.setattr(evaluation, "build_features", observe)
    assert (
        benchmark.main(
            [
                "synthetic",
                "--seeds",
                "0",
                "--noise",
                "0.5",
                "--train-size",
                "12",
                "--test-size",
                "8",
                "--max-patterns",
                "2",
                "--folds",
                "2",
                "--c-grid",
                "1",
                "--bootstrap",
                "10",
            ]
        )
        == 0
    )

    assert observed
    assert all(gp[1] == 60 and gcs[1] == 30 * len(DESCRIPTORS) for gp, gcs in observed)
    (manifest_path,) = tmp_path.glob("synthetic/results/*/run.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["experiment"] == "synthetic"
    assert {run["dataset_name"] for run in manifest["runs"]} == {"shared", "timing"}
    assert manifest["arguments"]["seeds"] == [0]
    assert all(
        (run["training_cases"], run["test_cases"]) == (12, 8)
        for run in manifest["runs"]
    )


def test_cv_caps_folds_at_the_smallest_training_class():
    """Check fold reduction and score averaging with two rare-class cases."""
    from benchmarks.joint_effect.uea.tools.evaluation import (
        BenchmarkConfig,
        evaluate_split,
    )

    train = np.tile(np.arange(8, dtype=float)[None, :, None], (8, 1, 2))
    train[:2] += 20
    labels = np.repeat(["rare", "common"], [2, 6])
    test_idx = [0, 2]
    result = evaluate_split(
        train,
        labels,
        train[test_idx] + 0.25,
        labels[test_idx],
        seed=1,
        config=BenchmarkConfig(max_patterns=2, folds=3, c_grid=(10.0,), bootstrap=10),
    )
    assert result["cv_folds"] == len(result["cv"]) == 2
    standard = next(row for row in result["metrics"] if row["method"] == "standard")
    assert standard["cv_balanced_accuracy"] == 1.0
    assert standard["balanced_accuracy"] == 1.0
