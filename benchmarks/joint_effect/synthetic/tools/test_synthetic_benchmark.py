import numpy as np


def test_composed_features_use_repository_descriptors_and_ablate_gcs_columns():
    """Check literal descriptor values and the columns removed by ablation.

    Encode a four-observation fixture, compare support and structural
    values with hand calculations, and compare each ablation with
    deleting its appended descriptor from the full GCS matrix.
    """
    from benchmarks.joint_effect.uea.tools.features import build_features

    # Both increasing: forward (0,1), (0,2), (1,2), and (3,2) in reverse.
    x = np.array([[[1, 1], [2, 2], [4, 4], [3, 0]]], dtype=float)
    features = build_features(x, (((0, 1), (1, 1)),)).matrices
    gp, gcs = features["gp"], features["gcs"]
    np.testing.assert_allclose(gp[0, -2:], [4 / 6, 3 / 6])
    np.testing.assert_array_equal(gcs[:, : gp.shape[1]], gp)
    # A triangle and one isolated observation.
    np.testing.assert_allclose(gcs[0, -4:], [3 / 6, 2, 2 / 3, 2 / 9])
    for index, name in enumerate(
        ("density", "connectivity", "singularity", "dispersion")
    ):
        np.testing.assert_array_equal(
            features[f"gcs_no_{name}"], np.delete(gcs, gp.shape[1] + index, axis=1)
        )


def test_reverse_only_support_and_empty_descriptors():
    """Check full support and empty canonical sets on decreasing channels.

    Verify that reverse comparisons contribute support while forward
    density is zero, and that undefined empty-set descriptors become
    finite values in the feature matrix.
    """
    from benchmarks.joint_effect.uea.tools.features import build_features

    x = np.array([[[3, 3], [2, 2], [1, 1]]], dtype=float)
    features = build_features(x, (((0, 1), (1, 1)),)).matrices
    np.testing.assert_array_equal(features["gp"][0, -2:], [1, 0])
    np.testing.assert_array_equal(features["gcs"][0, -3:], [3, 0, 0])
    assert np.isfinite(features["gcs"]).all()


def test_pattern_selection_keeps_both_directions():
    """Check that a selected family contributes both inverse orientations.

    Use identical increasing channels so support ties are exact, cap
    the vocabulary at two patterns, and assert deterministic ordering.
    """
    from benchmarks.joint_effect.uea.tools.features import select_patterns

    x = np.tile(np.arange(8, dtype=float)[None, :, None], (2, 1, 3))
    patterns = select_patterns(x, max_patterns=2, min_support=1)
    assert patterns == (((0, 1), (1, 1)), ((0, -1), (1, -1)))


def test_timing_layout_matches_gp_support_and_density():
    """Check matched agreement counts across the two timing layouts.

    Use the same seed for both labels, compare their final GP support
    and density columns, and require a difference in the GCS columns.
    """
    from benchmarks.joint_effect.synthetic.tools.generator import _timing_case
    from benchmarks.joint_effect.uea.tools.features import build_features

    patterns = (((0, 1), (1, 1)), ((0, -1), (1, -1)))
    cases = np.stack([_timing_case(y, np.random.default_rng(7), 0.5) for y in (0, 1)])
    features = build_features(cases, patterns).matrices
    np.testing.assert_array_equal(features["gp"][0, -4:], features["gp"][1, -4:])
    assert np.any(features["gcs"][0, -6:] != features["gcs"][1, -6:])


def test_pattern_mining_uses_fold_training_and_never_test(monkeypatch):
    """Verify that pattern discovery sees only the appropriate training cases.

    Record case identifiers entering each selection call during CV and
    final fitting. Check the fold subsets, complete final training set,
    reported fold count, and metric validity.
    """
    from benchmarks.joint_effect.uea.tools import evaluation
    from benchmarks.joint_effect.uea.tools.features import METHODS, select_patterns

    train = np.random.default_rng(5).normal(size=(12, 8, 3))
    train[:, 0, 0] = np.arange(12)
    test = np.random.default_rng(6).normal(size=(6, 8, 3))
    test[:, 0, 0] = np.arange(20, 26)
    seen = []

    def observe(cases, **kwargs):
        """Record the case identifiers entering mining and run the real selector."""
        seen.append(set(cases[:, 0, 0]))
        return select_patterns(cases, **kwargs)

    monkeypatch.setattr(evaluation, "select_patterns", observe)
    result = evaluation.evaluate_split(
        train,
        np.arange(12) % 2,
        test,
        np.arange(6) % 2,
        seed=1,
        protocol=evaluation.Protocol(max_patterns=4, folds=3, bootstrap=10),
    )
    assert result["cv_folds"] == len(result["cv"]) == 3
    assert len(seen) == 4
    assert all(len(fold) == 8 and fold < set(range(12)) for fold in seen[:-1])
    assert seen[-1] == set(range(12))
    assert [row["method"] for row in result["metrics"]] == list(METHODS)
    assert all(0 <= row["macro_f1"] <= 1 for row in result["metrics"])


def test_graank_discovers_families_within_recordings_and_ranks_mean_full_support(
    monkeypatch,
):
    """Verify GRAANK discovery and inclusive mean-support filtering.

    Capture the data processed by real GRAANK calls to detect altered
    recording boundaries or extra header rows. Check inverse patterns
    at two cutoffs bracketing the lower family's mean support.
    """
    from benchmarks.joint_effect.uea.tools.features import select_patterns
    from so4gp.algorithms import GRAANK

    cases = np.array(
        [
            [[1, 3], [2, 4], [3, 1], [4, 2], [5, 5]],
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
        ],
        dtype=float,
    )
    discover = GRAANK.discover
    observed = []

    def observe(miner, *args, **kwargs):
        """Run GRAANK and capture the recording held by its mining engine."""
        result = discover(miner, *args, **kwargs)
        observed.append(miner.mining_engine.data.copy())
        return result

    monkeypatch.setattr(GRAANK, "discover", observe)
    selected = select_patterns(cases, max_patterns=4, min_support=0.2)
    np.testing.assert_array_equal(np.stack(observed), cases)
    assert selected == (
        ((0, 1), (1, 1)),
        ((0, -1), (1, -1)),
        ((0, 1), (1, -1)),
        ((0, -1), (1, 1)),
    )
    assert select_patterns(cases, max_patterns=4, min_support=0.21) == selected[:2]


def test_empty_pattern_intersection_is_valid():
    """Check the feature representation of an empty GP intersection.

    Intersect opposite directions on identical increasing channels and
    assert zero support, zero density, and one component per observation.
    """
    from benchmarks.joint_effect.uea.tools.features import build_features

    x = np.array([[[1, 1], [2, 2], [3, 3]]], dtype=float)
    features = build_features(x, (((0, 1), (1, -1)),)).matrices
    np.testing.assert_array_equal(features["gp"][0, -2:], [0, 0])
    np.testing.assert_array_equal(features["gcs"][0, -4:], [0, 3, 0, 0])


def test_run_without_command_line_arguments(tmp_path, monkeypatch):
    """Check the no-argument launcher against the expected synthetic settings.

    Redirect output to a temporary directory, select the synthetic
    experiment, and inspect the generated manifest's tasks and seeds.
    """
    import json

    from benchmarks.joint_effect import benchmark

    monkeypatch.setattr(benchmark, "HERE", tmp_path)
    monkeypatch.setattr(benchmark, "RUN_EXPERIMENT", "synthetic")
    assert benchmark.main() == 0

    (manifest_path,) = tmp_path.glob("synthetic/results/*/run.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["experiment"] == "synthetic"
    assert {run["dataset"] for run in manifest["runs"]} == {"shared", "timing"}
    assert manifest["arguments"]["seeds"] == [0]


def test_cv_caps_folds_at_the_smallest_training_class():
    """Check fold reduction and validation-score averaging for a rare class.

    Use separable signals with two rare-class training cases, request
    three folds, and require two folds with correctly averaged scores.
    """
    from benchmarks.joint_effect.uea.tools.evaluation import Protocol, evaluate_split

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
        protocol=Protocol(max_patterns=2, folds=3, c_grid=(10.0,), bootstrap=10),
    )

    assert result["cv_folds"] == len(result["cv"]) == 2
    assert result["classes"] == ["common", "rare"]
    standard = next(row for row in result["metrics"] if row["method"] == "standard")
    assert standard["cv_balanced_accuracy"] == 1.0
    assert standard["balanced_accuracy"] == 1.0
