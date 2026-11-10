import numpy as np


def test_composed_features_use_repository_descriptors_and_keep_density_in_ablations():
    from benchmarks.joint_effect.uea.tools.features import build_features

    # Both increasing: forward (0,1), (0,2), (1,2), and (3,2) in reverse.
    x = np.array([[[1, 1], [2, 2], [4, 4], [3, 0]]], dtype=float)
    features = build_features(x, (((0, 1), (1, 1)),)).matrices
    gp, gcs = features["gp"], features["gcs"]
    np.testing.assert_allclose(gp[0, -2:], [4 / 6, 3 / 6])
    np.testing.assert_array_equal(gcs[:, : gp.shape[1]], gp)
    # A triangle and one isolated observation.
    np.testing.assert_allclose(gcs[0, -3:], [2, 2 / 3, 2 / 9])
    for index, name in enumerate(("connectivity", "singularity", "dispersion")):
        np.testing.assert_array_equal(
            features[f"gcs_no_{name}"], np.delete(gcs, gp.shape[1] + index, axis=1)
        )


def test_reverse_only_support_and_empty_descriptors():
    from benchmarks.joint_effect.uea.tools.features import build_features

    x = np.array([[[3, 3], [2, 2], [1, 1]]], dtype=float)
    features = build_features(x, (((0, 1), (1, 1)),)).matrices
    np.testing.assert_array_equal(features["gp"][0, -2:], [1, 0])
    np.testing.assert_array_equal(features["gcs"][0, -3:], [3, 0, 0])
    assert np.isfinite(features["gcs"]).all()


def test_pattern_selection_keeps_both_directions():
    from benchmarks.joint_effect.uea.tools.features import select_patterns

    x = np.tile(np.arange(8, dtype=float)[None, :, None], (2, 1, 3))
    patterns = select_patterns(x, max_patterns=2, min_support=1)
    assert patterns == (((0, 1), (1, 1)), ((0, -1), (1, -1)))


def test_timing_layout_matches_gp_support_and_density():
    from benchmarks.joint_effect.synthetic.tools.generator import _timing_case
    from benchmarks.joint_effect.uea.tools.features import build_features

    patterns = (((0, 1), (1, 1)), ((0, -1), (1, -1)))
    cases = np.stack([_timing_case(y, np.random.default_rng(7), 0.5) for y in (0, 1)])
    features = build_features(cases, patterns).matrices
    np.testing.assert_array_equal(features["gp"][0, -4:], features["gp"][1, -4:])
    assert np.any(features["gcs"][0, -6:] != features["gcs"][1, -6:])


def test_pattern_mining_uses_fold_training_and_never_test(monkeypatch):
    from benchmarks.joint_effect.uea.tools import evaluation
    from benchmarks.joint_effect.uea.tools.features import METHODS, select_patterns

    train = np.random.default_rng(5).normal(size=(12, 8, 3))
    train[:, 0, 0] = np.arange(12)
    test = np.random.default_rng(6).normal(size=(6, 8, 3))
    test[:, 0, 0] = np.arange(20, 26)
    seen = []

    def observe(cases, **kwargs):
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
    assert len(seen) == 4
    assert all(len(fold) == 8 and fold < set(range(12)) for fold in seen[:-1])
    assert seen[-1] == set(range(12))
    assert [row["method"] for row in result["metrics"]] == list(METHODS)
    assert all(0 <= row["macro_f1"] <= 1 for row in result["metrics"])
