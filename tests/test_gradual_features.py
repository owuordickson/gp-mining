"""Regression checks for GRAANK's array-to-gradual-feature interface."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from so4gp.algorithms import GRAANK

DESCRIPTORS = [
    "support",
    "density",
    "connectivity",
    "avg_separation",
    "rank_dispersion",
    "singularity",
]


def test_two_dimensional_patterns_have_numeric_ids_and_exact_descriptors():
    """Check full support and all GCS descriptors against literal pair sets."""
    data = np.array([[1, 1], [2, 2], [4, 4], [3, 0]], dtype=float)
    original = data.copy()
    result = GRAANK.build_gradual_features(data, max_patterns=None)

    assert isinstance(result, pd.DataFrame)
    assert result.index.tolist() == ["0+ 1+", "0+ 1-"]
    assert result.index.name == "pattern"
    assert result.columns.tolist() == DESCRIPTORS
    np.testing.assert_allclose(
        result.to_numpy(),
        [
            [2 / 3, 0.5, 2, 4 / 3, 2 / 9, 2 / 3],
            [1 / 3, 1 / 3, 2, 2.5, 0.25, 1],
        ],
    )
    np.testing.assert_array_equal(data, original)


def test_single_channel_gradual_items_keep_both_directions_and_empty_set():
    """A single channel needs no second attribute to describe either GI."""
    result = GRAANK.build_gradual_features(
        np.array([[1], [2], [3]], dtype=float), pattern_level=1
    )
    assert result.index.tolist() == ["0+", "0-"]
    np.testing.assert_allclose(result.loc["0+"], [1, 1, 1, 4 / 3, 2 / 9, 2 / 3])
    np.testing.assert_allclose(result.loc["0-"].iloc[:3], [1, 0, 3])
    assert result.loc["0-"].iloc[3:].isna().all()


def test_gradual_item_support_uses_existing_bitmap_support():
    """GI support is the fitted GI score, not its forward density or popcount."""
    result = GRAANK.build_gradual_features(
        np.array([[1, 1], [2, 2], [4, 4], [3, 0]], dtype=float),
        pattern_level=1,
    )
    assert result.index.tolist() == ["0+", "0-", "1+", "1-"]
    np.testing.assert_allclose(result["support"], [5 / 6, 5 / 6, 0.5, 0.5])
    np.testing.assert_allclose(result["density"], [5 / 6, 1 / 6, 0.5, 0.5])


def test_pattern_level_means_exact_size_even_when_larger_patterns_exist():
    """Retain all 16 canonical triples instead of replacing them by quadruples."""
    data = np.tile(np.arange(4, dtype=float)[:, None], (1, 4))
    result = GRAANK.build_gradual_features(data, pattern_level=3)
    assert result.shape == (16, 6)
    assert all(len(pattern.split()) == 3 for pattern in result.index)
    np.testing.assert_allclose(result.loc["0+ 1+ 2+"], [1, 1, 1, 5 / 3, 5 / 9, 0.5])
    assert result.loc["0+ 1+ 2-", "support"] == 0
    assert result.loc["0+ 1+ 2-", "connectivity"] == 4


def test_three_dimensional_features_align_patterns_and_keep_recording_rows():
    """An inverse recording keeps its own support and empty-set descriptors."""
    recordings = np.array(
        [
            [[1, 1], [2, 2], [3, 3]],
            [[3, 3], [2, 2], [1, 1]],
        ],
        dtype=float,
    )
    original = recordings.copy()
    result = GRAANK.build_gradual_features(recordings)
    assert isinstance(result.columns, pd.MultiIndex)
    assert result.columns.names == ["pattern", "descriptor"]
    assert result.index.name == "recording"
    assert result.shape == (2, 12)
    assert result.columns.get_level_values("pattern").unique().tolist() == [
        "0+ 1+",
        "0+ 1-",
    ]
    np.testing.assert_allclose(
        result["0+ 1+"].to_numpy(),
        [[1, 1, 1, 4 / 3, 2 / 9, 2 / 3], [1, 0, 3, 0, 0, 0]],
    )
    np.testing.assert_allclose(result["0+ 1-"].to_numpy(), [[0, 0, 3, 0, 0, 0]] * 2)
    assert np.isfinite(result.to_numpy()).all()
    np.testing.assert_array_equal(recordings, original)


def test_pattern_limit_is_a_cap_and_does_not_create_extra_padding():
    """A small cap selects a deterministic prefix; oversized caps add no columns."""
    data = np.array([[1, 2], [2, 1], [3, 3]], dtype=float)
    limited = GRAANK.build_gradual_features(data, max_patterns=1)
    oversized = GRAANK.build_gradual_features(data, max_patterns=100)
    assert limited.index.tolist() == ["0+ 1+"]
    assert oversized.index.tolist() == ["0+ 1+", "0+ 1-"]
    np.testing.assert_allclose(limited.to_numpy(), oversized.iloc[:1].to_numpy())
    recordings = np.stack((data, data[::-1]))
    assert GRAANK.build_gradual_features(recordings, max_patterns=1).shape == (2, 6)


def test_constant_patterns_preserve_empty_set_meaning_in_two_dimensions():
    """Undefined statistics stay NaN in a descriptive table and become zero features."""
    data = np.ones((3, 2))
    result = GRAANK.build_gradual_features(data)
    np.testing.assert_array_equal(result.iloc[:, :3], [[0, 0, 3], [0, 0, 3]])
    assert result.iloc[:, 3:].isna().all().all()
    encoded = GRAANK.build_gradual_features(data[None])
    np.testing.assert_array_equal(
        encoded.to_numpy(), [[0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0]]
    )


def test_encoded_recordings_work_with_the_existing_classifier_pipeline():
    """The feature table preserves case count and separates known joint behavior."""
    increasing = [[1, 1], [2, 2], [3, 3], [4, 4]]
    opposing = [[1, 4], [2, 3], [3, 2], [4, 1]]
    train = np.array([increasing, opposing, increasing, opposing], dtype=float)
    test = np.array([increasing, opposing], dtype=float)
    x_train = GRAANK.build_gradual_features(train)
    x_test = GRAANK.build_gradual_features(test)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000))
    model.fit(x_train, [0, 1, 0, 1])
    np.testing.assert_array_equal(model.predict(x_test), [0, 1])


@pytest.mark.parametrize(
    "data,kwargs",
    [
        (np.ones(3), {}),
        (np.ones((1, 3)), {}),
        (np.ones((0, 3, 2)), {}),
        (np.ones((3, 0)), {}),
        (np.ones((2, 3, 2, 1)), {}),
        (np.array([[1, 2], [np.nan, 3]]), {}),
        (np.array([[1, 2], [np.inf, 3]]), {}),
        (np.ones((3, 2), dtype=complex), {}),
        (np.ones((3, 2)), {"pattern_level": 0}),
        (np.ones((3, 2)), {"pattern_level": 3}),
        (np.ones((3, 2)), {"pattern_level": 1.5}),
        (np.ones((3, 2)), {"pattern_level": True}),
        (np.ones((3, 2)), {"max_patterns": 0}),
        (np.ones((3, 2)), {"max_patterns": -1}),
        (np.ones((3, 2)), {"max_patterns": 1.5}),
        (np.ones((3, 2)), {"max_patterns": True}),
    ],
)
def test_invalid_inputs_are_rejected_before_mining(data, kwargs):
    """Reject malformed arrays and invalid counts before computing features."""
    with pytest.raises(ValueError):
        GRAANK.build_gradual_features(data, **kwargs)
