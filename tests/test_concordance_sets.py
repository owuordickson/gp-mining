"""Regression coverage for concordance descriptors and packed comparisons."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from so4gp import GI, GP, DataGP, PairwiseMatrix

BACKENDS = ["numpy", "cpu"]
if torch.cuda.is_available():
    BACKENDS.append("cuda")


@pytest.fixture(params=BACKENDS)
def backend(request):
    """Exercise NumPy and tensor implementations with the same literal fixtures."""
    def convert(values, *, packed=False):
        array = np.asarray(values, dtype=np.uint8 if packed else np.int64)
        if request.param == "numpy":
            return array
        return torch.as_tensor(array, device=request.param)

    return convert


def assert_array_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "pairs, size, density, mean_gap, variance, components, singularity",
    [
        # Triangle {0, 1, 3}, edge {4, 5}, and isolated observation 2.
        ([(0, 1), (0, 3), (1, 3), (4, 5)], 6, 4 / 15, 1.75, 0.6875, 3, 0.5),
        # A star has maximum degree equal to its pair count.
        ([(0, 1), (0, 2), (0, 3), (0, 4)], 5, 0.4, 2.5, 1.25, 1, 1.0),
        ([(1, 4)], 5, 0.1, 3.0, 0.0, 4, 1.0),
        # Connectivity must propagate along the whole chain on tensor backends.
        ([(0, 5), (1, 4), (2, 3), (3, 4), (4, 5)], 6, 1 / 3, 2.2, 2.56, 1, 0.6),
        (
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            4, 1.0, 5 / 3, 5 / 9, 1, 0.5,
        ),
    ],
    ids=["isolates", "star", "single-pair", "chain", "complete"],
)
def test_descriptors_match_hand_calculated_values(
    backend, pairs, size, density, mean_gap, variance, components, singularity
):
    pattern = GP()

    assert pattern.compute_descriptors(backend(pairs), size)

    assert pattern.density == pytest.approx(density)
    assert pattern.avg_deviation_from_diagonal == pytest.approx(mean_gap)
    assert pattern.rank_dispersion == pytest.approx(variance)
    assert pattern.graph_connectivity == components
    assert pattern.singularity_score == pytest.approx(singularity)


@pytest.mark.parametrize("pairs", [[], np.empty((0, 2), dtype=np.int64)])
def test_empty_concordance_set_resets_previous_descriptors(backend, pairs):
    pattern = GP()
    assert pattern.compute_descriptors(backend([(0, 1), (0, 2)]), 4)

    assert pattern.compute_descriptors(backend(pairs), 4)

    assert pattern.density == 0.0
    assert pattern.graph_connectivity == 4
    assert math.isnan(pattern.avg_deviation_from_diagonal)
    assert math.isnan(pattern.rank_dispersion)
    assert math.isnan(pattern.singularity_score)


def test_concordance_extraction_ignores_diagonal_lower_triangle_and_padding(backend):
    # Row-major 3x3 bits: 110 / 111 / 101, followed by seven padding ones.
    packed = backend([0xDE, 0xFF], packed=True)

    pairs = GP.gen_gradual_concordance_set(packed, 3)

    assert_array_equal(pairs, [[0, 1], [1, 2]])
    assert type(pairs) is type(packed)
    if isinstance(pairs, torch.Tensor):
        assert pairs.device == packed.device


def test_empty_bitmap_has_a_two_column_concordance_set(backend):
    pairs = GP.gen_gradual_concordance_set(backend([0, 0], packed=True), 3)

    assert pairs.shape == (0, 2)


@pytest.mark.parametrize(
    "packed",
    [
        # Only (0, 1) or its inverse (1, 0), then seven padding ones.
        [0x40, 0x7F],
        [0x10, 0x7F],
    ],
    ids=["upper", "lower"],
)
def test_selected_rows_include_both_pair_endpoints_and_ignore_padding(backend, packed):
    selected = GP.get_selected_rows(backend(packed, packed=True), 3)

    assert_array_equal(selected, [0, 1])


def test_packed_intersection_counts_bits_in_both_orientations(backend):
    left = PairwiseMatrix(backend([0x66, 0], packed=True), 0, {"0+"})
    right = PairwiseMatrix(backend([0x56, 0], packed=True), 0, {"1-"})

    result = GP.perform_and(left, right, 3)

    # Three comparisons: (0, 1), (1, 2), (2, 0). Only two are forward pairs.
    assert_array_equal(result.packed_bin_mat, [0x46, 0])
    assert result.support == 1.0  # Three set bits / C(3, 2), not byte sums.
    assert result.pattern == {"0+", "1-"}
    assert_array_equal(left.packed_bin_mat, [0x66, 0])
    assert_array_equal(right.packed_bin_mat, [0x56, 0])
    pairs = GP.gen_gradual_concordance_set(result.packed_bin_mat, 3)
    assert_array_equal(pairs, [[0, 1], [1, 2]])
    pattern = GP()
    assert pattern.compute_descriptors(pairs, 3)
    assert pattern.density == pytest.approx(2 / 3)


def test_disjoint_packed_bitmaps_have_zero_support(backend):
    left = PairwiseMatrix(backend([0x40, 0], packed=True), 0, {"0+"})
    right = PairwiseMatrix(backend([0x20, 0], packed=True), 0, {"1-"})

    result = GP.perform_and(left, right, 3)

    assert_array_equal(result.packed_bin_mat, [0, 0])
    assert result.support == 0.0


def test_packed_intersection_counts_valid_bits_in_a_partially_filled_byte(backend):
    # A 6x6 bitmap uses four bits of the final byte; its last four bits are padding.
    left = PairwiseMatrix(backend([0, 0, 0, 0, 0xA0], packed=True), 0, {"0+"})
    right = PairwiseMatrix(backend([0, 0, 0, 0, 0xC0], packed=True), 0, {"1-"})

    result = GP.perform_and(left, right, 6)

    assert_array_equal(result.packed_bin_mat, [0, 0, 0, 0, 0x80])
    assert result.support == pytest.approx(1 / 15)
    assert_array_equal(GP.get_selected_rows(result.packed_bin_mat, 6), [2, 5])


@pytest.mark.parametrize("missing_left", [True, False])
def test_missing_bitmap_returns_correctly_sized_zero_bitmap(backend, missing_left):
    present = PairwiseMatrix(backend([0x66, 0], packed=True), 1, {"0+"})
    left, right = (None, present) if missing_left else (present, None)

    result = GP.perform_and(left, right, 3)

    assert_array_equal(result.packed_bin_mat, [0, 0])
    assert type(result.packed_bin_mat) is type(present.packed_bin_mat)
    assert result.support == 0.0
    assert result.pattern == set()


@pytest.fixture
def monotonic_data():
    data = DataGP(pd.DataFrame({"Age": [10, 20, 30, 40], "Expenses": [4, 3, 2, 1]}))
    data.fit_concordance_set()
    return data


def test_fitted_concordance_sets_keep_empty_inverse_orientations(monotonic_data):
    assert monotonic_data.concordance_set == {
        "0+": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        "0-": [],
        "1+": [],
        "1-": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
    }

    descriptors = monotonic_data.compute_attribute_descriptors().set_index("Pattern")
    for name in ("Age-", "Expenses+"):
        row = descriptors.loc[name]
        assert row["Support"] == 1.0
        assert row["Density"] == 0.0
        assert row["Graph Connectivity"] == 4
        assert math.isnan(row["Avg. Deviation from Diagonal"])
        assert math.isnan(row["Rank Dispersion"])
        assert math.isnan(row["Singularity Score"])
    for name in ("Age+", "Expenses-"):
        row = descriptors.loc[name]
        assert row["Density"] == 1.0
        assert row["Avg. Deviation from Diagonal"] == pytest.approx(1.667)
        assert row["Rank Dispersion"] == pytest.approx(0.556)
        assert row["Graph Connectivity"] == 1
        assert row["Singularity Score"] == 0.5


@pytest.mark.parametrize("symbols", [("+", "-"), ("-", "+")])
def test_tree_validation_counts_forward_and_reverse_support(monotonic_data, symbols):
    candidate = GP()
    for column, symbol in enumerate(symbols):
        candidate.add_gradual_item(GI(column, symbol))

    tree = candidate.validate_via_tree(monotonic_data)
    bitmap = candidate.validate_via_graank(monotonic_data, target_col=None)

    assert tree.as_set == bitmap.as_set == candidate.as_set
    assert tree.support == bitmap.support == 1.0
