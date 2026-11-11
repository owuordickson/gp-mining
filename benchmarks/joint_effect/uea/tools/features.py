"""Packed comparison features with a vocabulary fitted on training cases."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from so4gp.algorithms import GRAANK
from so4gp.data_gp import DataGP
from so4gp.gradual_patterns import GP, PairwiseMatrix

type Pattern = tuple[tuple[int, int], tuple[int, int]]

DESCRIPTORS = ("density", "connectivity", "singularity", "dispersion")
METHODS = ("standard", "gi", "gp", "gcs", *(f"gcs_no_{d}" for d in DESCRIPTORS))


@dataclass
class FeatureSet:
    """Store recording features and the time spent extracting them.

    Attributes:
        matrices: A feature matrix for each method, with one row per
            recording and columns in the order used by build_features.
        seconds: Durations in seconds for the shared standard, GI, GP
            support, and GCS descriptor extraction stages.
    """
    matrices: dict[str, np.ndarray]
    seconds: dict[str, float]


def _recording_frame(case: np.ndarray) -> pd.DataFrame:
    # Numeric column labels would be interpreted by DataGP as a data row.
    """Give one recording the named columns expected by SO4GP.

    Convert the observation-by-channel array to a DataFrame with names
    such as channel_0. These names prevent DataGP from interpreting
    numeric column labels as an additional observation.
    """
    return pd.DataFrame(case, columns=[f"channel_{i}" for i in range(case.shape[1])])


def _item_key(channel: int, sign: int) -> str:
    """Encode a channel and direction as a SO4GP gradual-item key.

    Append '+' to the zero-based channel index for a positive sign and
    '-' otherwise, producing keys such as '0+' and '2-'.
    """
    return f"{channel}{'+' if sign > 0 else '-'}"


def _inverse(pattern: Pattern) -> Pattern:
    """Reverse both directions of a two-channel pattern.

    Negate the two signs while retaining channel order. Both forms have
    the same full support and can have different directional densities.
    """
    (a, sign_a), (b, sign_b) = pattern
    return (a, -sign_a), (b, -sign_b)


def select_patterns(
    sequences: np.ndarray, *, max_patterns: int, min_support: float
) -> tuple[Pattern, ...]:
    """Learn a fixed vocabulary of two-channel patterns from training cases.

    Run SO4GP's Apriori GRAANK separately on each recording, stopping at
    two-item patterns and accepting candidates with positive full
    support. Group a pattern with its inverse and sum its supporting
    pair counts across recordings. An absent family contributes zero.
    Use GP.perform_and to obtain support before GP's display rounding.

    Rank families by mean full support, apply the inclusive support
    cutoff, and retain both orientations of each selected family.
    Support ties follow channel order, with equal directions first.

    Args:
        sequences: Training array shaped (cases, observations, channels).
            Comparisons stay within each recording.
        max_patterns: Positive even limit counting both orientations.
        min_support: Minimum mean full support in the interval [0, 1].

    Returns:
        An ordered tuple of patterns. Each pattern contains two
        (channel_index, sign) pairs, with signs +1 or -1.

    Raises:
        ValueError: The limit or cutoff is invalid, or the input has
            no cases, fewer than two observations, or fewer than two
            channels.
    """
    if max_patterns < 2 or max_patterns % 2:
        raise ValueError("max_patterns must be a positive even integer")
    if not 0 <= min_support <= 1:
        raise ValueError("min_support must be in [0, 1]")
    cases, n, channels = sequences.shape
    if cases == 0 or n < 2 or channels < 2:
        raise ValueError("Pattern selection needs cases with >=2 times and channels")

    pair_count = GP.pair_count(n)
    totals: dict[Pattern, int] = {}
    for case in sequences:
        miner = GRAANK(_recording_frame(case), min_sup=0)
        miner.discover(max_iteration=2, compute_descriptors=False)
        engine = miner.mining_engine
        bins = engine.valid_bins
        assert bins is not None
        for gp in engine.gradual_patterns or ():
            left, right = sorted(gp.gradual_items, key=lambda gi: gi.attribute_col)
            a, b = int(left.attribute_col), int(right.attribute_col)
            sign = 1 if left.symbol == right.symbol else -1
            family = ((a, 1), (b, sign))
            # Use exact library support. GP.support rounds its value for display.
            scored = GP.perform_and(bins[left.to_string()], bins[right.to_string()], n)
            totals[family] = totals.get(family, 0) + round(scored.support * pair_count)

    # Undiscovered families contribute zero on that recording. Rank by exact
    # counts and use channel order, then same direction, for support ties.
    ranked = sorted(totals, key=lambda p: (-totals[p], p[0][0], p[1][0], -p[1][1]))
    selected = []
    for pattern in ranked[: max_patterns // 2]:
        if totals[pattern] / (cases * pair_count) < min_support:
            break
        selected.extend((pattern, _inverse(pattern)))
    return tuple(selected)


def _standard(case: np.ndarray) -> np.ndarray:
    """Summarize one recording using moments, trends, and correlations.

    Return channel means, population standard deviations, and
    least-squares slopes over a time axis spanning -0.5 to 0.5,
    followed by Pearson correlations for distinct channel pairs in
    upper-triangle order. Constant channels contribute zero
    correlations. The output is a vector with 3*m + m*(m-1)/2 values
    for an input shaped (observations, m).
    """
    means = case.mean(axis=0)
    centered = case - means
    time = np.linspace(-0.5, 0.5, len(case))
    slopes = time @ centered / (time @ time)
    norms = np.linalg.norm(centered, axis=0)
    unit = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)
    correlation = unit.T @ unit
    upper = np.triu_indices(case.shape[1], k=1)
    return np.concatenate((means, case.std(axis=0), slopes, correlation[upper]))


def _describe(bitmap: PairwiseMatrix, n: int) -> GP:
    """Compute SO4GP descriptors from a packed comparison bitmap.

    Extract supporting pairs with i < j using
    GP.gen_gradual_concordance_set, then delegate the descriptor
    calculations to GP.compute_descriptors. Return the populated GP
    object. Empty sets have zero density and n connected components,
    with undefined separation and singularity statistics stored as NaN.

    Args:
        bitmap: Packed pairwise matrix for one GI or GP.
        n: Number of observations in the original square bitmap.

    Raises:
        ValueError: SO4GP rejects the extracted concordance pairs.
    """
    pairs = GP.gen_gradual_concordance_set(bitmap.packed_bin_mat, n)
    gp = GP()
    if not gp.compute_descriptors(pairs, n):
        raise ValueError("Invalid concordance pairs")
    return gp


def build_features(sequences: np.ndarray, patterns: tuple[Pattern, ...]) -> FeatureSet:
    """Encode each recording using a fixed vocabulary and several baselines.

    Build SO4GP GI bitmaps for both directions of each channel. For
    every selected GP, intersect the constituent bitmaps with
    GP.perform_and and compute its concordance descriptors through
    _describe. Apply the same ordered pattern columns to every case.

    The GI block contains density, mean separation, connectivity,
    singularity, and dispersion for each signed channel. The GP matrix
    adds support and directional density per pattern. GCS adds another
    block containing density, connectivity, singularity, and dispersion
    in DESCRIPTORS order. Each ablation removes its named descriptor
    from this additional block. Standard features form a separate
    matrix. Undefined empty-set statistics become zero in the matrices.

    Args:
        sequences: Array shaped (cases, observations, channels).
        patterns: Ordered two-item patterns obtained from training data.

    Returns:
        FeatureSet containing method matrices and shared extraction
        timings. All matrices contain one row per input recording.

    Raises:
        ValueError: The input has no cases, fewer than two observations
            or channels, or SO4GP cannot describe a concordance set.
    """
    cases, n, channels = sequences.shape
    if cases == 0 or n < 2 or channels < 2:
        raise ValueError("Feature extraction needs cases with >=2 times and channels")

    item_keys = [
        _item_key(channel, sign) for channel in range(channels) for sign in (1, -1)
    ]
    pattern_keys = tuple(
        (_item_key(*left), _item_key(*right)) for left, right in patterns
    )
    standard = np.empty((cases, 3 * channels + channels * (channels - 1) // 2))
    gi = np.empty((cases, 2 * channels, 5))
    gp_basic = np.empty((cases, len(patterns), 2))
    gp_extra = np.empty((cases, len(patterns), len(DESCRIPTORS)))
    seconds = dict.fromkeys(("standard", "gi", "gp", "gcs"), 0.0)

    for index, raw_case in enumerate(sequences):
        case = np.asarray(raw_case, dtype=np.float64)
        start = perf_counter()
        standard[index] = _standard(case)
        seconds["standard"] += perf_counter() - start

        start = perf_counter()
        data = DataGP(_recording_frame(case), min_sup=0)
        data.fit_bitmap()
        bins = data.valid_bins
        assert bins is not None
        for item, key in enumerate(item_keys):
            desc = _describe(bins[key], n)
            gi[index, item] = (
                desc.density,
                desc.avg_deviation_from_diagonal,
                desc.graph_connectivity,
                desc.singularity_score,
                desc.rank_dispersion,
            )
        seconds["gi"] += perf_counter() - start

        for pattern, (left, right) in enumerate(pattern_keys):
            start = perf_counter()
            bitmap = GP.perform_and(bins[left], bins[right], n)
            seconds["gp"] += perf_counter() - start

            start = perf_counter()
            desc = _describe(bitmap, n)
            gp_basic[index, pattern] = (bitmap.support, desc.density)
            gp_extra[index, pattern] = (
                desc.density,
                desc.graph_connectivity,
                desc.singularity_score,
                desc.rank_dispersion,
            )
            seconds["gcs"] += perf_counter() - start

    # Undefined descriptors on empty sets use zero. Density is present in
    # every corresponding baseline and distinguishes that case.
    gi = np.nan_to_num(gi.reshape(cases, -1), nan=0.0)
    gp_extra = np.nan_to_num(gp_extra, nan=0.0)
    ordinary = np.concatenate((gi, gp_basic.reshape(cases, -1)), axis=1)
    matrices = {
        "standard": standard,
        "gi": gi,
        "gp": ordinary,
        "gcs": np.concatenate((ordinary, gp_extra.reshape(cases, -1)), axis=1),
    }
    for i, descriptor in enumerate(DESCRIPTORS):
        remaining = np.delete(gp_extra, i, axis=2).reshape(cases, -1)
        matrices[f"gcs_no_{descriptor}"] = np.concatenate((ordinary, remaining), axis=1)
    return FeatureSet(matrices, seconds)
