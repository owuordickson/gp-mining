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
    matrices: dict[str, np.ndarray]
    seconds: dict[str, float]


def _recording_frame(case: np.ndarray) -> pd.DataFrame:
    # Numeric column labels would be interpreted by DataGP as a data row.
    return pd.DataFrame(case, columns=[f"channel_{i}" for i in range(case.shape[1])])


def _item_key(channel: int, sign: int) -> str:
    return f"{channel}{'+' if sign > 0 else '-'}"


def _inverse(pattern: Pattern) -> Pattern:
    (a, sign_a), (b, sign_b) = pattern
    return (a, -sign_a), (b, -sign_b)


def select_patterns(
    sequences: np.ndarray, *, max_patterns: int, min_support: float
) -> tuple[Pattern, ...]:
    """Discover with GRAANK and rank inverse families by mean full support."""
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
    pairs = GP.gen_gradual_concordance_set(bitmap.packed_bin_mat, n)
    gp = GP()
    if not gp.compute_descriptors(pairs, n):
        raise ValueError("Invalid concordance pairs")
    return gp


def build_features(sequences: np.ndarray, patterns: tuple[Pattern, ...]) -> FeatureSet:
    """Build nested baselines and remove one GP descriptor block per ablation."""
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
