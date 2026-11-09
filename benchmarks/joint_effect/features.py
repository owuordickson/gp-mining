"""Packed comparison features with a vocabulary fitted on training cases."""

from dataclasses import dataclass
from itertools import combinations
from time import perf_counter

import numpy as np

from so4gp.gradual_patterns import GP

type Pattern = tuple[tuple[int, int], tuple[int, int]]

DESCRIPTORS = ("connectivity", "singularity", "dispersion")
METHODS = ("standard", "gi", "gp", "gcs", *(f"gcs_no_{d}" for d in DESCRIPTORS))


@dataclass
class FeatureSet:
    matrices: dict[str, np.ndarray]
    seconds: dict[str, float]


def _packed_items(case: np.ndarray) -> np.ndarray:
    n, channels = case.shape
    packed = np.empty((2 * channels, (n * n + 7) // 8), dtype=np.uint8)
    for channel, column in enumerate(case.T):
        forward = column[:, None] < column[None, :]
        packed[2 * channel] = np.packbits(forward, bitorder="big")
        packed[2 * channel + 1] = np.packbits(forward.T, bitorder="big")
    return packed


def _item_index(channel: int, sign: int) -> int:
    return 2 * channel + int(sign < 0)


def _indices(patterns: tuple[Pattern, ...]) -> tuple[np.ndarray, np.ndarray]:
    left = np.array([_item_index(*pattern[0]) for pattern in patterns], dtype=int)
    right = np.array([_item_index(*pattern[1]) for pattern in patterns], dtype=int)
    return left, right


def select_patterns(
    sequences: np.ndarray, *, max_patterns: int, min_support: float
) -> tuple[Pattern, ...]:
    """Rank inverse families by mean full support and keep both orientations."""
    if max_patterns < 2 or max_patterns % 2:
        raise ValueError("max_patterns must be a positive even integer")
    if not 0 <= min_support <= 1:
        raise ValueError("min_support must be in [0, 1]")
    cases, n, channels = sequences.shape
    if cases == 0 or n < 2 or channels < 2:
        raise ValueError("Pattern selection needs cases with >=2 times and channels")

    families = tuple(
        ((a, 1), (b, sign))
        for a, b in combinations(range(channels), 2)
        for sign in (1, -1)
    )
    left, right = _indices(families)
    totals = np.zeros(len(families), dtype=np.float64)
    for case in sequences:
        packed = _packed_items(case)
        for start in range(0, len(families), 128):
            batch = slice(start, start + 128)
            intersections = packed[left[batch]] & packed[right[batch]]
            totals[batch] += np.bitwise_count(intersections).sum(axis=1)

    means = totals / (cases * GP.pair_count(n))
    # Stable ordering makes ties reproducible without looking at test cases.
    ranked = np.argsort(-means, kind="stable")[: max_patterns // 2]
    selected = []
    for index in ranked:
        if means[index] < min_support:
            continue
        pattern = families[index]
        (a, sign_a), (b, sign_b) = pattern
        selected.extend((pattern, ((a, -sign_a), (b, -sign_b))))
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


def _describe(pairs: np.ndarray, n: int) -> GP:
    gp = GP()
    if not gp.compute_descriptors(pairs, n):
        raise ValueError("Invalid concordance pairs")
    return gp


def build_features(sequences: np.ndarray, patterns: tuple[Pattern, ...]) -> FeatureSet:
    """Build nested baselines and remove one GP descriptor block per ablation."""
    cases, n, channels = sequences.shape
    if cases == 0 or n < 2 or channels < 2:
        raise ValueError("Feature extraction needs cases with >=2 times and channels")

    total_pairs = GP.pair_count(n)
    left, right = _indices(patterns)
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
        packed = _packed_items(case)
        for item, bits in enumerate(packed):
            desc = _describe(GP.gen_gradual_concordance_set(bits, n), n)
            gi[index, item] = (
                desc.density,
                desc.avg_deviation_from_diagonal,
                desc.graph_connectivity,
                desc.singularity_score,
                desc.rank_dispersion,
            )
        seconds["gi"] += perf_counter() - start

        start = perf_counter()
        intersections = packed[left] & packed[right]
        supports = np.bitwise_count(intersections).sum(axis=1) / total_pairs
        pairs = [GP.gen_gradual_concordance_set(bits, n) for bits in intersections]
        gp_basic[index, :, 0] = supports
        gp_basic[index, :, 1] = [len(pair_set) / total_pairs for pair_set in pairs]
        seconds["gp"] += perf_counter() - start

        start = perf_counter()
        for pattern, pair_set in enumerate(pairs):
            desc = _describe(pair_set, n)
            gp_extra[index, pattern] = (
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
