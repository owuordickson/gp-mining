"""Encode each recording with independent standard, GI, GP, and GCS features."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from so4gp.algorithms import GRAANK
from so4gp.gradual_patterns import GI, GP, PairwiseMatrix

type Pattern = tuple[tuple[int, int], tuple[int, int]]

DESCRIPTORS = ("density", "connectivity", "singularity", "dispersion", "avg_separation")
METHODS = ("standard", "gi", "gp", "gcs", *(f"gcs_no_{d}" for d in DESCRIPTORS))

CONTRASTS = (
    ("gp", "gi"),
    ("gcs", "gp"),
    *(("gcs", f"gcs_no_{d}") for d in DESCRIPTORS),
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configure pattern selection, classifier tuning, and uncertainty.

    Every feature method uses the same folds and C grid. The evaluator
    caps the requested folds by the smallest training class.

    Attributes:
        max_patterns: Maximum number of two-item GRAANK results retained
            per recording. Unused pattern slots are filled with zeros.
        min_support: Minimum support passed to GRAANK for each recording.
        folds: Maximum number of stratified training folds.
        c_grid: Positive inverse-regularization values for logistic
            regression.
        bootstrap: Number of stratified resamples of held-out cases
            used to estimate score and paired-difference intervals.
    """

    max_patterns: int = 24
    min_support: float = 0.0
    folds: int = 3
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    bootstrap: int = 1000


@dataclass
class FeatureSet:
    """Store recording features and the time spent extracting them.

    Attributes:
        matrices: A feature matrix for each method, with one row per
            recording and columns in the order used by build_features.
        run_time: Durations in seconds for standard features and the
            gradual_patterns stage, which includes mining and descriptors.
    """

    matrices: dict[str, np.ndarray]
    run_time: dict[str, float]


def _recording_df(recording: np.ndarray) -> pd.DataFrame:
    # Numeric column labels would be interpreted by DataGP as a data row.
    """Give one recording the named columns expected by SO4GP.

    Convert the observation-by-channel array to a DataFrame with names
    such as channel_0. These names prevent DataGP from interpreting
    numeric column labels as an additional observation.
    """
    return pd.DataFrame(
        recording, columns=[f"channel_{i}" for i in range(recording.shape[1])]
    )


def _standard_features(recording: np.ndarray) -> np.ndarray:
    """Summarize one recording using moments, trends, and correlations.

    Return channel means, population standard deviations, and
    least-squares slopes over a time axis spanning -0.5 to 0.5,
    followed by Pearson correlations for distinct channel pairs in
    upper-triangle order. Constant channels contribute zero
    correlations. The output is a vector with 3*m + m*(m-1)/2 values
    for an input shaped (observations, m).
    """
    means = recording.mean(axis=0)
    centered = recording - means
    time = np.linspace(-0.5, 0.5, len(recording))
    slopes = time @ centered / (time @ time)
    norms = np.linalg.norm(centered, axis=0)
    unit = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)
    correlation = unit.T @ unit
    upper = np.triu_indices(recording.shape[1], k=1)
    return np.concatenate((means, recording.std(axis=0), slopes, correlation[upper]))


def _gradual_features(
    recording: np.ndarray, min_sup: float, num_objects: int, max_pat: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mine one recording and extract its independent GI, GP, and GCS blocks.

    Run SO4GP's two-item Apriori GRAANK with descriptor computation enabled.
    Store retained GIs in fixed channel/sign positions, with '+' before '-'.
    Take up to max_pat GPs in the miner's output order. Each GP contributes
    support and confidence to the GP block and the five DESCRIPTORS values
    to the GCS block. GP and GCS slots refer to the same mined result.

    Args:
        recording: Array shaped (num_observations, num_channels).
        min_sup: Support threshold passed directly to GRAANK.
        num_objects: Observation count used to interpret GI bitmaps.
        max_pat: Maximum number of returned GP results to encode.

    Returns:
        Flattened GI, GP, and GCS arrays. The GI array reserves both signs
        of every channel. GP/GCS arrays contain the mined results, which
        build_features pads to the configured maximum.
    """

    def _compute_descriptors(bitmap: PairwiseMatrix, n: int) -> GP:
        """Extract canonical pairs and compute their descriptors with SO4GP.

        Use GP.gen_gradual_concordance_set to retain supporting i < j pairs,
        then delegate to GP.compute_descriptors. Empty sets have zero
        density and n components, with undefined statistics stored as NaN.

        Raises:
            ValueError: SO4GP rejects the extracted concordance set.
        """
        pairs = GP.gen_gradual_concordance_set(bitmap.packed_bin_mat, n)
        gp = GP()
        if not gp.compute_descriptors(pairs, n):
            raise ValueError("Invalid concordance pairs")
        return gp

    gi_features = np.zeros((2 * recording.shape[1], len(DESCRIPTORS)))
    lst_gp_basic = []
    lst_gp_extra = []
    mine_obj = GRAANK(_recording_df(recording), min_sup=min_sup)
    mine_obj.discover(max_iteration=2, compute_descriptors=True)
    miner = mine_obj.mining_engine

    for gi_key, gi_data in (miner.valid_bins or {}).items():
        item = GI.from_string(gi_key)
        slot = 2 * item.attribute_col + int(item.symbol == "-")
        desc = _compute_descriptors(gi_data, num_objects)
        gi_features[slot] = (
            desc.density,
            desc.graph_connectivity,
            desc.singularity_score,
            desc.rank_dispersion,
            desc.average_separation,
        )

    for gp in (miner.gradual_patterns or [])[:max_pat]:
        lst_gp_basic.extend((gp.support, gp.confidence))
        lst_gp_extra.extend(
            (
                gp.density,
                gp.graph_connectivity,
                gp.singularity_score,
                gp.rank_dispersion,
                gp.average_separation,
            )
        )
    return (
        gi_features.ravel(),
        np.asarray(lst_gp_basic, dtype=float),
        np.asarray(lst_gp_extra, dtype=float),
    )


def build_features(
    sequences: np.ndarray, max_patterns: int, min_support: float
) -> FeatureSet:
    """Encode recordings using independent standard, GI, GP, and GCS matrices.

    For each recording, compute standard summaries and call _gradual_features
    to mine GRAANK patterns and obtain SO4GP descriptors. Each matrix has one
    row per recording. GI columns have fixed channel/sign identities. GP and
    GCS slot i describes the i-th pattern returned for that recording, so
    pattern identities can differ across recordings.

    The GI matrix contains five descriptors per signed channel. The GP
    matrix contains support and confidence per pattern. The GCS matrix
    contains density, connectivity, singularity, dispersion, and mean index
    separation per pattern, in DESCRIPTORS order. Each ablation removes its
    named GCS descriptor from every pattern slot. The avg_separation
    descriptor is the mean j - i over canonical pairs, in observation steps.

    Pruned GI slots and unused GP/GCS slots are zero-filled. Undefined
    statistics of empty concordance sets are converted from NaN to zero.
    Retained GIs with empty sets keep one component per observation.

    Args:
        sequences: Finite array shaped
            (num_recordings, num_observations, num_channels).
        max_patterns: Positive maximum number of mined patterns per recording.
        min_support: Support threshold in [0, 1], passed to SO4GP GRAANK.

    Returns:
        FeatureSet containing the method matrices and the standard and
        gradual_patterns extraction durations in seconds.

    Raises:
        ValueError: The dimensions, pattern limit, or threshold are invalid,
            or SO4GP cannot compute a concordance descriptor.
    """
    if max_patterns < 1:
        raise ValueError("max_patterns must be a positive integer")
    if not 0 <= min_support <= 1:
        raise ValueError("min_support must be in [0, 1]")
    num_recordings, num_observations, num_channels = sequences.shape
    if num_recordings == 0 or num_observations < 2 or num_channels < 2:
        raise ValueError(
            "Feature extraction requires recordings with at least two observations "
            "and two channels"
        )

    max_patterns = num_channels * (num_channels - 1)
    standard = np.zeros(
        (num_recordings, 3 * num_channels + num_channels * (num_channels - 1) // 2)
    )
    gi_only = np.zeros((num_recordings, 2 * num_channels * len(DESCRIPTORS)))
    gp_basic = np.zeros((num_recordings, max_patterns * 2))
    gp_extra = np.zeros((num_recordings, max_patterns * len(DESCRIPTORS)))
    elapsed_time_dict = dict.fromkeys(("standard", "gradual_patterns"), 0.0)

    for index, raw_recording in enumerate(sequences):
        recording = np.asarray(raw_recording, dtype=np.float64)

        start = perf_counter()
        standard[index] = _standard_features(recording)
        elapsed_time_dict["standard"] += perf_counter() - start

        start = perf_counter()
        arr_gi, arr_gp, arr_gcs = _gradual_features(
            recording, min_support, num_observations, max_patterns
        )
        gi_only[index] = arr_gi
        gp_basic[index, : arr_gp.shape[0]] = arr_gp
        gp_extra[index, : arr_gcs.shape[0]] = arr_gcs
        elapsed_time_dict["gradual_patterns"] += perf_counter() - start
        #print(f"GI: {arr_gi.shape}, GP: {arr_gp.shape}, GCS: {arr_gcs.shape}")

    gi_only = np.nan_to_num(gi_only, nan=0.0)
    gp_basic = np.nan_to_num(gp_basic, nan=0.0)
    gp_extra = np.nan_to_num(gp_extra, nan=0.0)
    matrices = {
        "standard": standard,
        "gi": gi_only,
        "gp": gp_basic,
        "gcs": gp_extra,
    }
    gcs_blocks = gp_extra.reshape(num_recordings, max_patterns, len(DESCRIPTORS))
    for column, descriptor in enumerate(DESCRIPTORS):
        remaining = np.delete(gcs_blocks, column, axis=2).reshape(num_recordings, -1)
        matrices[f"gcs_no_{descriptor}"] = remaining

    return FeatureSet(matrices, elapsed_time_dict)
