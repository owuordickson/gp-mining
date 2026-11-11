"""Packed comparison features with a vocabulary fitted on training cases."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from so4gp.algorithms import GRAANK
from so4gp.gradual_patterns import GP, PairwiseMatrix

type Pattern = tuple[tuple[int, int], tuple[int, int]]

DESCRIPTORS = ("density", "connectivity", "singularity", "dispersion", "avg_deviation")
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
        max_patterns: Maximum vocabulary size, including both inverse
            orientations of every selected pattern family.
        min_support: Inclusive cutoff on mean training full support.
        folds: Maximum number of stratified training folds.
        c_grid: Positive inverse-regularization values for logistic
            regression.
        bootstrap: Number of stratified resamples of held-out cases
            used to estimate score and paired-difference intervals.
    """
    max_patterns: int = 24
    min_support: float = 0.2
    folds: int = 3
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    bootstrap: int = 1000

@dataclass
class FeatureSet:
    """Store recording features and the time spent extracting them.

    Attributes:
        matrices: A feature matrix for each method, with one row per
            recording and columns in the order used by build_features.
        run_time: Durations in seconds for the shared standard, GI, GP
            support, and GCS descriptor extraction stages.
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
    return pd.DataFrame(recording, columns=[f"channel_{i}" for i in range(recording.shape[1])])


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


def _gradual_features(recording: np.ndarray, min_sup: float, num_objects: int, max_pat: int) -> tuple[list, list, list]:

    def _compute_descriptors(bitmap: PairwiseMatrix, n: int) -> GP:
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

    lst_gi = []
    lst_gp_basic = []
    lst_gp_extra = []
    mine_obj = GRAANK(_recording_df(recording), min_sup=min_sup)
    mine_obj.discover(max_iteration=2, compute_descriptors=True)
    miner = mine_obj.mining_engine

    for gp_set, gi_data in (miner.valid_bins or {}).items():
        desc = _compute_descriptors(gi_data, num_objects)
        lst_gi.extend([
            desc.density,
            desc.graph_connectivity,
            desc.singularity_score,
            desc.rank_dispersion,
            desc.avg_deviation_from_diagonal
        ])

    for i, gp in enumerate(miner.gradual_patterns):
        if i > max_pat:
            break
        lst_gp_basic.extend([gp.support, gp.confidence])
        lst_gp_extra.extend([
            gp.density,
            gp.graph_connectivity,
            gp.singularity_score,
            gp.rank_dispersion,
            gp.avg_deviation_from_diagonal])
    return lst_gi, lst_gp_basic, lst_gp_extra


def build_features(sequences: np.ndarray, max_patterns: int, min_support: float) -> FeatureSet:
    """Encode each recording using a fixed vocabulary and several baselines.

    Build SO4GP GI bitmaps for both directions of each channel. For
    every selected GP, intersect the constituent bitmaps with
    GP.perform_and and compute its concordance descriptors through
    _compute_descriptors. Apply the same ordered pattern columns to every recording.

    The GI block contains density, mean separation, connectivity,
    singularity, and dispersion for each signed channel. The GP matrix
    adds support and directional density per pattern. GCS adds another
    block containing density, connectivity, singularity, and dispersion
    in DESCRIPTORS order. Each ablation removes its named descriptor
    from this additional block. Standard features form a separate
    matrix. Undefined empty-set statistics become zero in the matrices.

    Args:
        sequences: Array shaped (num_recordings, num_observations, num_channels).
        patterns: Ordered two-item patterns obtained from training data.

    Returns:
        FeatureSet containing method matrices and shared extraction
        timings. All matrices contain one row per input recording.

    Raises:
        ValueError: The input has no recordings, fewer than two observations
            or channels, or SO4GP cannot describe a concordance set.
    """
    num_recordings, num_observations, num_channels = sequences.shape
    if num_recordings == 0 or num_observations < 2 or num_channels < 2:
        raise ValueError(
            "Feature extraction requires recordings with at least two observations "
            "and two channels"
        )

    standard = np.zeros(
        (num_recordings, 3 * num_channels + num_channels * (num_channels - 1) // 2)
    )
    lst_gi = [[] for _ in range(num_recordings)]
    lst_gp_basic = [[] for _ in range(num_recordings)]
    lst_gp_extra = [[] for _ in range(num_recordings)]
    #gi = np.zeros((num_recordings, 2 * num_channels, 5))
    #gp_basic = np.zeros((num_recordings, len(patterns), 2))
    #gp_extra = np.zeros((num_recordings, len(patterns), len(DESCRIPTORS)))
    elapsed_time_dict = dict.fromkeys(("standard", "gradual_patterns"), 0.0)
    # print(f"Std:\n{standard}\nGI:\n{gi}\nGP:\n{gp_basic}\nGCS:\n{gp_extra}")
    #print(f"items:\n{item_keys}\n\npatterns:\n{pattern_keys}\n")
    #print(sequences[0])
    #return
    max_pat_count = 5

    for index, raw_recording in enumerate(sequences):
        recording = np.asarray(raw_recording, dtype=np.float64)

        # Standard Features: Mean, Std. Dev. etc.
        start = perf_counter()
        standard[index] = _standard_features(recording)
        elapsed_time_dict["standard"] += perf_counter() - start

        # GP Features: Support, Density, Deviation, Dispersion, Connectivity, Singularity
        start = perf_counter()
        l_gi, l_gp, l_gcs = _gradual_features(recording, min_support, num_observations, max_pat_count)
        lst_gi[index].extend(l_gi)
        lst_gp_basic[index].extend(l_gp)
        lst_gp_extra[index].extend(l_gcs)
        elapsed_time_dict["gradual_patterns"] += perf_counter() - start

    # Convert GP features to arrays
    gi = np.array(lst_gi)
    gp_basic = np.array(lst_gp_basic)
    gp_extra = np.array(lst_gp_extra)

    gi = np.nan_to_num(gi, nan=0.0)
    gp_basic = np.nan_to_num(gp_basic, nan=0.0)
    gp_extra = np.nan_to_num(gp_extra, nan=0.0)
    print(f"std: {standard.shape}, gi: {gi.shape}, gp_basic: {gp_basic.shape} gp_extra: {gp_extra.shape}")

    #gi = np.nan_to_num(gi.reshape(num_recordings, -1), nan=0.0)
    #gp_extra = np.nan_to_num(gp_extra, nan=0.0)
    #ordinary = np.concatenate((gi, gp_basic.reshape(num_recordings, -1)), axis=1)
    matrices = {
        "standard": standard,
        "gi": gi,
        "gp": gp_basic,
        "gcs": gp_extra,
    }
    for i, descriptor in enumerate(DESCRIPTORS):
        remaining = np.delete(gp_extra, i, axis=1)
        matrices[f"gcs_no_{descriptor}"] = remaining

    return FeatureSet(matrices, elapsed_time_dict)
