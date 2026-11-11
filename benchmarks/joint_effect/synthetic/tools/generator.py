"""Synthetic tasks with independent train and test sequences."""

import numpy as np

_LENGTH = 64
_CHANNELS = 6
_PLANTED = 3


def _ar_segment(innovations: np.ndarray) -> np.ndarray:
    """Transform innovations into a stationary first-order AR segment.

    Copy the input and recursively combine each previous output with
    the next innovation using coefficient 0.7 and innovation scale
    sqrt(1 - 0.7**2). Process channels independently and return an
    array with the same time/channel shape.
    """
    values = innovations.copy()
    for i in range(1, len(values)):
        values[i] = 0.7 * values[i - 1] + np.sqrt(1 - 0.7**2) * values[i]
    return values


def _contaminate(
    values: np.ndarray, rng: np.random.Generator, noise: float
) -> np.ndarray:
    """Add measurement noise and occasional heavy-tailed observations.

    Add standard Gaussian samples scaled by noise. Independently mark
    1.5% of entries for outliers drawn from a Student-t distribution
    with three degrees of freedom and multiplied by three. Outliers
    also occur when noise is zero. Return the perturbed array.
    """
    measurement = rng.standard_normal(values.shape)
    spikes = rng.random(values.shape) < 0.015
    return values + noise * measurement + spikes * 3 * rng.standard_t(3, values.shape)


def _share(innovations: np.ndarray, rng: np.random.Generator) -> None:
    """Add a shared innovation to the planted channel group in place.

    Draw one common Gaussian value per time step and mix it into the
    first _PLANTED channels with weight 0.9. Scale their independent
    innovations by sqrt(1 - 0.9**2) to retain unit marginal variance.
    The supplied array can be a view of a selected time interval.
    """
    common = rng.standard_normal((len(innovations), 1))
    innovations[:, :_PLANTED] = (
        0.9 * common + np.sqrt(1 - 0.9**2) * innovations[:, :_PLANTED]
    )


def _shared_case(label: int, rng: np.random.Generator, noise: float) -> np.ndarray:
    """Generate one recording with or without an interval of shared behavior.

    Choose a random interval with the same start and duration rules
    for both labels. A positive label mixes a common innovation into
    the planted channels inside that interval. Build independent AR
    segments before, inside, and after it, then apply measurement
    noise and outliers. Return (_LENGTH, _CHANNELS) observations.
    """
    start = int(rng.integers(8, 25))
    stop = start + int(rng.integers(20, 33))
    innovations = rng.standard_normal((_LENGTH, _CHANNELS))
    if label:
        _share(innovations[start:stop], rng)
    values = np.concatenate(
        [_ar_segment(part) for part in np.split(innovations, [start, stop])]
    )
    return _contaminate(values, rng, noise)


def _timing_case(label: int, rng: np.random.Generator, noise: float) -> np.ndarray:
    """Arrange equal amounts of shared behavior in long or short episodes.

    Generate independent and shared halves, apply AR dependence and
    contamination, then bound them with tanh and separate their value
    ranges. A positive label places shared rows in one middle episode.
    A zero label uses repeated two-row shared episodes. Preserve row
    order within each state while assigning it to the chosen layout.

    For the same generated halves, both layouts retain full support
    and directional density while changing the timing of the pairs.

    Returns:
        A recording shaped (_LENGTH, _CHANNELS).
    """
    half = _LENGTH // 2
    quarter = _LENGTH // 4
    independent = rng.standard_normal((half, _CHANNELS))
    shared = rng.standard_normal((half, _CHANNELS))
    _share(shared, rng)
    # Bound perturbations to keep cross-state comparisons ordered.
    low = np.tanh(_contaminate(_ar_segment(independent), rng, noise)) - 1.25
    high = np.tanh(_contaminate(_ar_segment(shared), rng, noise)) + 1.25
    layout = (
        np.repeat([0, 1, 0], [quarter, half, quarter])
        if label
        else np.tile([0, 1, 1, 0], quarter)
    )
    values = np.empty((_LENGTH, _CHANNELS))
    values[layout == 0] = low
    values[layout == 1] = high
    return values


def synthetic_split(
    regime: str, *, seed: int, noise: float, train_size: int, test_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate reproducible training and test splits for a synthetic task.

    Choose the shared-interval or timing generator and spawn separate
    child seeds for the two splits. Within each split, shuffle binary
    labels balanced to within one case, then independently generate
    a recording for each label.

    Args:
        regime: Either 'shared' or 'timing'.
        seed: Parent seed used to create independent split generators.
        noise: Finite, nonnegative Gaussian measurement-noise scale.
        train_size: Number of training recordings.
        test_size: Number of test recordings.

    Returns:
        x_train, y_train, x_test, y_test. Recording arrays use
        (cases, _LENGTH, _CHANNELS) order, and label arrays contain
        the generating process's binary outcomes.

    Raises:
        ValueError: The regime or noise is invalid, or either split
            contains fewer than two cases.
    """
    generators = {"shared": _shared_case, "timing": _timing_case}
    if regime not in generators:
        raise ValueError(f"Unknown synthetic regime: {regime!r}")
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    if min(train_size, test_size) < 2:
        raise ValueError("Each split needs at least two cases")

    def make_split(size: int, child_seed: np.random.SeedSequence):
        """Generate one labelled split from its dedicated child seed.

        Randomly permute alternating binary labels and stack one new
        regime-specific recording per label using the same split generator.
        """
        rng = np.random.default_rng(child_seed)
        labels = rng.permutation(np.arange(size) % 2)
        cases = np.stack([generators[regime](int(y), rng, noise) for y in labels])
        return cases, labels

    train_seed, test_seed = np.random.SeedSequence(seed).spawn(2)
    return (*make_split(train_size, train_seed), *make_split(test_size, test_seed))
