"""Synthetic tasks with independent train and test sequences."""

import numpy as np

_LENGTH = 64
_CHANNELS = 6
_PLANTED = 3


def _ar_segment(innovations: np.ndarray) -> np.ndarray:
    values = innovations.copy()
    for i in range(1, len(values)):
        values[i] = 0.7 * values[i - 1] + np.sqrt(1 - 0.7**2) * values[i]
    return values


def _contaminate(
    values: np.ndarray, rng: np.random.Generator, noise: float
) -> np.ndarray:
    measurement = rng.standard_normal(values.shape)
    spikes = rng.random(values.shape) < 0.015
    return values + noise * measurement + spikes * 3 * rng.standard_t(3, values.shape)


def _share(innovations: np.ndarray, rng: np.random.Generator) -> None:
    common = rng.standard_normal((len(innovations), 1))
    innovations[:, :_PLANTED] = (
        0.9 * common + np.sqrt(1 - 0.9**2) * innovations[:, :_PLANTED]
    )


def _shared_case(label: int, rng: np.random.Generator, noise: float) -> np.ndarray:
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
    """Arrange equal amounts of shared behavior in long or short episodes."""
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
    """Return independent splits with balanced classes in case/time/channel order."""
    generators = {"shared": _shared_case, "timing": _timing_case}
    if regime not in generators:
        raise ValueError(f"Unknown synthetic regime: {regime!r}")
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    if min(train_size, test_size) < 2:
        raise ValueError("Each split needs at least two cases")

    def make_split(size: int, child_seed: np.random.SeedSequence):
        rng = np.random.default_rng(child_seed)
        labels = rng.permutation(np.arange(size) % 2)
        cases = np.stack([generators[regime](int(y), rng, noise) for y in labels])
        return cases, labels

    train_seed, test_seed = np.random.SeedSequence(seed).spawn(2)
    return (*make_split(train_size, train_seed), *make_split(test_size, test_seed))
