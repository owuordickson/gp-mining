"""Tune matched classifiers and evaluate them on an untouched test split."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .features import DESCRIPTORS, METHODS, build_features, select_patterns

CONTRASTS = (
    ("gp", "gi"),
    ("gcs", "gp"),
    *(("gcs", f"gcs_no_{d}") for d in DESCRIPTORS),
)


@dataclass(frozen=True)
class Protocol:
    max_patterns: int = 24
    min_support: float = 0.2
    folds: int = 3
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    bootstrap: int = 1000


def _classifier(c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))


def _metrics(y: np.ndarray, pred: np.ndarray, classes: int) -> np.ndarray:
    confusion = np.bincount(y * classes + pred, minlength=classes**2)
    confusion = confusion.reshape(classes, classes)
    diagonal = confusion.diagonal()
    positives = confusion.sum(axis=1)
    predictions = confusion.sum(axis=0)
    return np.array(
        [
            np.mean(diagonal / positives),
            np.mean(2 * diagonal / (positives + predictions)),
        ]
    )


def _intervals(y, predictions, classes, draws, seed):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == label) for label in range(classes)]
    samples = np.empty((draws, len(METHODS), 2))
    for draw in range(draws):
        index = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        for method, pred in enumerate(predictions):
            samples[draw, method] = _metrics(y[index], pred[index], classes)
    return samples


def _score_columns(point: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(samples, (0.025, 0.975), axis=0)
    fields = {}
    for i, metric in enumerate(("balanced_accuracy", "macro_f1")):
        fields[metric] = float(point[i])
        fields[f"{metric}_ci_low"] = float(low[i])
        fields[f"{metric}_ci_high"] = float(high[i])
    return fields


def _check_inputs(x_train, y_train, x_test, y_test, protocol):
    if x_train.ndim != 3 or x_test.ndim != 3 or x_train.shape[1:] != x_test.shape[1:]:
        raise ValueError(
            "Expected train and test with matching time/channel dimensions"
        )
    if y_train.ndim != 1 or y_test.ndim != 1:
        raise ValueError("Expected one-dimensional labels")
    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("Each sequence needs exactly one label")
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Benchmark inputs must be finite")
    labels, counts = np.unique(y_train, return_counts=True)
    if len(labels) < 2 or not np.array_equal(labels, np.unique(y_test)):
        raise ValueError("Train and test must contain the same two or more classes")
    if protocol.folds < 2 or counts.min() < protocol.folds:
        raise ValueError("Each training class needs at least as many cases as CV folds")
    if protocol.bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    if not protocol.c_grid or any(
        not np.isfinite(c) or c <= 0 for c in protocol.c_grid
    ):
        raise ValueError("The C grid must contain finite positive values")
    return labels


def _prepare(x_train, x_eval, protocol):
    start = perf_counter()
    patterns = select_patterns(
        x_train, max_patterns=protocol.max_patterns, min_support=protocol.min_support
    )
    mining_seconds = perf_counter() - start
    train = build_features(x_train, patterns)
    evaluation = build_features(x_eval, patterns)
    return (
        patterns,
        train,
        evaluation,
        {
            "pattern_mining_seconds": mining_seconds,
            "train_feature_seconds": train.seconds,
            "evaluation_feature_seconds": evaluation.seconds,
        },
    )


def evaluate_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    protocol: Protocol = Protocol(),
) -> dict:
    """Tune on fold-local training data, refit on training, then score test once."""
    labels = _check_inputs(x_train, y_train, x_test, y_test, protocol)
    y_train, y_test = np.searchsorted(labels, y_train), np.searchsorted(labels, y_test)
    grid = tuple(sorted(set(protocol.c_grid)))
    validation = {method: np.zeros(len(grid)) for method in METHODS}
    tune_seconds = dict.fromkeys(METHODS, 0.0)
    fold_details = []
    cv = StratifiedKFold(n_splits=protocol.folds, shuffle=True, random_state=seed)
    total_start = perf_counter()

    # Limit BLAS threads so all methods share the same compute configuration.
    with threadpool_limits(limits=1):
        for train_idx, validation_idx in cv.split(x_train, y_train):
            patterns, train, heldout, timing = _prepare(
                x_train[train_idx], x_train[validation_idx], protocol
            )
            fold_details.append({"patterns": patterns, **timing})
            for method in METHODS:
                start = perf_counter()
                for index, c in enumerate(grid):
                    model = _classifier(c)
                    model.fit(train.matrices[method], y_train[train_idx])
                    pred = model.predict(heldout.matrices[method])
                    validation[method][index] += balanced_accuracy_score(
                        y_train[validation_idx], pred
                    )
                tune_seconds[method] += perf_counter() - start

        patterns, train, test, timing = _prepare(x_train, x_test, protocol)
        predictions = []
        model_details = []
        for method in METHODS:
            best = int(np.argmax(validation[method]))
            model = _classifier(grid[best])
            start = perf_counter()
            model.fit(train.matrices[method], y_train)
            fit_seconds = perf_counter() - start
            start = perf_counter()
            predictions.append(model.predict(test.matrices[method]))
            predict_seconds = perf_counter() - start
            model_details.append(
                {
                    "method": method,
                    "features": train.matrices[method].shape[1],
                    "c": grid[best],
                    "cv_balanced_accuracy": float(
                        validation[method][best] / protocol.folds
                    ),
                    "tune_seconds": tune_seconds[method],
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                }
            )

    points = np.stack([_metrics(y_test, pred, len(labels)) for pred in predictions])
    samples = _intervals(y_test, predictions, len(labels), protocol.bootstrap, seed)
    metrics = [
        {**detail, **_score_columns(points[i], samples[:, i])}
        for i, detail in enumerate(model_details)
    ]
    contrasts = []
    for method, baseline in CONTRASTS:
        i, j = METHODS.index(method), METHODS.index(baseline)
        contrasts.append(
            {
                "method": method,
                "baseline": baseline,
                **_score_columns(points[i] - points[j], samples[:, i] - samples[:, j]),
            }
        )
    return {
        "metrics": metrics,
        "contrasts": contrasts,
        "patterns": patterns,
        "training_cases": len(x_train),
        "test_cases": len(x_test),
        "observations": x_train.shape[1],
        "channels": x_train.shape[2],
        "classes": labels.tolist(),
        "cv": fold_details,
        "final": timing,
        "total_seconds": perf_counter() - total_start,
    }
