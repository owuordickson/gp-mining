"""Tune matched classifiers and evaluate them on an untouched test split."""

from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .features import Configs, CONTRASTS, METHODS, build_features, select_patterns


def _classifier(c: float):
    """Create the shared classifier pipeline for one regularization value.

    Place StandardScaler before logistic regression so fitting learns
    scaling statistics from the supplied training features. Use c as
    the inverse regularization strength and allow 2,000 optimizer
    iterations.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))


def _metrics(y: np.ndarray, pred: np.ndarray, classes: int) -> np.ndarray:
    """Calculate balanced accuracy and macro-F1 from encoded predictions.

    Count true/predicted label combinations into a confusion matrix,
    then average class recall and class F1. Inputs use integer labels
    from zero through classes-1, and y must contain every class.
    Return a two-value array in balanced-accuracy, macro-F1 order.
    """
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
    """Bootstrap held-out scores while preserving the observed class sizes.

    Sample each true-label group with replacement using the seeded
    generator. Reuse each sampled index vector for every method so
    score differences can be evaluated on paired observations.

    Args:
        y: True labels encoded as integers from zero to classes-1.
        predictions: Prediction arrays in METHODS order.
        classes: Number of classes, all represented in y.
        draws: Number of bootstrap resamples.
        seed: Seed controlling the resampling generator.

    Returns:
        An array shaped (draws, methods, 2), whose last axis contains
        balanced accuracy and macro-F1.
    """
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == label) for label in range(classes)]
    samples = np.empty((draws, len(METHODS), 2))
    for draw in range(draws):
        index = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        for method, pred in enumerate(predictions):
            samples[draw, method] = _metrics(y[index], pred[index], classes)
    return samples


def _score_columns(point: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    """Attach percentile intervals to a pair of point estimates.

    Take the 2.5th and 97.5th percentiles of the bootstrap samples and
    return named balanced-accuracy and macro-F1 fields with their
    lower and upper bounds. The same calculation handles method
    scores and paired score differences.
    """
    low, high = np.quantile(samples, (0.025, 0.975), axis=0)
    fields = {}
    for i, metric in enumerate(("balanced_accuracy", "macro_f1")):
        fields[metric] = float(point[i])
        fields[f"{metric}_ci_low"] = float(low[i])
        fields[f"{metric}_ci_high"] = float(high[i])
    return fields


def _check_inputs(x_train, y_train, x_test, y_test, configs) -> tuple[np.ndarray, int]:
    """Validate a split and determine its feasible stratified fold count.

    Check sequence dimensions, finite values, label lengths, matching
    class sets, and the fold, bootstrap, and C-grid settings. Require
    two training recordings per class, then cap the requested folds
    at the smallest class size.

    Returns:
        The sorted original class labels and the effective fold count.

    Raises:
        ValueError: An input or configuration setting fails these checks.
    """
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
    if configs.folds < 2:
        raise ValueError("CV folds must be at least two")
    min_class_count = int(counts.min())
    if min_class_count < 2:
        raise ValueError(
            "Each training class needs at least two cases for stratified CV"
        )
    if configs.bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    if not configs.c_grid or any(
        not np.isfinite(c) or c <= 0 for c in configs.c_grid
    ):
        raise ValueError("The C grid must contain finite positive values")
    return labels, min(configs.folds, min_class_count)


def _prepare(x_train, x_eval, configs):
    """Mine patterns on training cases and encode training and held-out cases.

    Select the vocabulary using x_train alone, then pass that same
    vocabulary to build_features for both arrays. Measure mining
    duration separately from the shared feature extraction stages.

    Returns:
        A tuple containing patterns, training features, evaluation
        features, and a dictionary of mining and extraction timings.
    """
    start = perf_counter()
    patterns = select_patterns(
        x_train, max_patterns=configs.max_patterns, min_support=configs.min_support
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
    configs: Configs = Configs(),
) -> dict:
    """Tune matched classifiers on training folds and score a held-out split.

    Encode labels as consecutive integers and create seeded stratified
    folds, capped by the smallest training class. Each fold mines its
    own vocabulary and fits its scaler using the fold's training cases.
    Select C by mean validation balanced accuracy, resolving ties with
    the smallest C. Then select patterns on all training cases, refit
    each classifier, and predict the supplied test recordings.

    Use one BLAS thread during fitting. Estimate 95% score intervals
    and paired method differences with stratified test resampling.

    Args:
        x_train: Training array shaped (cases, observations, channels).
        y_train: One class label per training recording.
        x_test: Held-out array with the same observation/channel shape.
        y_test: One class label per held-out recording.
        seed: Seed for fold shuffling and bootstrap resampling.
        configs: Shared vocabulary, tuning, and resampling settings.

    Returns:
        A report dictionary containing method metrics, paired
        contrasts, the final vocabulary, dataset dimensions, original
        class labels, the effective fold count, fold-level and final
        extraction details, and elapsed seconds.

    Raises:
        ValueError: Input validation, pattern selection, or feature
            extraction fails.
    """
    labels, cv_folds = _check_inputs(x_train, y_train, x_test, y_test, configs)
    y_train, y_test = np.searchsorted(labels, y_train), np.searchsorted(labels, y_test)
    grid = tuple(sorted(set(configs.c_grid)))
    validation = {method: np.zeros(len(grid)) for method in METHODS}
    tune_seconds = dict.fromkeys(METHODS, 0.0)
    fold_details = []
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    total_start = perf_counter()

    # Limit BLAS threads so all methods share the same compute configuration.
    with threadpool_limits(limits=1):
        for train_idx, validation_idx in cv.split(x_train, y_train):
            patterns, train, heldout, timing = _prepare(
                x_train[train_idx], x_train[validation_idx], configs
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

        patterns, train, test, timing = _prepare(x_train, x_test, configs)
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
                    "cv_balanced_accuracy": float(validation[method][best] / cv_folds),
                    "tune_seconds": tune_seconds[method],
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                }
            )

    points = np.stack([_metrics(y_test, pred, len(labels)) for pred in predictions])
    samples = _intervals(y_test, predictions, len(labels), configs.bootstrap, seed)
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
        "cv_folds": cv_folds,
        "cv": fold_details,
        "final": timing,
        "total_seconds": perf_counter() - total_start,
    }
