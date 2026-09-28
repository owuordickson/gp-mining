# Import libraries

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def generate_timegraph_mock(length=1000):
    """Simulates a TimeGraph dataset with an explicit lag: X -> Y (lag=2)"""
    np.random.seed(42)
    X = np.random.normal(0, 1, length)
    Y = np.zeros(length)
    # Introducing temporal causal relationship with lag 2
    for t in range(2, length):
        Y[t] = 0.7 * X[t - 2] + np.random.normal(0, 0.5)

    df = pd.DataFrame({"X": X, "Y": Y})

    # Ground truth matrix: row causes column (X causes Y, so index 0 -> index 1)
    ground_truth = np.zeros((2, 2))
    ground_truth[0, 1] = 1
    return df, ground_truth


def generate_causalrivers_mock(length=1000):
    """Simulates a CausalRivers hydro-station network: up_stream -> down_stream"""
    np.random.seed(101)
    up_stream = np.sin(np.linspace(0, 50, length)) + np.random.normal(0, 0.2, length)
    down_stream = np.zeros(length)
    # up_stream water takes 3 steps to hit downstream measuring tool
    for t in range(3, length):
        down_stream[t] = 0.85 * up_stream[t - 3] + np.random.normal(0, 0.1)

    df = pd.DataFrame({"up_stream": up_stream, "down_stream": down_stream})
    ground_truth = np.zeros((2, 2))
    ground_truth[0, 1] = 1
    return df, ground_truth


def evaluate_predictions(true_matrix, pred_matrix):
    """Computes Precision, Recall, and F1 metrics for the flattened graph structures"""
    y_true = true_matrix.flatten()
    y_pred = pred_matrix.flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return precision, recall, f1
