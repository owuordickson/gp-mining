# Import libraries

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from baseline_methods import (
    run_classical_statistics,
    run_granger_causality,
    run_pc_algorithm,
    run_pcmci_tigramite,
    run_t_graank,
)


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



def main() -> int:
    datasets = {
        "TimeGraph (Synthetic)": generate_timegraph_mock(),
        "CausalRivers (Real-World)": generate_causalrivers_mock(),
    }

    results_master = []

    for dataset_name, (df, ground_truth) in datasets.items():
        # print(f"\nEvaluating Frameworks on: {dataset_name}...")
        # print(f"{df.head()}\n")

        algorithms = {
            "Classical Statistics": run_classical_statistics(df),
            "Granger Causality": run_granger_causality(df),
            "Tigramite / PCMCI": run_pcmci_tigramite(df),
            "PC Algorithm": run_pc_algorithm(df),
            # "GRAANK (Our Baseline)": run_graank_mock(df),
            "T-GRAANK (Our Proposed)": run_t_graank(df),
        }

        for algo_name, pred_matrix in algorithms.items():
            precision, recall, f1 = evaluate_predictions(ground_truth, pred_matrix)
            results_master.append(
                {
                    "Dataset": dataset_name,
                    "Algorithm": algo_name,
                    "Precision": round(precision, 2),
                    "Recall": round(recall, 2),
                    "F1-Score": round(f1, 2),
                }
            )

    df_results = pd.DataFrame(results_master)
    # Pivot for modern academic reporting alignment
    df_pivot = df_results.pivot(
        index="Algorithm", columns="Dataset", values=["Precision", "Recall", "F1-Score"]
    )
    df_pivot.to_csv("results.csv", index=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
