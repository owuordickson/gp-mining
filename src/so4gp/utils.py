# SPDX-License-Identifier: MIT
# See the LICENSE file at the root of this
# repository for complete details.

"""
@author: Dickson Owuor
@credits: Thomas Runkler, Edmond Menya, and Anne Laurent
@license: MIT
@email: owuordickson@gmail.com
@created: 21 July 2021

A collection of miscellaneous classes and methods.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def write_file(data, path, wr=True) -> None:
    """
    Writes data into a file

    :param data: information to be written
    :param path: name of the file and storage path
    :param wr: writes data into the file if True

    :return:
    """
    if wr:
        with open(path, "w") as f:
            f.write(data)
            f.close()
    else:
        pass


def gp_spider_plot(
    df_list: list[pd.DataFrame],
    labels: list[str],
    parameters: list[str],
    grid_levels: int = 6,
) -> Figure:
    """Compare mean GP descriptors and one sample standard deviation per group.

    Each descriptor has its own scale shared by all groups. Missing values are
    skipped, and a single observation is drawn without an error bar.

    :param df_list: Tables with numeric descriptor columns, such as those
        returned by ``GRAANK.build_gradual_features`` for 2D or 3D inputs.
    :param labels: One legend label per table.
    :param parameters: Descriptor column names in the desired spoke order.
    :param grid_levels: Number of grid rings.
    :return: The Matplotlib figure.
    """
    if not df_list or len(df_list) != len(labels):
        raise ValueError("Provide at least one table and one label per table")
    if not parameters:
        raise ValueError("Provide at least one descriptor column")
    if grid_levels < 1:
        raise ValueError("grid_levels must be positive")

    grid_radii = np.linspace(0, 1, grid_levels + 1)[1:]
    mean_rows, std_rows = [], []
    for frame in df_list:
        values = frame.loc[:, parameters].replace([np.inf, -np.inf], np.nan)
        mean_rows.append(values.mean())
        std_rows.append(values.std().fillna(0.0))

    means = pd.DataFrame(mean_rows)
    deviations = pd.DataFrame(std_rows)
    lower = (means - deviations).min().to_numpy(dtype=float, na_value=np.nan)
    upper = (means + deviations).max().to_numpy(dtype=float, na_value=np.nan)
    span = upper - lower
    variable = span > 0

    # A constant descriptor is placed at the midpoint of its spoke.
    mean_values = means.to_numpy(dtype=float, na_value=np.nan)
    radii = np.divide(
        mean_values - lower, span, out=np.full(means.shape, 0.5), where=variable
    )
    radii[np.isnan(mean_values)] = np.nan
    error_radii = np.divide(
        deviations.to_numpy(dtype=float),
        span,
        out=np.zeros(means.shape),
        where=variable,
    )

    angles = np.linspace(np.pi / 2, -3 * np.pi / 2, len(parameters), endpoint=False)
    spokes = np.column_stack((np.cos(angles), np.sin(angles)))
    grid = np.vstack((spokes, spokes[0]))
    figure, axes = plt.subplots(figsize=(9, 8), layout="constrained")

    for radius in grid_radii:
        axes.plot(*(radius * grid).T, color="0.85", linewidth=0.8, zorder=0)

    for index, (parameter, spoke) in enumerate(zip(parameters, spokes, strict=True)):
        axes.plot([0, spoke[0]], [0, spoke[1]], color="0.8", linewidth=0.8, zorder=0)
        axes.text(*(1.13 * spoke), parameter, ha="center", va="center", weight="bold")

        if variable[index]:
            ticks = zip(grid_radii, lower[index] + grid_radii * span[index])
        elif np.isfinite(lower[index]):
            ticks = [(0.5, lower[index])]
        else:
            ticks = []
        for radius, value in ticks:
            axes.annotate(
                f"{value:.3g}",
                radius * spoke,
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
                color="0.35",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
            )

    for label, profile, errors in zip(labels, radii, error_radii, strict=True):
        points = profile[:, None] * spokes
        closed_points = np.vstack((points, points[0]))
        (line,) = axes.plot(*closed_points.T, marker="o", linewidth=2, label=label)
        color = line.get_color()
        if len(parameters) >= 3 and np.isfinite(points).all():
            axes.fill(*points.T, color=color, alpha=0.1)

        for radius, error, spoke in zip(profile, errors, spokes, strict=True):
            if error > 0:
                endpoints = (radius + np.array([-error, error]))[:, None] * spoke
                axes.plot(*endpoints.T, color=color, linewidth=1.4, alpha=0.7)

    axes.set_aspect("equal")
    axes.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
    axes.set_axis_off()
    axes.set_title("GP descriptor means ± 1 SD", pad=18)
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncols=2, frameon=False)
    return figure


def classify_ftgps(lst_test_data, lst_ground_truth, threshold=0.5) -> dict:
    """
    Classify FTGPs by comparing a test dataset against a reference dataset.

    Each FTGP is classified according to whether its support is above or
    below the specified threshold in the test and reference datasets.

    A pattern that is absent from one dataset is assigned support = 0
    in that dataset.

    Pattern matching is performed using ``is_similar_to()`` and each
    pattern is matched at most once.

    Parameters
    ----------
    lst_test_data : list
        FTGPs extracted from the test dataset.

    lst_ground_truth : list
        FTGPs extracted from the reference (proxy ground-truth) dataset.

    threshold : float, optional
        Support threshold used to distinguish supported and unsupported
        FTGPs. Defaults to 0.5.

    Returns
    -------
    dict
        Counts of TP, FP, FN, and TN.
    """

    confusion_matrix_counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    # Track which patterns have already been matched.
    matched_test = set()
    matched_ground_truth = set()

    # ------------------------------------------------------------
    # Step 1: Match test FTGPs to reference FTGPs.
    # ------------------------------------------------------------
    for i, test_pat in enumerate(lst_test_data):
        for j, gt_pat in enumerate(lst_ground_truth):
            if i in matched_test or j in matched_ground_truth:
                continue

            if test_pat.is_similar_to(gt_pat):
                test_support = test_pat.support
                gt_support = gt_pat.support

                if test_support >= threshold and gt_support >= threshold:
                    confusion_matrix_counts["TP"] += 1

                elif test_support >= threshold > gt_support:
                    confusion_matrix_counts["FP"] += 1

                elif test_support < threshold <= gt_support:
                    confusion_matrix_counts["FN"] += 1

                else:
                    confusion_matrix_counts["TN"] += 1

                matched_test.add(i)
                matched_ground_truth.add(j)

                break

    # ------------------------------------------------------------
    # Step 2: Handle test FTGPs with no corresponding reference FTGP.
    # Their reference support is therefore zero.
    # ------------------------------------------------------------
    for i, test_pat in enumerate(lst_test_data):
        if i in matched_test:
            continue

        if test_pat.support >= threshold:
            confusion_matrix_counts["FP"] += 1
        else:
            confusion_matrix_counts["TN"] += 1

    # ------------------------------------------------------------
    # Step 3: Handle reference FTGPs with no corresponding test FTGP.
    # Their test support is therefore zero.
    # ------------------------------------------------------------
    for j, gt_pat in enumerate(lst_ground_truth):
        if j in matched_ground_truth:
            continue

        if gt_pat.support >= threshold:
            confusion_matrix_counts["FN"] += 1
        else:
            confusion_matrix_counts["TN"] += 1

    return confusion_matrix_counts
