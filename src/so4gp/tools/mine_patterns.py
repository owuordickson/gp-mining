# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Dickson Owuor
# See the LICENSE file at the root of this repository for complete details.

"""Gradual Pattern Mining Tools for MCP.

This module provides a collection of wrapper methods that expose 'so4gp' algorithms as
Model Context Protocol (MCP)
tools for mining sequential and temporal gradual patterns.

Authors:
    Dickson Owuor (owuordickson@gmail.com)

Credits:
    Thomas Runkler and Anne Laurent
"""

import pandas as pd

# from mcp.server.fastmcp import FastMCP

# 1. Initialize the FastMCP server instance
# mcp = FastMCP("Gradual Pattern Miner")


# @mcp.tool()
def mine_gps(
    data: list[list[str | float | int]],
    min_support: float = 0.5,
    target_column: int | None = None,
    algorithm: str = "graank",
    max_iteration: int | None = None,
) -> str:
    """
    Extract standard co-occurring gradual patterns from a tabular dataset.

    Analyzes numerical and categorical datasets to find parallel trends (e.g.,
    'The more X increases, the more Y decreases') based on Gradual Concordance Set
    (GCS) descriptors.

    Args:
        data: Matrix containing row data. The first row must contain column headers.
            Subsequent rows contain numerical values. A date-time column may be
            included.
            Example 1: [["Age", "Salary"], [30, 3.5], [35, 2.6]].
            Example 2: [["Date", "Age", "Salary"], ["2021-03", 30, 3.5], ["2021-04", 35,
            2.6]].
        min_support: Minimum frequency threshold for a pattern (range: 0.0 to 1.0).
        target_column: Zero-based index of the target variable column.
        algorithm: A string name of the mining algorithm to execute.
            Allowed values: 'graank', 'graank-ga', 'graank-aco', 'cluster-gp'.
        max_iteration: Maximum optimization loops for evolutionary or heuristic
            algorithms ('graank-ga', 'graank-aco', 'graank-hc').

    Returns:
        A JSON-serialized dictionary containing execution telemetry and discovered
        patterns matching the criteria.

        Format Schema:
        {
            "Algorithm": str,
            "Run-time": str,
            "Patterns": list[list[str]],
            "Invalid Count": str
        }

        Schema Fields:
        - Algorithm: Name of the executed algorithm.
        - Run-time: Total computational time taken for processing.
        - Invalid Count: Total count of candidate patterns discarded because their
          calculated support fell below the min_support threshold.
        - Patterns: A list of discovered patterns. Each pattern is a list of strings
          starting with the trend description followed by 6 structural metrics:
            1. sup: Full bitmap comparison count divided by C(n, 2).
            2. density: Stored forward pair count divided by C(n, 2).
            3. avg_dev: Mean separation j - i across stored pairs (i, j), i < j.
            4. dispersion: Population variance of those j - i separations.
            5. connect: Undirected connected component count, including isolated
               observations.
            6. singularity_scr: Maximum vertex degree divided by stored pair count.

        Pairs use zero-based observation indices. An empty GCS has density 0 and
        n connected components; avg_dev, dispersion, and singularity_scr are NaN
        (rendered as "nan" in the metric strings).

        Example Output Payload:
        The metrics below describe four observations with all six forward pairs.
        {
            "Algorithm": "GRAANK",
            "Run-time": "1.329559 seconds",
            "Patterns": [
                [
                    "Expenses-, Age+",
                    "sup=1.0",
                    "density=1.0",
                    "avg_dev=1.667",
                    "dispersion=0.556",
                    "connect=1",
                    "singularity_scr=0.5"
                ]
            ],
            "Invalid Count": "9"
        }
    """

    # Separate column names from data
    column_names = data[0]
    data = data[1:]
    data_df = pd.DataFrame(data, columns=column_names)

    # Run the mining algorithm
    if algorithm == "cluster-gp":
        from ..algorithms.cluster_gp import ClusterGP

        if max_iteration is not None:
            mine_obj = ClusterGP(data_df, min_sup=min_support, max_iter=max_iteration)
        else:
            mine_obj = ClusterGP(data_df, min_sup=min_support)
        return str(mine_obj.discover(save_results=False))
    elif algorithm == "graank-aco":
        from ..algorithms.graank import GRAANK

        mine_obj = GRAANK(data_df, min_sup=min_support)
        return mine_obj.discover(
            search_type="aco",
            target_col=target_column,
            max_iteration=max_iteration,
            save_results=False,
        )
    elif algorithm == "graank-ga":
        from ..algorithms.graank import GRAANK

        mine_obj = GRAANK(data_df, min_sup=min_support)
        return mine_obj.discover(
            search_type="ga",
            target_col=target_column,
            max_iteration=max_iteration,
            save_results=False,
        )
    elif algorithm == "graank":
        from ..algorithms.graank import GRAANK

        mine_obj = GRAANK(data_df, min_sup=min_support)
        return mine_obj.discover(target_col=target_column, save_results=False)
    else:
        raise ValueError("Invalid algorithm!")


# @mcp.tool()
def mine_tgps(
    data: list[list[str | float | int]],
    target_column: int,
    min_support: float = 0.5,
    min_rep: float = 0.5,
) -> str:
    """
    Mine temporal gradual patterns from time-series data using time lags.

    Extracts gradual trends that occur within specific chronological delay
    windows, factoring in dataset transformation thresholds.

    Args:
        data: Matrix containing row data. The first row must contain column headers.
            Subsequent rows contain time-series values. A valid date-time column
            is required.
            Example: [["Date", "Age"], ["2021-03", 30], ["2021-04", 35]]
        target_column: Zero-based index of the target variable column.
        min_support: Minimum frequency threshold for a pattern (range: 0.0 to 1.0).
        min_rep: Minimum representativity threshold for transforming the dataset
            (range: 0.0 to 1.0).

    Returns:
        A JSON-serialized dictionary listing validated temporal patterns, time-lag
        telemetry, and the raw alignment transformations.

        Format Schema:
        {
            "Algorithm": str,
            "Minimum Representation": str,
            "MI Minimum Error": str,
            "MI Error": str,
            "Target Column": str,
            "Run-time": str,
            "Patterns": list[list[str]],
            "Transformation Steps": dict[int, int],
            "Time Data": list[list[str]],
            "Transformed Data": list[list[str | float | int]]
        }

        Schema Fields:
        - Algorithm: Name of the executed temporal mining algorithm (e.g., 'TGradAMI').
        - Minimum Representation: The active representativity filter configuration
        string.
        - MI Minimum Error / MI Error: Mutual Information error evaluation statistics.
        - Target Column: String representation of the chosen target variable index.
        - Run-time: Total computational time taken for processing.
        - Transformation Steps: Mapping indices showing structural dataset shifts.
        - Time Data: Extracted step delays represented in elapsed epoch/seconds.
        - Transformed Data: The resulting matrix after time-lag alignment is applied.
        - Patterns: Discovered cross-temporal trends followed by 6 structural metrics:
            1. sup: Full transformed bitmap comparison count divided by C(n, 2).
            2. density: Stored forward pair count divided by C(n, 2), where n is the
               number of transformed observations.
            3. avg_dev: Mean separation j - i across stored pairs (i, j), i < j.
            4. dispersion: Population variance of those j - i separations.
            5. connect: Undirected connected component count, including isolated
               transformed observations.
            6. singularity_scr: Maximum vertex degree divided by stored pair count.

        Pairs use zero-based transformed observation indices. An empty GCS has
        density 0 and n connected components; avg_dev, dispersion, and
        singularity_scr are NaN (rendered as "nan" in the metric strings).

        Example Output Payload:
        The metrics below describe four transformed observations with all six
        forward pairs.
        {
            "Algorithm": "TGradAMI",
            "Minimum Representation": "0.60",
            "MI Minimum Error": "0.10",
            "MI Error": "-1.00",
            "Target Column": "1",
            "Run-time": "2.904684 seconds",
            "Patterns": [
                [
                    "Age+, (Expenses-) +2.0 months",
                    "sup=1.0",
                    "density=1.0",
                    "avg_dev=1.667",
                    "dispersion=0.556",
                    "connect=1",
                    "singularity_scr=0.5"
                ]
            ],
            "Transformation Steps": {"2": 2, "3": 2, "4": 2},
            "Time Data": [
                ["Salary", "Cars", "Expenses"],
                ["5270400.0", "5270400.0", "5270400.0"],
                ["5270400.0", "5270400.0", "5270400.0"],
                ["5270400.0", "5270400.0", "5270400.0"],
                ["5270400.0", "5270400.0", "5270400.0"]
            ],
            "Transformed Data": [
                ["Date", "Age", "Salary", "Cars", "Expenses"],
                ["2021-03", 30, 4, 2, 7],
                ["2021-04", 35, 1, 1, 6],
                ["2021-05", 40, 3, 2, 5],
                ["2021-06", 45, 2, 1, 4]
            ]
        }
    """

    # Separate column names from data
    column_names = data[0]
    data = data[1:]
    data_df = pd.DataFrame(data, columns=column_names)

    # Run the mining algorithm
    from ..algorithms.tgraank import TGRAANK

    mine_obj = TGRAANK(data_df, min_sup=min_support, min_rep=min_rep)
    return mine_obj.discover(
        target_col=target_column, transformations="ami", save_results=False
    )
