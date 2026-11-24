import numpy as np
import pandas

from so4gp.algorithms import GRAANK

dummy_data_only = [
    [30, 3, 1, 10],
    [35, 2, 2, 8],
    [40, 4, 3, 7],
    [50, 1, 4, 6],
    [52, 7, 5, 2],
]

dummy_data = [
    ["2021-03", 30, 3, 1, 10],
    ["2021-04", 35, 2, 2, 8],
    ["2021-05", 40, 4, 2, 7],
    ["2021-06", 50, 1, 1, 6],
    ["2021-07", 52, 7, 1, 2],
]

dummy_df1 = pandas.DataFrame(
    dummy_data_only, columns=["Age", "Salary", "Cars", "Expenses"]
)

dummy_df = pandas.DataFrame(
    dummy_data, columns=["Date", "Age", "Salary", "Cars", "Expenses"]
)


def test_graank():
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"🚀 Running execution pipeline on device: {device.upper()}")

    ## Test Algorithms
    mine_obj = GRAANK(dummy_df, min_sup=0.4, eq=False, device="cpu")
    # mine_obj = GRAANK(
    # "../benchmarks/optimization/data/breast_cancer.csv",
    # min_sup=0.2
    # )

    result_json = mine_obj.discover(
        search_type="ga",  # aco, pso, hc, random, clustergp
        target_col=1,
        # exclude_target=False,
        # n_pop=20,
        max_iteration=10,
    )  # GA

    print("\n", result_json)
    assert result_json is not None


def test_gp_features():
    feat_df = GRAANK.build_gradual_features(
        np.array(dummy_data_only), pattern_level=2
    )
    print("\n", feat_df)
    assert feat_df is not None

