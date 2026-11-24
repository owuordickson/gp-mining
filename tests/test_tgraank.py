
from test_graank import dummy_df

from so4gp.algorithms import TGRAANK


def test_tgraank():

    ## Test Algorithms
    mine_obj = TGRAANK(dummy_df, min_sup=0.05, min_rep=0.1)

    result_json = mine_obj.discover(
        target_col=1,
        transformations="all",
        search_algorithm="apriori",
        max_iteration=10
    )  # TGRAANK

    # result_json=mine_obj.discover(
    #        target_col=1,
    #        transformations="all",
    #        search_algorithm="apriori",
    #        max_iteration=1,
    #        eval_mode=True,
    #        compute_causality=False,
    # )  # TGRAANK-AMI

    print("\n", result_json)
    assert result_json is not None


def test_causal_mat():
    mine_obj = TGRAANK(dummy_df, min_sup=0.05, min_rep=0.1)
    corr_df = mine_obj.discover_causal_relationships(max_lag=3)

    print("\n", corr_df)  # Print the DataFrame so it can be displayed
    assert corr_df is not None
