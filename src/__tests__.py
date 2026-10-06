import pandas
import torch

from so4gp.algorithms import TGRAANK

# make get_selected_rows -- less intensive and fast

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Running execution pipeline on device: {device.upper()}")

    # dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 3, 7], [50, 1, 4, 6], [52, 7, 5, 2]]
    dummy_data = [
        ["2021-03", 30, 3, 1, 10],
        ["2021-04", 35, 2, 2, 8],
        ["2021-05", 40, 4, 2, 7],
        ["2021-06", 50, 1, 1, 6],
        ["2021-07", 52, 7, 1, 2],
    ]
    # dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
    dummy_df = pandas.DataFrame(
        dummy_data, columns=["Date", "Age", "Salary", "Cars", "Expenses"]
    )
    # dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    # dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])

    ## Test Algorithms
    # mine_obj = GRAANK(dummy_df, min_sup=0.4, eq=False, device=device)
    # mine_obj = ClusterGP(dummy_df, 0.5, max_iter=3, e_prob=0.0, device=device)
    mine_obj1 = TGRAANK(dummy_df, min_sup=0.05, min_rep=0.1, device=device)
    # result_json = mine_obj.discover(target_col=1, compute_descriptors=True)  # GRAANK
    # result_json = mine_obj.discover()                                          # GRAANK/ClusterGP
    # result_json = mine_obj.discover(search_type='aco', target_col=1, exclude_target=False, max_iteration=10)    # ACO
    # result_json = mine_obj.discover(search_type='ga', target_col=1, exclude_target=False, n_pop=10, max_iteration=10)     # GA
    # result_json = mine_obj.discover(search_type='pso', target_col=1, exclude_target=False, max_iteration=10)    # PSO
    # result_json = mine_obj.discover(search_type='hc', target_col=1, exclude_target=False, max_iteration=10)     # HL
    # result_json = mine_obj.discover(search_type='random', target_col=1, exclude_target=False, max_iteration=10) # Random
    # result_json = mine_obj.discover(search_type='clustergp', target_col=1, exclude_target=False, max_iteration=10, e_prob=0.0) # ClusterGP

    # result_json = mine_obj1.discover(target_col=1, transformations='all', search_algorithm='ga', max_iteration=10)                                      # TGRAANK
    result_json = mine_obj1.discover(target_col=1, transformations="all", search_algorithm="apriori", max_iteration=1, eval_mode=True, compute_causality=False,)  # TGRAANK-AMI
    print(f"{result_json}\n")
    # start = time.time()
    # corr_df = mine_obj1.discover_causal_relationships(max_lag=3)
    # print(corr_df)
    # end = time.time() - start
    # print(f"Time: {end}")

    ## Test Time
    # print(sgp.DataGP.test_time("09-01-2005"))

    # Generate dataset
    # sgp.save_pairwise_data(dummy_df)

    # MCP Tools
    # from so4gp.tools import mine_gps, mine_tgps
    # lst_data = [["Date", "Age"], ["2021-03", 30], ["2021-04", 35]]
    # res = mine_tgps(lst_data, min_support=0.1, target_column=0, min_rep=0.1)
    # print(res)
