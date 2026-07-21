import numpy as np
import pandas

from so4gp import DataGP
from so4gp.algorithms import GRAANK, TGRAANK, ClusterGP
from src import so4gp as sgp

if __name__ == "__main__":

    # dummy_data = [["2021-03", 30, 3, 1, 10], ["2021-03", 35, 2, 2, 8], ["2021-03", 40, 4, 3, 7], ["2021-03", 50, 1, 4, 6], ["2021-03", 52, 7, 5, 2]]
    dummy_data = [["2021-03", 30, 3, 1, 10], ["2021-04", 35, 2, 2, 8], ["2021-05", 40, 4, 2, 7], ["2021-06", 50, 1, 1, 6], ["2021-07", 52, 7, 1, 2]]
    dummy_df = pandas.DataFrame(dummy_data, columns=['Date', 'Age', 'Salary', 'Cars', 'Expenses'])
    # dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    # dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])

    ## Test Algorithms
    mine_obj = GRAANK(dummy_df, min_sup=0.5, eq=False)
    # mine_obj = ClusterGP(dummy_df, 0.5, max_iter=3, e_prob=0.0)
    # mine_obj1 = TGRAANK(dummy_df, target_col=1, min_sup=0.2, min_rep=0.1)
    # result_json = mine_obj.discover(target_col=2, compute_descriptors=True)  # GRAANK
    result_json = mine_obj.discover()                                          # GRAANK/ClusterGP
    # result_json = mine_obj.discover(search_type='aco', target_col=1, exclude_target=False)    # ACO
    # result_json = mine_obj.discover(search_type='ga', target_col=1, exclude_target=False)     # GA
    # result_json = mine_obj.discover(search_type='pso', target_col=1, exclude_target=False)    # PSO
    # result_json = mine_obj.discover(search_type='hl', target_col=1, exclude_target=False)     # HL
    # result_json = mine_obj.discover(search_type='random', target_col=1, exclude_target=False) # Random

    # result_json = mine_obj1.discover()                          # TGrad
    # result_json = mine_obj1.discover(transformation_algorithm='ami', use_clustering=False, eval_mode=True) # TGradAMI
    print(f"{result_json}\n")

    ## Test Time
    #print(sgp.DataGP.test_time("09-01-2005"))

    # Generate dataset
    # sgp.save_pairwise_data(dummy_df)

    # MCP Tools
    # from so4gp.tools import mine_gps, mine_tgps
    # lst_data = [["Date", "Age"], ["2021-03", 30], ["2021-04", 35]]
    # res = mine_tgps(lst_data, min_support=0.1, target_column=0, min_rep=0.1)
    # print(res)

    """
    ## Test Warping Path
    tgt_col = 0
    graank = GRAANK(dummy_df)
    ## graank.discover(target_col=tgt_col)
    ## graank.discover()
    graank.fit_warpingset()
    plot_data = []
    for k, val in graank.warping_set.items():
        val_arr = np.array(list(val), dtype=int)
        plot_data.append(f"{k}: {[val_arr[:, 0], val_arr[:, 1]]}")
    print(f"\n{plot_data}")

    
    import math
    import matplotlib.pyplot as plt
    # Calculate the number of rows needed
    num_plots = len((graank.warping_set or {}).items())
    cols = 4
    rows = math.ceil(num_plots / cols)

    # Create subplots with the required number of rows and columns
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5))
    axes = axes.flatten()  # Flatten to make indexing easier

    # Plot each component in its subplot
    for idx, (key, val) in enumerate(graank.warping_set.items()):
        val = np.array(list(val), dtype=int)
        axes[idx].plot(val[:,0], val[:,1], '-')#, label=f"{key}")
        axes[idx].set_xlabel("Index i")
        axes[idx].set_ylabel("Index j")
        #axes[idx].legend()
        axes[idx].set_title(f"'{key}' Warping Path")

    # Hide any extra subplots
    for ax in axes[num_plots:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


    ## Analyze GPs
    estimated_gps = list()
    temp_gp = sgp.GP()
    for gi_str in ['1+', '4-']:
        temp_gp.add_gradual_item(sgp.GI.from_string(gi_str))
    temp_gp.support = 0.5
    estimated_gps.append(temp_gp)
    temp_gp = sgp.GP()
    for gi_str in ['1+', '3-', '0+']:
        temp_gp.add_gradual_item(sgp.GI.from_string(gi_str))
    temp_gp.support = 0.48
    estimated_gps.append(temp_gp)
    res = sgp.analyze_gps(dummy_df, min_sup=0.4, est_gps=estimated_gps, approach='bfs')
    print(res)
    """

