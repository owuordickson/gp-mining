import pandas
import torch

from so4gp.algorithms import TGRAANK

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
    result_json = mine_obj1.discover(
        target_col=1,
        transformations="all",
        search_algorithm="apriori",
        max_iteration=1,
        eval_mode=True,
        compute_causality=False,
    )  # TGRAANK-AMI
    print(f"{result_json}\n")
    # start = time.time()
    # corr_df = mine_obj1.get_lagged_dependencies(max_lag=3)
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

    """"
    import numpy as np

    # Simulate 1,000 samples, each with 500 rows of data across 10 variables
    # Shape: (1000, 500, 10)
    all_samples = np.random.randn(1000, 500, 10)
    predictions = []

    # --- NESTED LOOP ENGINE ---
    # Loop 1: Iterates through every single sample sequentially
    for sample in all_samples:
        # Loop 2 & 3: Hidden inside np.cov and np.linalg.inv calculating element-by-element
        cov_matrix = np.cov(sample, rowvar=False)
        inv_matrix = np.linalg.inv(cov_matrix)

        predictions.append(inv_matrix)

    # Convert list back to an array
    final_output = np.array(predictions)
    """

    """
    import torch

    # 1. Send your entire dataset block straight to the GPU memory
    # Shape: (1000, 500, 10) -> (Batch, Time_Steps, Variables)
    gpu_samples = torch.tensor(all_samples, device="cuda")

    # --- VECTORIZED ENGINE ---
    # Step A: Transpose data to calculate covariance across all batches at once
    # We compute the mean across the time dimension (dim=1)
    centered_data = gpu_samples - gpu_samples.mean(dim=1, keepdim=True)

    # Step B: Vectorized batch matrix multiplication (BMM) replaces the loop!
    # This calculates 1,000 covariance matrices at the exact same millisecond
    batch_cov = torch.bmm(centered_data.transpose(1, 2), centered_data) / (500 - 1)

    # Step C: Vectorized linear inversion 
    # PyTorch natively computes inversions for a whole batch of matrices at once
    gpu_predictions = torch.linalg.inv(batch_cov)

    # 2. Move final outputs back to CPU for scoring/saving
    final_output = gpu_predictions.cpu().numpy()
    """
