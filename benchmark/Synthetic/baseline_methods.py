# Import libraries

import numpy as np
from causallearn.search.ConstraintBased.PC import pc

# Statistical & Baselines
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests

# Tigramite & Graph Learning
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

# Custom frameworks (so4gp package)
from so4gp.algorithms import TGRAANK


def run_classical_statistics(df, threshold=0.3):
    """Pearson correlation baseline"""
    adj_matrix = np.zeros((2, 2))
    corr, _ = pearsonr(df.iloc[:, 0], df.iloc[:, 1])
    if abs(corr) > threshold:
        # Classical correlation is symmetrical (non-directional)
        adj_matrix[0, 1] = 1
        adj_matrix[1, 0] = 1
    return adj_matrix


def run_granger_causality(df, max_lag=3, alpha=0.05):
    """Vector Autoregressive Granger Causality"""
    adj_matrix = np.zeros((2, 2))
    # Test if Column 0 Granger-causes Column 1
    try:
        res = grangercausalitytests(df[[df.columns[1], df.columns[0]]], maxlag=max_lag, verbose=False)
        p_values = [res[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]
        if min(p_values) < alpha:
            adj_matrix[0, 1] = 1
    except:
        pass
    return adj_matrix


def run_pcmci_tigramite(df, max_lag=3, alpha=0.05):
    """Tigramite PCMCI Framework"""
    adj_matrix = np.zeros((2, 2))
    dataframe = pp.DataFrame(df.values, var_names=list(df.columns))
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=False)
    results = pcmci.run_pcmci(tau_max=max_lag, pc_alpha=None)
    #print(f"\nTigramite PCMCI Results: {results}\n")

    # Extract structural matrix edges
    p_matrix = results['p_matrix']
    #  print(f"\nTigramite PCMCI Results. P-Matrix\n{p_matrix}\n\nRes: {p_matrix[0, 1, 1:]}\n")
    # If any lag reveals a significant p-value from 0 -> 1
    if np.min(p_matrix[0, 1, 1:]) < alpha:
        adj_matrix[0, 1] = 1
    return adj_matrix


def run_pc_algorithm(df):
    """Constraint-based PC Graph Algorithm"""
    adj_matrix = np.zeros((2, 2))
    cg = pc(df.values)
    # Parse causal-learn graph output format
    graph_out = cg.G.graph
    if graph_out[0, 1] != 0:
        adj_matrix[0, 1] = 1
    return adj_matrix


def run_t_graank(df):
    """
        T-GRAANK correctly discovers directional gradual dependencies with time shifts
    """
    mine_obj = TGRAANK(df, min_sup=0.005)
    corr_df = mine_obj.get_lagged_dependencies(max_lag=3)
    corr_arr = corr_df.values
    # print(f"\n\nTGRAANK Corr:\n{corr_df}\n\n")

    adj_matrix = np.zeros((2, 2))
    if corr_arr[0, 1] != 0:
        adj_matrix[0, 1] = 1
    return adj_matrix
