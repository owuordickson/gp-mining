# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# See the LICENSE file at the root of this
# repository for complete details.

import math
import time
import numpy as np
from sklearn.cluster import KMeans

from .base.graank_base import BaseGrad
from ..gradual_patterns import GI, GP, TGP, TimeDelay


class ClusterGP(BaseGrad):

    def __init__(self, *args, e_prob: float = 0.5, max_iter: int = 10, **kwargs):
        """
        CluDataGP stands for Clustering DataGP. It is a class that inherits the DataGP class to create data-gp
        objects for the clustering approach. This class inherits the DataGP class which is used to create data-gp objects.
        The classical data-gp object is meant to store all the parameters required by GP algorithms to extract gradual
        patterns (GP). It takes a numeric file (in CSV format) as input and converts it into an object whose attributes are
        used by algorithms to extract GPs.

        A class for creating data-gp objects for the clustering approach. This class inherits the DataGP class which is
        used to create data-gp objects. This class adds the parameters required for clustering gradual items to the
        data-gp object.

        :param args: [required] data source path of Pandas DataFrame, [optional] minimum-support, [optional] eq
        :param e_prob: [optional] erasure probability, the default is 0.5
        :param max_iter: [optional] maximum iteration for score vector estimation, the default is 10

        >>> import pandas
        >>> from so4gp.algorithms import ClusterGP
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = ClusterGP(data_source=dummy_df, min_sup=0.5, max_iter=3, e_prob=0.5)
        >>> result_json = mine_obj.discover()
        >>> print(result_json) # doctest: +SKIP
        """
        super(ClusterGP, self).__init__(*args, **kwargs)
        self._erasure_probability: float = e_prob
        self._max_iteration: int = max_iter
        self._gradual_items: list[GI] = []
        self._win_mat: np.ndarray|None = None
        self._cum_wins: np.ndarray|None = None
        self._net_win_mat: np.ndarray|None = None
        self._ij: np.ndarray|None = None
        self._construct_matrices(e_prob)

    def _construct_matrices(self, e: float=0):
        """
        Generates all the gradual items and constructs: (1) net-win matrix, (2) cumulative wins, (3) pairwise objects.

        :param e: [required] erasure probability
        :return: List of gradual items, net-win matrix, cumulative win matrix, selected pairwise (ij) objects
        """

        n = self.row_count
        prob = 1 - e  # Sample probability

        if prob == 1:
            # 1a. Generate all possible pairs
            pair_ij = np.array(np.meshgrid(np.arange(n), np.arange(n))).T.reshape(-1, 2)

            # 1b. Remove duplicates or reversed pairs
            pair_ij = pair_ij[np.argwhere(pair_ij[:, 0] < pair_ij[:, 1])[:, 0]]
        else:
            # 1a. Generate random pairs using erasure-probability
            total_pair_count = int(n * (n - 1) * 0.5)
            rand_1d = np.random.choice(n, int(prob * total_pair_count) * 2, replace=True)
            pair_ij = np.reshape(rand_1d, (-1, 2))

            # 1b. Remove duplicates
            pair_ij = pair_ij[np.argwhere(pair_ij[:, 0] != pair_ij[:, 1])[:, 0]]

        # 2. Variable declarations
        attr_data = self.data.T  # Feature data objects
        s_mat = []  # S-Matrix (made up of S-Vectors)
        w_mat = []  # win matrix
        cum_wins = []  # Cumulative wins

        # 3. Construct S matrix from the data set
        for col in self.attr_cols:
            # Feature data objects
            col_data = np.array(attr_data[col], dtype=float)  # Feature data objects

            # Cumulative Wins: for estimation of score-vector
            temp_cum_wins = np.where(col_data[pair_ij[:, 0]] < col_data[pair_ij[:, 1]], 1,
                                     np.where(col_data[pair_ij[:, 0]] > col_data[pair_ij[:, 1]], -1, 0))

            # S-vector
            s_vec = np.zeros((n,), dtype=np.int32)
            for w in [1, -1]:
                positions = np.flatnonzero(temp_cum_wins == w)
                i, counts_i = np.unique(pair_ij[positions, 0], return_counts=True)
                j, counts_j = np.unique(pair_ij[positions, 1], return_counts=True)
                s_vec[i] += w * counts_i  # 'i' wins/loses (1/-1)
                s_vec[j] += -w * counts_j  # 'j' loses/wins (1/-1)

            # Normalize S-vector
            if np.count_nonzero(s_vec) > 0:
                w_mat.append(np.copy(s_vec))
                # nodes_mat.append(nodes_vec)

                s_vec[s_vec > 0] = 1  # Normalize net wins
                s_vec[s_vec < 0] = -1  # Normalize net loses

                self._gradual_items .append(GI(col, '+'))
                cum_wins.append(temp_cum_wins)
                s_mat.append(s_vec)

                self._gradual_items .append(GI(col, '-'))
                cum_wins.append(-temp_cum_wins)
                s_mat.append(-s_vec)

        self._win_mat = np.array(w_mat)
        self._cum_wins = np.array(cum_wins)
        self._net_win_mat = np.array(s_mat)
        self._ij = pair_ij

    def _infer_gps(self, clusters: np.ndarray, exclude_target:bool=False, time_data: dict|None=None) -> bool:
        """
        A function that infers GPs from clusters of gradual items.

        :param clusters: [required] groups of gradual items clustered through K-MEANS algorithm
        :param exclude_target: Only accept GP candidates that do not contain the target feature.
        :param time_data: (optional) time data for estimating time lag.

        :return: True if GPs are inferred successfully, False otherwise
        """

        def estimate_score_vector(c_win_vec: np.ndarray) -> np.ndarray:
            """Description

            A function that estimates the score vector based on the cumulative wins.

            :param c_win_vec: [required] cumulative wins
            :return: Score vector
            """

            # Estimate score vector from pairs
            n = self.row_count
            score_vector = np.ones(shape=(n,))
            arr_ij = self._ij

            # Construct a win-matrix
            temp_vec = np.zeros(shape=(n,))
            pair_count = arr_ij.shape[0] if arr_ij is not None else 0

            # Compute score vector
            for _ in range(self._max_iteration):
                if np.count_nonzero(score_vector == 0) > 1:
                    break
                else:
                    for pr in range(pair_count):
                        if arr_ij is not None:
                            pr_val = c_win_vec[pr]
                            i = arr_ij[pr][0]
                            j = arr_ij[pr][1]
                            if pr_val == 1:
                                log = math.log(
                                    math.exp(score_vector[i]) / (math.exp(score_vector[i]) + math.exp(score_vector[j])),
                                    10)
                                temp_vec[i] += pr_val * log
                            elif pr_val == -1:
                                log = math.log(
                                    math.exp(score_vector[j]) / (math.exp(score_vector[i]) + math.exp(score_vector[j])),
                                    10)
                                temp_vec[j] += -pr_val * log
                    score_vector = abs(temp_vec / np.sum(temp_vec))
            return score_vector

        def estimate_support(score_vecs: list) -> float:
            """Description

            A function that estimates the frequency support of a GP based on its score vector.

            :param score_vecs: Score vector
            :return: Estimated support (float)
            """

            # Estimate support - use different score-vectors to construct pairs
            n = self.row_count
            bin_mat = np.ones((n, n), dtype=bool)
            for vec in score_vecs:
                temp_bin = vec < vec[:, np.newaxis]
                bin_mat = np.multiply(bin_mat, temp_bin)

            support = float(np.sum(bin_mat)) / GP.pair_count(n)
            return support

        time_lag = None
        target_col = self._target_col
        all_gis = self._gradual_items
        cum_wins = self._cum_wins

        lst_indices = [np.where(clusters == element)[0] for element in np.unique(clusters)]
        for grp_idx in lst_indices:
            if grp_idx.size > 1:
                # 1. Retrieve all cluster-pairs and the corresponding GIs
                cluster_gis = [all_gis[idx] for idx in grp_idx]
                cluster_cum_wins = cum_wins[grp_idx] if cum_wins is not None else np.array([]) # All the rows of selected groups

                # 2. Compute score vector from R matrix
                score_vectors = []  # Approach 2
                for c_win in cluster_cum_wins:
                    temp = estimate_score_vector(c_win)
                    score_vectors.append(temp)

                # 3. Estimate support
                est_sup = estimate_support(score_vectors)

                # 4. Estimate Time Delay
                if time_data is not None:
                    #t_data = time_data["time_data"]
                    #use_gp = time_data["use_gp"]
                    #fuzzy_mf = time_data["tri_mf"]
                    gp_set = set([gi.to_string() for gi in cluster_gis])
                    bin_data = np.ones([self.row_count, self.row_count], dtype=bool)
                    selected_rows = GP.get_selected_rows(packed_bit_mat, self.row_count, )
                    time_lag = TimeDelay.approx_time_lag(selected_rows, time_data, gp_set=gp_set)
                    # time_lag = TimeDelay.approx_time_lag(bin_data, t_data, gp_set=gp_set, tri_mf_data=fuzzy_mf)

                # 5. Infer GPs from the clusters
                if est_sup >= self.thd_supp:
                    # Create GP object
                    gp: GP|TGP = TGP() if time_data is not None else GP()
                    for gi in cluster_gis:
                        GP.add_gradual_item_strict(gp, gi, target_col=target_col , time_lag=time_lag)
                    gp.support = est_sup

                    # 4a. Check if the GP candidate is valid (has more than one GI)
                    length_ok = (len(gp.gradual_items) > 1)
                    # 4b. Check if target-feature is present in the GP candidate
                    target_col_ok = self.check_target_feature(gp, exclude_target=exclude_target)
                    is_valid = (length_ok and target_col_ok)
                    if is_valid:
                        self.add_gradual_pattern(gp)
        return True

    def discover(self, target_col: int | None = None, time_data: dict|None= None, exclude_target: bool = False) -> dict:
        """
        Applies spectral clustering to determine which gradual items belong to the same group based on the similarity
        of net-win vectors. Gradual items in the same cluster should have almost the same score vector. The candidates
        are validated if their computed support is greater than or equal to the minimum support threshold specified by
        the user.

        :param target_col: Target feature's column index.
        :param time_data: (optional) time data for estimating time lag.
        :param exclude_target: Only accept GP candidates that do not contain the target feature.

        :return: A dict object
        """

        start_time = time.time()
        self._target_col = target_col
        self.clear_gradual_patterns()
        # 1. Generate net-win matrices
        s_matrix = self._net_win_mat  # Net-win matrix (S)
        s_matrix_size = s_matrix.size if s_matrix is not None else 0
        if s_matrix_size < 1:
            raise Exception("Erasure probability is too high, consider reducing it.")

        # 2a. Spectral Clustering: perform SVD to determine the independent rows
        u, s, vt = np.linalg.svd(s_matrix)

        # 2b. Spectral Clustering: compute rank of net-wins matrix
        r = np.linalg.matrix_rank(s_matrix)  # approximated r

        # 2c. Spectral Clustering: rank approximation
        s_matrix_approx = u[:, :r] @ np.diag(s[:r]) @ vt[:r, :]

        # 2d. Clustering using K-Means (using the sklearn library)
        kmeans = KMeans(n_clusters=r, random_state=0)
        y_predicted = kmeans.fit_predict(s_matrix_approx)

        # 3. Infer GPs
        self._infer_gps(y_predicted, exclude_target=exclude_target, time_data=time_data)

        duration = time.time() - start_time
        out_dict: dict[str, str | list] = {
            "Algorithm": "Clu-GRAANK",
            # "Memory Usage (MiB)": f{mem_use)}"
            "Erasure probability": f"{self._erasure_probability}",
            "Number of iterations": f"{self._max_iteration}",
            "Run-time": f"{duration:.6f} seconds",
            "Invalid Count": f"{0}"}

        return out_dict
