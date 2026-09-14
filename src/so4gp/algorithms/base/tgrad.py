# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# See the LICENSE file at the root of this
# repository for complete details.


import time
import copy
import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from .graank_alg import OrigGRAANK
from ..graank import GRAANK
from ...data_gp import DataGP
from ...gradual_patterns import TGP, NO_TIME_LABEL


class TGrad(OrigGRAANK):

    def __init__(self, *args, min_rep: float = 0.5, **kwargs):
        """
        TGrad is an algorithm used to extract temporal gradual patterns from numeric datasets. An algorithm for mining
        temporal gradual patterns using fuzzy membership functions. It uses a technique
        published in: https://ieeexplore.ieee.org/abstract/document/8858883.

        :param args: [required] a data source path of Pandas DataFrame, [optional] minimum-support, [optional] eq
        :param min_rep: [optional] minimum representativity value.

        """

        super(TGrad, self).__init__(*args, **kwargs)
        self._search_algorithm: str = "apriori"
        self._algorithm_max_iter: int = 3
        self._min_rep: float = min_rep
        self._max_step: int = self.row_count - int(min_rep * self.row_count)
        self._full_attr_data: np.ndarray = copy.deepcopy(self.data).T
        if len(self.time_cols) > 0:
            # print("Dataset Ok")
            self._time_ok: bool = True
        else:
            # print("Dataset Error")
            self._time_ok: bool = False
            raise Exception('No date-time datasets found')

    @property
    def min_rep(self):
        return self._min_rep

    @property
    def max_step(self):
        return self._max_step

    @property
    def full_attr_data(self):
        return self._full_attr_data

    @min_rep.setter
    def min_rep(self, value):
        if 0 < value <= 1:
            self._min_rep = value

    def discover_tgp(self, target_col: int, search_algorithm: str="apriori", max_iteration: int=3, num_cores: int = 1) -> dict:
        """
        Mine Fuzzy Temporal Gradual Patterns (FTGPs) from a temporal dataset.

        The method applies the complete temporal gradual pattern mining pipeline:
        fuzzy-logic-based data transformation, temporal transformation, and
        gradual pattern mining. The temporal transformations are estimated
        relative to a specified target attribute, after which a gradual pattern
        search algorithm is applied to the transformed dataset.

        The gradual pattern search can use the classical APRIORI (GRAANK) algorithm or
        one of several metaheuristic and alternative search strategies.

        Supported search algorithms are:

        * ``apriori``:
          Classical APRIORI level-wise search for exhaustive gradual pattern
          candidate generation.

        * ``ga``:
          Genetic GRAANK. Uses a genetic algorithm to search the gradual pattern
          space.

        * ``aco``:
          ACO-GRAANK. Uses Ant Colony Optimization and pheromone-guided search
          to identify promising gradual pattern candidates.

        * ``pso``:
          PSO-GRAANK. Uses Particle Swarm Optimization to search for high-support
          gradual patterns.

        * ``hc``:
          Hill-Climbing GRAANK. Iteratively searches neighboring candidates and
          moves toward patterns with improved support.

        * ``random``:
          Random Search GRAANK. Randomly samples and evaluates gradual pattern
          candidates.

        * ``clustergp``:
          ClusterGP. Uses clustering-based search to identify gradual patterns
          from the transformed dataset.

        Args:
            target_col:
                Index of the target attribute or feature. Temporal transformations
                and time-delay estimation are performed relative to this attribute.

            search_algorithm:
                Gradual pattern mining algorithm to apply to the transformed
                dataset. Supported values are ``apriori``, ``ga``, ``aco``,
                ``pso``, ``hc``, ``random``, and ``clustergp``.
                Defaults to ``"apriori"``.

            max_iteration:
                The maximum number of iterations to run the search algorithm.

            num_cores:
                Number of CPU cores available for parallel computation during
                temporal transformation and pattern mining.

        Returns:
            A list containing the mined Fuzzy Temporal Gradual Patterns.

        Raises:
            ValueError:
                If ``target_col`` is invalid or ``search_algorithm`` is not one
                of the supported algorithms.

            TypeError:
                If ``target_col`` or another argument has an invalid type.

        Notes:
            Metaheuristic search algorithms such as ``ga``, ``aco``, ``pso``,
            ``hc``, and ``random`` generally provide approximate solutions and
            may not enumerate all frequent gradual patterns. APRIORI provides
            exhaustive level-wise candidate generation subject to the configured
            search constraints.

            Multiprocessing can significantly reduce computation time for large
            datasets, particularly during the temporal transformation and
            evaluation stages.
        """

        start = time.time()
        self._target_col = target_col
        self._search_algorithm = search_algorithm
        self._algorithm_max_iter = max_iteration
        self.clear_gradual_patterns()

        # 1. Mine FTGPs (using parallel multi-processing)
        #with mp.Pool(num_cores) as pool:
        #    steps = range(1, self._max_step)
        #    pattern_data = pool.map(self._safe_transform_and_mine, steps)
        pattern_data = []
        for step in (1, self._max_step):
            pattern_data.append(self._safe_transform_and_mine(step))

        # 2. Organize FTGPs into a single list
        for item in pattern_data:
            if item is None:
                continue

            # Standardize 'item' into a list so we only need one loop
            lst_pattern = item if isinstance(item, list) else [item]

            for pat in lst_pattern:
                if isinstance(pat, TGP):
                    self.add_gradual_pattern(pat)

        duration = time.time() - start
        out_dict: dict[str, str | list] = {
            "Algorithm": "TGrad",
            # "Memory Usage (MiB)": f{mem_use)}",
            "GP Search Algorithm": f"{self._search_algorithm}",
            "Maximum Iteration for Search Algorithm": f"{self._algorithm_max_iter}",
            "Minimum Representation": f"{self.min_rep:.2f}",
            "Target Column": f"{target_col}",
            "Run-time": f"{duration:.6f} seconds"}
        return out_dict

    def transform_and_mine(self, step: int, return_patterns: bool = True):
        """
        A method that: (1) transforms data according to a step value and, (2) mines the transformed data for FTGPs.

        :param step: Data transformation step.
        :param return_patterns: Allow method to mine TGPs.
        :return: List of TGPs
        """
        # NB: Restructure dataset based on target/reference col
        if self._time_ok:
            # 1. Calculate the time difference using a step
            ok, time_diffs, time_diffs_arr = self.get_time_diffs(step)
            if not ok:
                msg = "Error: Time in row " + str(time_diffs.keys()) \
                      + " or row " + str(time_diffs.values()) + " is not valid."
                raise Exception(msg)
            else:
                tgt_col = self._target_col
                if tgt_col in self.time_cols:
                    msg = "Target column is a 'date-time' attribute"
                    raise Exception(msg)
                elif (tgt_col < 0) or (tgt_col >= self.col_count):
                    msg = "Target column does not exist\nselect column between: " \
                          "0 and " + str(self.col_count - 1)
                    raise Exception(msg)
                else:
                    # 2. Transform datasets
                    delayed_attr_data = None
                    n = self.row_count
                    for col_index in range(self.col_count):
                        # Transform the datasets using (row) n+step
                        if (col_index == tgt_col) or (col_index in self.time_cols):
                            # date-time column OR target column
                            temp_col = self._full_attr_data[col_index][0: (n - step)]
                        else:
                            # other attributes
                            temp_col = self._full_attr_data[col_index][step: n]

                        delayed_attr_data = temp_col if (delayed_attr_data is None) \
                            else np.vstack((delayed_attr_data, temp_col))
                    # print(f"Time Diffs: {time_diffs}\n")
                    # print(f"{self.full_attr_data}: {type(self.full_attr_data)}\n")
                    # print(f"{delayed_attr_data}: {type(delayed_attr_data)}\n")

                    if return_patterns:
                        # 2. Execute t-graank for each transformation
                        t_gps = self._mine_gps_at_step(time_delay_data=time_diffs_arr, attr_data=delayed_attr_data)
                        if len(t_gps) > 0:
                            return t_gps
                        return False
                    else:
                        return delayed_attr_data, time_diffs
        else:
            msg = "Fatal Error: Time format in column could not be processed"
            raise Exception(msg)

    def _safe_transform_and_mine(self, step: int, return_patterns: bool = True):
        """Wrapper to catch exceptions during parallel mining."""
        try:
            return self.transform_and_mine(step, return_patterns=return_patterns)
        except Exception as e:
            print(f"Error at step {step}: {e}")
            return None

    def _mine_gps_at_step(self, time_delay_data: dict|np.ndarray, attr_data: np.ndarray|None = None,
                          clustering_method: bool = False) -> list[TGP]:
        """
        Uses apriori algorithm to find GP candidates based on the target-attribute. The candidates are validated if
        their computed support is greater than or equal to the minimum support threshold specified by the user.

        :param time_delay_data: Time-delay values
        :param attr_data: the transformed data.
        :param clustering_method: Find and approximate the best time-delay value using KMeans and Hill-climbing approach.
        :return: Temporal-GPs as a list.
        """

        if attr_data is None:
            return []

        if clustering_method:
            if isinstance(time_delay_data, dict):
                t_lag_arr = np.array(list(time_delay_data.values()))
            else:
                t_lag_arr = np.array(time_delay_data)

            # Build the main triangular MF using the clustering algorithm
            a, b, c = TGrad.build_mf_w_clusters(t_lag_arr)
            tri_mf_data = np.array([a, b, c])
        else:
            tri_mf_data = None

        if type(self) is TGrad:
            time_data: dict = {"time_data": time_delay_data, "use_gp": False, "tri_mf": tri_mf_data}
        else:
            time_data: dict = {"time_data": time_delay_data, "use_gp": True, "tri_mf": tri_mf_data}
        data_df = pd.DataFrame(attr_data.T, columns=self.titles)
        mine_obj = GRAANK(data_df, min_sup=self.thd_supp, eq=self._include_equal_values)
        mine_obj.discover(search_type=self._search_algorithm, target_col=self._target_col, time_data=time_data,
                          compute_descriptors=False, max_iteration=self._algorithm_max_iter,)
        return mine_obj.mining_engine.gradual_patterns

    def get_time_diffs(self, step: int) -> tuple[bool, dict, np.ndarray]:  # optimized
        """
        A method that computes the difference between 2 timestamps separated by a specific transformation step.

        :param step: Data transformation step.
        :return: Dict of time delay values
        """
        size = self.row_count
        time_diffs = {}  # {row: time-lag}
        time_diffs_arr = []
        for i in range(size):
            if i < (size - step):
                stamp_1 = 0
                stamp_2 = 0
                for col in self.time_cols:  # sum timestamps from all time-columns
                    time_col_title = self.titles[col]
                    if time_col_title == NO_TIME_LABEL:
                        stamp_1 += int(self.data[i][int(col)])
                        stamp_2 += int(self.data[i + step][int(col)])
                        continue

                    temp_1 = str(self.data[i][int(col)])
                    temp_2 = str(self.data[i + step][int(col)])
                    temp_stamp_1 = TGrad.get_timestamp(temp_1)
                    temp_stamp_2 = TGrad.get_timestamp(temp_2)
                    if (not temp_stamp_1) or (not temp_stamp_2):
                        # Unable to read time
                        return False, {i + 1: i + step + 1}, np.array(time_diffs_arr)
                    else:
                        stamp_1 += temp_stamp_1
                        stamp_2 += temp_stamp_2
                time_diff = (stamp_2 - stamp_1)
                # if time_diff < 0:
                # Error time CANNOT go backwards,
                # print(f"Problem {i} and {i + step} - {self.time_cols}")
                #    return False, [i + 1, i + step + 1]
                time_diff_abs = float(abs(time_diff))
                time_diffs[int(i)] = time_diff_abs
                time_diffs_arr.append(time_diff_abs)
        return True, time_diffs, np.array(time_diffs_arr)

    @staticmethod
    def get_timestamp(time_str: str):
        """
        A method that computes the corresponding timestamp from a DateTime string.

        :param time_str: DateTime value as a string
        :return: timestamp value
        """
        try:
            ok, stamp = DataGP.test_time(time_str)
            if ok:
                return stamp
            else:
                return False
        except ValueError:
            return False

    @staticmethod
    def build_mf_w_clusters(time_data: np.ndarray | None):
        """
        A method that builds the boundaries of a fuzzy Triangular membership function (MF) using Singular Value
        Decomposition (to estimate the number of centers) and KMeans algorithm to group time data according to the
        identified centers. We then use the largest cluster to build the MF.

        :param time_data: Time-delay values as an array.
        :return: The boundary values of the triangular membership function.
        """

        if time_data is None:
            return 0, 0, 0

        try:
            # 1. Reshape into 1-column dataset
            time_data = time_data.reshape(-1, 1)

            # 2. Standardize data
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(time_data)

            # 3. Apply SVD
            u, s, vt = np.linalg.svd(data_scaled, full_matrices=False)

            # 4. Plot singular values to help determine the number of clusters
            # Based on the plot, choose the number of clusters (e.g., 3 clusters)
            num_clusters = int(s[0])

            # 5. Perform k-means clustering
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(data_scaled)

            # 6. Get cluster centers
            centers = kmeans.cluster_centers_.flatten()

            # 7. Define membership functions to ensure membership > 0.5
            largest_mf = [0, 0, 0]
            for center in centers:
                half_width = 0.5 / 2  # since the membership value should be > 0.5
                a = center - half_width
                b = center
                c = center + half_width
                if abs(c - a) > abs(largest_mf[2] - largest_mf[0]):
                    largest_mf = [a, b, c]

            # 8. Reverse the scaling
            a = scaler.inverse_transform([[largest_mf[0]]])[0, 0]
            b = scaler.inverse_transform([[largest_mf[1]]])[0, 0]
            c = scaler.inverse_transform([[largest_mf[2]]])[0, 0]

            # 9. Shift to remove negative MF (we do not want negative timestamps)
            if a < 0:
                shift_by = abs(a)
                a = a + shift_by
                b = b + shift_by
                c = c + shift_by
            return a, b, c
        except Exception as e:
            print(e)
            return 0, 0, 0
