# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0.
# See the LICENSE file at the root of this
# repository for complete details.

import gc
import json
import time
import numpy as np
from typing import cast

from .graank_base import BaseGrad
from ..gradual_patterns import GI, GP, PairwiseMatrix


class AntGRAANK(BaseGrad):

    def __init__(self, *args, max_iter: int = 1, e_factor: float = 0.5, **kwargs):
        """Extract gradual patterns (GPs) from a numeric data source using the Ant Colony Optimization approach
    (proposed in a published paper by Dickson Owuor). A GP is a set of gradual items (GI), and its quality is
    measured by its computed support value. For example, given a data set with 3 columns (age, salary, cars) and 10
    objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects
    have the values of column age 'increasing' and column 'salary' decreasing.

    In this approach, it is assumed that every column can be converted into a gradual item (GI). If the GI is valid
    (i.e., its computed support is greater than the minimum support threshold), then it is either increasing or
    decreasing (+ or -), otherwise it is irrelevant (x). Therefore, a pheromone matrix is built using the number of
    columns and the possible variations (increasing, decreasing, irrelevant) or (+, -, x). The algorithm starts by
    randomly generating GP candidates using the pheromone matrix, each candidate is validated by confirming that
    its computed support is greater or equal to the minimum support threshold. The valid GPs are used to update the
    pheromone levels and better candidates are generated.

    :param args: [required] data source path of Pandas DataFrame, [optional] minimum-support, [optional] eq
    :param max_iter: [optional] maximum_iteration, default is 1
    :param e_factor: [optional] evaporation factor, default is 0.5

        >>> from so4gp.algorithms import AntGRAANK
        >>> import pandas
        >>>
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = AntGRAANK(data_source=dummy_df, min_sup=0.5, max_iter=3, e_factor=0.5)
        >>> result_json = mine_obj.discover()
        >>> # print(result['Patterns'])
        >>> print(result_json) # doctest: +SKIP
        {"Algorithm": "ACO-GRAANK", "Best Patterns": [[["Expenses-", "Age+"], 1.0]], "Invalid Count": 1, "Iterations":3}

        """
        super(AntGRAANK, self).__init__(*args, **kwargs)
        self._evaporation_factor: float = e_factor
        self._max_iteration: int = max_iter
        self._distance_matrix: np.ndarray | None = None
        self._attribute_keys: list[str] | None = None

    def _fit(self):
        """
        Generates the distance matrix (d)
        :return: distance matrix (d) and attribute keys
        """

        # Get valid GI bitmaps
        gi_dict = self.valid_bins

        # 1. Fetch valid bins group
        gi_key_list = list(gi_dict.keys()) if gi_dict is not None else []
        attr_keys = [GI.from_string(gi_str).to_string() for gi_str in gi_key_list]

        # 2. Initialize an empty d-matrix
        n = len(attr_keys)
        d = np.zeros((n, n), dtype=np.dtype('i8'))  # cumulative sum of all segments
        for i in range(n):
            for j in range(n):
                if GI.from_string(attr_keys[i]).attribute_col == GI.from_string(attr_keys[j]).attribute_col:
                    # Ignore similar attributes (+ or/and -)
                    continue
                else:
                    if gi_dict is None:
                        continue
                    res_pw_mat: PairwiseMatrix = GP.perform_and(gi_dict[attr_keys[i]], gi_dict[attr_keys[j]], n)
                    # Cumulative sum of all segments for 2x2 (all attributes) gradual items
                    d[i][j] += np.sum(res_pw_mat.bin_mat)
        # print(d)
        self._distance_matrix = d
        self._attribute_keys: list[str] = attr_keys
        gc.collect()

    def _gen_aco_candidates(self, p_matrix):
        """
        Generates GP candidates based on the pheromone levels.

        :param p_matrix: The pheromone matrix
        :type p_matrix: np.ndarray
        :return: pheromone matrix (ndarray)
        """
        v_matrix = self._distance_matrix
        pattern: GP = GP()
        if v_matrix is None:
            return pattern, p_matrix

        # 1. Generate gradual items with the highest pheromone and visibility
        m = p_matrix.shape[0]
        for i in range(m):
            combine_feature = np.multiply(v_matrix[i], p_matrix[i])
            total = np.sum(combine_feature)
            with np.errstate(divide='ignore', invalid='ignore'):
                probability = combine_feature / total
            cum_prob = np.cumsum(probability)
            r = np.random.random_sample()
            try:
                j = np.nonzero(cum_prob > r)[0][0]
                gi_str: str = cast(str, self._attribute_keys[j]) if self._attribute_keys is not None else ""
                gi: GI = GI.from_string(gi_str)
                if not pattern.contains_attr(gi):
                    pattern.add_gradual_item(gi)
            except IndexError:
                continue

        # 2. Evaporate pheromones by factor e
        p_matrix = (1 - self._evaporation_factor) * p_matrix
        return pattern, p_matrix

    def _update_pheromones(self, pattern: GP, p_matrix: np.ndarray):
        """
        Updates the pheromone level of the pheromone matrix

        :param pattern: pattern used to update values
        :param p_matrix: an existing pheromone matrix
        :return: updated pheromone matrix
        """
        if self._attribute_keys is None:
            return p_matrix

        idx = [self._attribute_keys.index(x.to_string()) for x in pattern.gradual_items]
        for n in range(len(idx)):
            for m in range(n + 1, len(idx)):
                i = idx[n]
                j = idx[m]
                p_matrix[i][j] += 1
                p_matrix[j][i] += 1
        return p_matrix

    def discover(self, save_results: bool = True) -> str:
        """
        Applies ant-colony optimization algorithm and uses pheromone levels to find GP candidates. The candidates are
        validated if their computed support is greater than or equal to the minimum support threshold specified by the
        user.

        :param save_results: [optional] Save results to a csv file.

        :return: JSON string object
        """

        start = time.time()
        self.init_search_space(0, 0)
        self._fit()  # distance matrix (d) & attributes corresponding to d

        d = self._distance_matrix
        if d is None:
            out = json.dumps(
                {"Algorithm": "ACO-GRAANK", "Best Patterns": self.display_patterns, "Invalid Count": 0, "Iterations": 0})
            """:type str: object"""
            return out

        a = self.attr_size
        loser_gps = list()  # supersets
        repeated = 0
        it_count = 0
        counter = 0

        if self.valid_bins is None:
            return "Pairwise matrices not available!"

        # 1. Remove d[i][j] < frequency-count of min_supp
        fr_count = ((self.thd_supp * a * (a - 1)) / 2)
        d[d < fr_count] = 0

        # 3. Initialize pheromones (p_matrix)
        pheromones = np.ones(d.shape, dtype=float)

        invalid_count = 0
        # 4. Iterations for ACO
        # while repeated < 1:
        while counter < self._max_iteration:
            rand_gp, pheromones = self._gen_aco_candidates(pheromones)
            if len(rand_gp.gradual_items) > 1:
                # print(rand_gp.get_pattern())
                exits = rand_gp.is_duplicate(self.gradual_patterns, loser_gps)
                if not exits:
                    repeated = 0
                    # check for anti-monotony
                    is_super = rand_gp.check_am(loser_gps, subset=False)
                    is_sub = rand_gp.check_am(self.gradual_patterns, subset=True)
                    if is_super or is_sub:
                        continue
                    gen_gp: GP = rand_gp.validate_graank(self)
                    is_present = gen_gp.is_duplicate(self.gradual_patterns, loser_gps)
                    is_sub = gen_gp.check_am(self.gradual_patterns, subset=True)
                    if is_present or is_sub:
                        repeated += 1
                    else:
                        if gen_gp.support >= self.thd_supp:
                            pheromones = self._update_pheromones(gen_gp, pheromones)
                            self.add_gradual_pattern(gen_gp)
                        else:
                            loser_gps.append(gen_gp)
                            invalid_count += 1
                    if gen_gp.as_set != rand_gp.as_set:
                        loser_gps.append(rand_gp)
                else:
                    repeated += 1
            else:
                invalid_count += 1
            it_count += 1
            if self._max_iteration == 1:
                counter = repeated
            else:
                counter = it_count

        duration = time.time() - start
        out_dict: dict[str, str | list] = {
            "Algorithm": "ACO-GRAANK",
            # "Memory Usage (MiB)": f{mem_use)}"
            "Evaporation factor": f"{self._evaporation_factor}",
            "Number of iterations": f"{it_count}",
            "Run-time": f"{duration:.6f} seconds"}
        if save_results:
            self.generate_output_files(out_dict)

        out_dict.update({"Best Patterns": self.display_patterns, "Invalid Count": str(invalid_count)})
        out: str = json.dumps(out_dict, indent=4)
        return out
