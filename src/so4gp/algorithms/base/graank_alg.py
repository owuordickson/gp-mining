# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# See the LICENSE file at the root of this
# repository for complete details.

import copy
import time
import torch
import numpy as np
from itertools import combinations

from .graank_base import BaseGrad
from ...data_gp import DataGP
from ...gradual_patterns import GI, GP, TGP


class OrigGRAANK(BaseGrad):

    def __init__(self, *args, max_apriori_level: int|None=None, **kwargs):
        """
        Extracts gradual patterns (GPs) from a numeric dataset using the GRAANK algorithm. The algorithm relies on the
        APRIORI approach for generating GP candidates. This work was proposed by Anne Laurent
        and published in: https://link.springer.com/chapter/10.1007/978-3-642-04957-6_33.

             A GP is a set of gradual items (GI), and its quality is measured by its computed support value. For example,
             given a data set with 3 columns (age, salary, cars) and 10 objects. A GP may take the form: {age+, salary-}
             with a support of 0.8. This implies that 8 out of 10 objects have the values of column age 'increasing' and
             column 'salary' decreasing.

        This class extends class DataGP which is responsible for generating the GP bitmaps.

        :param args: [required] data source path of Pandas DataFrame, [optional] minimum-support, [optional] eq

        :param max_apriori_level: [optional] Maximum APRIORI level for generating candidates.

        """
        super(OrigGRAANK, self).__init__(*args, **kwargs)
        self._max_apriori_level: int|None = max_apriori_level

    def _gen_apriori_candidates(self, valid_dict: dict|None, time_data: dict|None= None, exclude_target: bool = False) -> dict:
        """
        Generates Apriori GP candidates (w.r.t target-feature/reference-column if provided). If a user wishes to generate
        candidates that do not contain the target-feature, then they do so by specifying the exclude_target parameter.

        :param valid_dict: List of GIs/GPs together with bitmap arrays.
        :param exclude_target: Only accepts GP candidates that do not contain the target feature.

        :return: List of extracted GPs.
        """

        res_dict = {}
        if valid_dict is None:
            return res_dict

        search_space = self.search_space
        min_sup = self.thd_supp
        dim = self.attr_size
        all_candidates = []

        for key1, key2 in combinations(valid_dict, 2):
            pw_mat1 = valid_dict[key1]
            pw_mat2 = valid_dict[key2]

            # 1. Create a GP candidate by union of both sets
            gp_cand: GP = GP()
            gp_cand_set = set(pw_mat1.pattern) | set(pw_mat2.pattern)
            for gi_str in gp_cand_set:
                gi: GI = GI.from_string(gi_str)
                gp_cand.add_gradual_item(gi)

            # 2a. Check if the GP candidate is valid (has more than one GI)
            length_ok = (len(gp_cand.gradual_items) > 1)
            # 2b. Check if target-feature is present in the GP candidate
            target_col_ok = self.check_target_feature(gp_cand, exclude_target=exclude_target)
            # 2c. Check if the GP candidate is already present in the list of candidates
            not_exists = (not gp_cand.is_duplicate(all_candidates))
            is_valid = (length_ok and target_col_ok and not_exists)
            if not is_valid:
                continue

            # 3. Compute the support of the GP candidate
            all_candidates.append(gp_cand)
            pw_mat = GP.perform_and(pw_mat1, pw_mat2, dim, time_data)
            if pw_mat.support > min_sup:
                res_dict[tuple(pw_mat.pattern)] = pw_mat
            else:
                if search_space is not None:
                    search_space.invalid_count += 1
        return res_dict

    def discover(self, target_col: int|None = None, time_data: dict|None= None, exclude_target: bool = False,
                 compute_descriptors: bool = True) -> dict:
        """
        Uses apriori algorithm to find gradual pattern (GP) candidates. The candidates are validated if their computed
        support is greater than or equal to the minimum support threshold specified by the user.

        :param target_col: Target feature's column index.
        :param time_data: (optional) time data for estimating time lag.
        :param exclude_target: Only accept GP candidates that do not contain the target feature.
        :param compute_descriptors: [optional] compute descriptors for each GP candidate.

        :return: A dict object
        """

        start = time.time()
        self._target_col = target_col
        s_space = self.blank_search_space()
        if s_space is None:
            return {"Error": "Search space is empty!"}
        valid_bins_dict: dict|None = copy.deepcopy(self.valid_bins)

        if valid_bins_dict is None:
            return {"Error": "Pairwise matrices not available!"}

        candidate_level = 1
        while valid_bins_dict:
            valid_bins_dict = self._gen_apriori_candidates(valid_bins_dict, time_data=time_data, exclude_target=exclude_target)

            for gp_set, gi_data in (valid_bins_dict or {}).items():
                self.remove_subsets(set(gp_set))
                gp: GP|TGP = TGP() if time_data is not None else GP()

                for gi_str in gp_set:
                    gi: GI = GI.from_string(gi_str)
                    GP.add_gradual_item_strict(gp, gi, target_col=target_col, time_lag=gi_data.time_lag)
                gp.support = gi_data.support
                if compute_descriptors:
                    n = self._attr_size
                    warping_set_arr: np.ndarray|torch.Tensor = DataGP.gen_gradual_warping_set(gi_data.packed_bin_mat, n)
                    gp.compute_descriptors(warping_set_arr, obj_count=self.row_count)
                self.add_gradual_pattern(gp)
            candidate_level += 1
            if (self._max_apriori_level is not None) and candidate_level >= self._max_apriori_level:
                break

        duration = time.time() - start
        out_dict: dict[str, str|list]= {
            "Algorithm": "GRAANK",
            # "Memory Usage (MiB)": f{mem_use)}"
            "Run-time": f"{duration:.6f} seconds",
            "Invalid Count": f"{s_space.invalid_count}"}
        return out_dict
