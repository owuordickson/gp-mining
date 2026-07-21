# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0.
# See the LICENSE file at the root of this
# repository for complete details.

import gc
import copy
import time
import numpy as np

from .graank_base import BaseGrad
from ...data_gp import DataGP
from ...gradual_patterns import GI, GP, PairwiseMatrix


class OrigGRAANK(BaseGrad):

    def __init__(self, *args, **kwargs):
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

        """
        super(OrigGRAANK, self).__init__(*args, **kwargs)

    def _gen_apriori_candidates(self, gi_dict: dict|None, ignore_sup: bool = False,
                                target_col: int | None = None, exclude_target: bool = False):
        """
        Generates Apriori GP candidates (w.r.t target-feature/reference-column if provided). If a user wishes to generate
        candidates that do not contain the target-feature, then they do so by specifying the exclude_target parameter.

        :param gi_dict: List of GIs together with bitmap arrays.
        :param ignore_sup: Do not filter GPs based on the minimum support threshold.
        :param target_col: Target feature's column index.
        :param exclude_target: Only accepts GP candidates that do not contain the target feature.
        :return: List of extracted GPs and the invalid count.
        """

        def invert_symbol(gi_item: str) -> str:
            """Description

            Computes the inverse of a GI formatted as an array or tuple

            :param gi_item: gradual item as a string (e.g., '1+' or '1-')
            :return: inverted gradual item
            """
            if gi_item.endswith("+"):
                return gi_item.replace("+", "-")
            elif gi_item.endswith("-"):
                return gi_item.replace("-", "+")
            else:
                return gi_item

        min_sup = self.thd_supp
        n = self.attr_size

        if gi_dict is None:
            return {}, 0

        all_candidates = []
        invalid_count = 0
        res_dict = {}

        gi_key_list = list(gi_dict.keys())
        for i in range(len(gi_dict) - 1):
            for j in range(i + 1, len(gi_dict)):
                # 1. Fetch pairwise matrix
                gi_str_i = gi_key_list[i]
                gi_str_j = gi_key_list[j]

                """
                TO DELETE
                try:
                    gi_i = {list(gi_str_i) if isinstance(gi_str_i, tuple) else gi_str_i}
                    gi_j = {gi_str_j}
                    gi_o = {gi_key_list[0]}
                except TypeError:
                    gi_i = set(list(gi_str_i) if isinstance(gi_str_i, tuple) else gi_str_i)
                    gi_j = set(list(gi_str_j) if isinstance(gi_str_j, tuple) else gi_str_j)
                    gi_o = set(gi_key_list[0])
                """
                if isinstance(gi_str_i, (tuple, list)):
                    gi_i = set(gi_str_i)
                    gi_o = set(gi_key_list[0])
                else:
                    gi_i = {gi_str_i}
                    gi_o = {gi_key_list[0]}

                if isinstance(gi_str_j, (tuple, list)):
                    gi_j = set(gi_str_j)
                else:
                    gi_j = {gi_str_j}

                # 2. Identify a GP candidate (create its inverse)
                gp_cand = gi_i | gi_j
                inv_gp_cand = {invert_symbol(x) for x in gp_cand}

                # 3. Apply target-feature search
                target_col_ok = BaseGrad.apply_target_feature(gp_cand, target_col=target_col, exclude_target=exclude_target)
                if not target_col_ok:
                    continue

                # 4. Verify the validity of the GP candidate through the following conditions
                is_length_valid = (len(gp_cand) == len(gi_o) + 1)
                is_unique_candidate = ((not (all_candidates != [] and gp_cand in all_candidates)) and
                                    (not (all_candidates != [] and inv_gp_cand in all_candidates)))

                # 4. Validate GP and save it
                if is_length_valid and is_unique_candidate:
                    test = 1
                    repeated_attr = -1
                    for k in gp_cand:
                        if k[0] == repeated_attr:
                            test = 0
                            break
                        else:
                            repeated_attr = k[0]
                    if test == 1:
                        res_pw_mat: PairwiseMatrix = GP.perform_and(gi_dict[gi_str_i], gi_dict[gi_str_j], n)
                        if res_pw_mat.support > min_sup or ignore_sup:
                            # res_dict.append([gp_cand, bin_mat, sup])
                            res_dict[tuple(gp_cand)] = res_pw_mat
                        else:
                            invalid_count += 1
                    all_candidates.append(gp_cand)
                    gc.collect()
        return res_dict, invalid_count

    def discover(self, ignore_support: bool = False, apriori_level: int | None = None,
                 target_col: int | None = None, exclude_target: bool = False, compute_descriptors: bool = True) -> dict:
        """
        Uses apriori algorithm to find gradual pattern (GP) candidates. The candidates are validated if their computed
        support is greater than or equal to the minimum support threshold specified by the user.

        :param ignore_support: Do not filter extracted GPs using a user-defined minimum support threshold.
        :param apriori_level: Maximum APRIORI level for generating candidates.
        :param target_col: Target feature's column index.
        :param exclude_target: Only accept GP candidates that do not contain the target feature.
        :param compute_descriptors: [optional] compute descriptors for each GP candidate.

        :return: A dict object
        """

        start = time.time()
        self.init_search_space(0, 0)
        valid_bins_dict: dict|None = copy.deepcopy(self.valid_bins)

        if valid_bins_dict is None:
            return {"Error": "Pairwise matrices not available!"}

        invalid_count = 0
        candidate_level = 1
        while valid_bins_dict:
            valid_bins_dict, inv_count = self._gen_apriori_candidates(valid_bins_dict,
                                                                 ignore_sup=ignore_support,
                                                                 target_col=target_col,
                                                                 exclude_target=exclude_target)
            invalid_count += inv_count
            for gp_set, gi_data in (valid_bins_dict or {}).items():
                self.remove_subsets(set(gp_set))
                gp: GP = GP()
                for gi_str in gp_set:
                    gi: GI = GI.from_string(gi_str)
                    gp.add_gradual_item(gi)
                gp.support = gi_data.support
                if compute_descriptors:
                    warping_set_arr: np.ndarray = np.array(DataGP.gen_gradual_warping_set(gi_data.bin_mat, as_array=True))
                    gp.compute_descriptors(warping_set_arr, obj_count=self.row_count)
                self.add_gradual_pattern(gp)
            candidate_level += 1
            if (apriori_level is not None) and candidate_level >= apriori_level:
                break

        duration = time.time() - start
        out_dict: dict[str, str|list]= {
            "Algorithm": "GRAANK",
            # "Memory Usage (MiB)": f{mem_use)}"
            "Run-time": f"{duration:.6f} seconds",
            "Invalid Count": f"{invalid_count}"}
        return out_dict
