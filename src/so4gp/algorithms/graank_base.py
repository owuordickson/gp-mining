# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0.
# See the LICENSE file at the root of this
# repository for complete details.


import random
import numpy as np
from dataclasses import dataclass
from ..data_gp import DataGP
from ..gradual_patterns import GI, GP, PairwiseMatrix


class BaseGrad(DataGP):

    @dataclass
    class Candidate:
        position: float|None=None
        cost: float|None=None

    @dataclass
    class SearchSpace:
        var_min: int
        var_max: int
        iter_count: int
        eval_count: int
        counter: int
        invalid_count: int
        best_sol: "BaseGrad.Candidate"
        best_costs: np.ndarray
        best_patterns: list[GP]
        str_best_gps: list
        pop: list["BaseGrad.Candidate"]

    def __init__(self, *args, **kwargs):
        # Initialize DataGP
        super(BaseGrad, self).__init__(*args, **kwargs)

    def init_search_space(self, pop_size: int, max_iter: int) -> "BaseGrad.SearchSpace|str":
        """
        Initialize the search space with pairwise matrices
        :param pop_size: population size
        :param max_iter: maximum number of iterations

        :return: Search space or error message
        """
        # Prepare data set
        self.fit_bitmap()
        self.clear_gradual_patterns()
        if self.valid_bins is None:
            return "Pairwise matrices not available!"

        # Initialize search space
        s_space = BaseGrad.initialize_search_space(self.valid_bins, pop_size, max_iter)
        if s_space is None:
            return "Search space is empty!"
        return s_space

    @staticmethod
    def initialize_search_space(valid_bins_dict: dict | None, total_pop: int, max_iter: int):
        """Create a population of candidate solutions."""
        if valid_bins_dict is None:
            return None

        gi_key_list = list(valid_bins_dict.keys())
        attr_keys = [GI.from_string(gi_str).to_string() for gi_str in gi_key_list]

        # Empty Individual Template
        empty_candidate = BaseGrad.Candidate (
            position=None,
            cost=None
        )

        # Initialize Population
        var_min = 0
        var_max = int(''.join(['1'] * len(attr_keys)), 2)
        pop = [empty_candidate] * total_pop
        for i in range(total_pop):
            pop[i].position = random.randrange(var_min, var_max)
            pop[i].cost = 1

        # Initialize best candidate
        best_candidate = BaseGrad.Candidate(
            position=pop[0].position,
            cost = BaseGrad.cost_function(pop[0].position, valid_bins_dict)
        )

        # Initialize SearchSpace parameters
        search_space = BaseGrad.SearchSpace(
            iter_count=0,
            eval_count=0,
            counter=0,
            invalid_count=0,
            var_min=var_min,
            var_max=var_max,
            best_sol=best_candidate,
            best_costs=np.empty(max_iter),
            best_patterns=[],
            str_best_gps=[],
            pop=pop,
        )
        return search_space

    @staticmethod
    def decode_gp(position: float|None, valid_bins_dict: dict|None) -> GP:
        """Description

        Decodes a numeric value (position) into a GP

        :param position: a value in the numeric search space
        :param valid_bins_dict: a dictionary of valid bins
        :return: GP that is decoded from the position value
        """

        temp_gp: GP = GP()
        if position is None or valid_bins_dict is None:
            return temp_gp

        gi_key_list = list(valid_bins_dict.keys())
        attr_keys = [GI.from_string(gi_str).to_string() for gi_str in gi_key_list]
        bin_str = bin(int(position))[2:]
        bin_arr = np.array(list(bin_str), dtype=int)

        for i in range(bin_arr.size):
            bin_val = bin_arr[i]
            if bin_val == 1:
                gi = GI.from_string(attr_keys[i])
                if not temp_gp.contains_attr(gi):
                    temp_gp.add_gradual_item(gi)
        return temp_gp

    @staticmethod
    def cost_function(position: float|None, valid_bins_dict: dict|None) -> float:
        """Description

        Computes the fitness of a GP

        :param position: a value in the numeric search space
        :param valid_bins_dict: a dictionary of valid bins
        :return: a floating point value that represents the fitness of the position
        """

        cost = 1
        if valid_bins_dict is None or position is None:
            return cost

        gi_key_list = list(valid_bins_dict.keys())
        pattern = BaseGrad.decode_gp(position, valid_bins_dict)

        pw_mat: PairwiseMatrix|None = None
        for gi in pattern.gradual_items:
            arg = np.argwhere(np.isin(np.array(gi_key_list), gi.to_string()))
            if len(arg) > 0:
                i = arg[0][0]
                bin_dict = valid_bins_dict[gi_key_list[i]]
                if pw_mat is None:
                    pw_mat = PairwiseMatrix(bin_mat=bin_dict.bin_mat, support=bin_dict.support)
                else:
                    pw_mat = GP.perform_and(pw_mat, bin_dict, -1)
        bin_sum = int(np.sum(pw_mat.bin_mat)) if pw_mat is not None else 0
        if bin_sum > 0:
            cost = (1 / bin_sum)
        return cost

    @staticmethod
    def evaluate_candidate(candidate: "BaseGrad.Candidate|None", s_space: "BaseGrad.SearchSpace|None", valid_bins_dict: dict | None)-> "BaseGrad.SearchSpace|None":
        """"""

        if candidate is None or s_space is None or valid_bins_dict is None:
            return s_space

        def apply_bound() -> None:
            """
            Modifies x (a numeric value) if it exceeds the lower/upper bound of the numeric search space.
            :return: None
            """
            candidate.position = float(np.maximum(candidate.position if candidate else 0, s_space.var_min if s_space else 0))
            candidate.position = float(np.minimum(candidate.position if candidate else 0, s_space.var_max if s_space else 0))

        apply_bound()
        # Update: What about duplicate candidate (position already exists in the search-space)?
        
        candidate.cost = BaseGrad.cost_function(candidate.position, valid_bins_dict)
        if candidate.cost == 1:
            s_space.invalid_count += 1
        if candidate.cost is not None and s_space.best_sol.cost is not None:
            if candidate.cost < s_space.best_sol.cost:
                s_space.best_sol = BaseGrad.Candidate(position=candidate.position, cost=candidate.cost)
        s_space.eval_count += 1
        return s_space

    @staticmethod
    def evaluate_gradual_pattern(repeat_count: int, s_space: "BaseGrad.SearchSpace", data_gp: DataGP) -> tuple[
        "BaseGrad.SearchSpace", int]:
        """"""
        dim = data_gp.attr_size
        best_gp: GP = BaseGrad.decode_gp(s_space.best_sol.position, data_gp.valid_bins)
        best_gp.support = float(1 / s_space.best_sol.cost) / float(dim * (dim - 1.0) / 2.0)
        is_present = best_gp.is_duplicate(s_space.best_patterns)
        is_sub = best_gp.check_am(s_space.best_patterns, subset=True)
        if is_present or is_sub:
            repeat_count += 1
        else:
            if best_gp.support >= data_gp.thd_supp:
                s_space.best_patterns.append(best_gp)
                s_space.str_best_gps.append(best_gp.print(data_gp.titles))

        try:
            # Show Iteration Information (store Best Cost)
            s_space.best_costs[s_space.iter_count] = s_space.best_sol.cost
        except IndexError:
            pass
        s_space.iter_count += 1

        s_space.counter = s_space.iter_count
        return s_space, repeat_count
