# SPDX-License-Identifier: MIT
# See the LICENSE file at the root of this
# repository for complete details.


import random
import time

from .graank_base import BaseGrad


class HillClimbingGRAANK(BaseGrad):
    def __init__(self, *args, max_iter: int = 1, step_size: float = 0.5, **kwargs):
        """
        Extract gradual patterns (GPs) from a numeric data source using the Hill Climbing (Local Search) Algorithm
        approach (proposed in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI), and its
        quality is measured by its computed support value. For example, given a data set with 3 columns (age, salary,
        cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of
        10 objects have the values of column age 'increasing' and column 'salary' decreasing.

        In this approach, we assume that every GP candidate may be represented as a position that has cost value
        associated with it. The cost is derived from the computed support of that candidate, the higher the support
        value, the lower the cost. The aim of the algorithm is to search through a group of positions and find those with
        the lowest cost as efficiently as possible.

        :param args: [required] a data source path of Pandas DataFrame, [optional] minimum-support, [optional] eq
        :param max_iter: [optional] maximum_iteration, default is 1
        :param step_size: [optional] step size, default is 0.5

        """
        super().__init__(*args, **kwargs)
        self._step_size: float = step_size
        self._max_iteration: int = max_iter
        self._n_var: int = 1

    def discover(
        self,
        target_col: int | None = None,
        time_data: dict | None = None,
        exclude_target: bool = False,
    ) -> dict:
        """
        Uses hill-climbing algorithm to find GP candidates. The candidates are validated if their computed support is
        greater than or equal to the minimum support threshold specified by the user.

        :param target_col: Target feature's column index.
        :param time_data: (optional) time data for estimating time lag.
        :param exclude_target: Only accept GP candidates that do not contain the target feature.

        :return: A dict object
        """

        start = time.time()
        self.target_col = target_col
        s_space = self.blank_search_space()
        if s_space is None:
            return {"Error": "Search space is empty!"}

        # run the hill climb
        candidate = BaseGrad.Candidate()
        while s_space.iter_count < self._max_iteration:
            # while eval_count < max_evaluations:
            # take a step
            candidate.position = None
            if candidate.position is None:
                best_pos = s_space.best_candidate.position
                if best_pos is not None:
                    candidate.position = best_pos + (
                        random.randrange(s_space.var_min, s_space.var_max)
                        * self._step_size
                    )

            # Evaluate candidate
            self.evaluate_candidate(candidate, exclude_target, time_data=time_data)

            # Increment iteration count
            s_space.iter_count += 1

        duration = time.time() - start
        out_dict: dict[str, str | list] = {
            "Algorithm": "LS-GRAANK",
            # "Memory Usage (MiB)": f{mem_use)}"
            "Step Size": f"{self._step_size}",
            "Number of iterations": f"{s_space.iter_count}",
            "Run-time": f"{duration:.6f} seconds",
            "Invalid Count": f"{s_space.invalid_count}",
        }
        return out_dict
