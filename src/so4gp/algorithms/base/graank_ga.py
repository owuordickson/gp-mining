# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0.
# See the LICENSE file at the root of this
# repository for complete details.


import time
import random
import numpy as np
from .graank_base import BaseGrad


class GeneticGRAANK(BaseGrad):

    def __init__(self, *args, max_iter=1, n_pop=5, pc=0.5, gamma=1.0, mu=0.9, sigma=0.9, **kwargs):
        """
        Extract gradual patterns (GPs) from a numeric data source using the Genetic Algorithm approach (proposed
        in a published paper by Dickson Owuor). A GP is a set of gradual items (GI), and its quality is measured by
        its computed support value. For example, given a data set with 3 columns (age, salary, cars) and 10 objects.
        A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects have the
        values of column age 'increasing' and column 'salary' decreasing.

             In this approach, we assume that every GP candidate may be represented as a binary gene (or individual)
             that has a unique position and cost. The cost is derived from the computed support of that candidate, the
             higher the support value, the lower the cost. The aim of the algorithm is to search through a population of
             individuals (or candidates) and find those with the lowest cost as efficiently as possible.

        :param args: [required] a data source path of Pandas DataFrame, [optional] minimum-support, [optional] eq
        :param max_iter: [optional] maximum_iteration, default is 1
        :type max_iter: int

        :param n_pop: [optional] initial individual population, default is 5
        :type n_pop: int

        :param pc: [optional] children proportion, default is 0.5
        :type pc: float

        :param gamma: [optional] cross-over gamma ratio, the default is 1
        :type gamma: float

        :param mu: [optional] mutation mu ratio, default is 0.9
        :type mu: float

        :param sigma: [optional] mutation sigma ratio, default is 0.9
        :type sigma: float

        """
        super(GeneticGRAANK, self).__init__(*args, **kwargs)
        self._max_iteration: int = max_iter
        self._parent_pop: int = n_pop
        self._children_pop: float = pc
        self._gamma: float = gamma
        self._mu: float = mu
        self._sigma: float = sigma
        print(f"GA Population: {n_pop}")

    def _crossover(self, p1: BaseGrad.Candidate, p2: BaseGrad.Candidate) -> tuple[BaseGrad.Candidate, BaseGrad.Candidate]:
        """
        Crosses over the genes of 2 parents (an individual with a specific position and cost) to generate 2
        different offsprings.

        :param p1: The parent-1 individual
        :param p2: The parent-2 individual
        :return: Two offsprings (children)
        """
        c1 = BaseGrad.Candidate()
        c2 = BaseGrad.Candidate()
        alpha: float = random.uniform(0, self._gamma)
        c1.position = float(alpha * p1.position + (1 - alpha) * p2.position)
        c2.position = float(alpha * p2.position + (1 - alpha) * p1.position)
        return c1, c2

    def _mutate(self, x: BaseGrad.Candidate):
        """

        Mutates an individual's position to create a new and different individual.

        :param x: The existing individual
        :return: A new individual
        """
        y = BaseGrad.Candidate(position=x.position, cost=x.cost)
        str_x = str(int(y.position) if y.position is not None else 0)
        flag = np.random.rand(*(len(str_x),)) <= self._mu
        ind = np.argwhere(flag)
        str_y = "0"
        for i in ind:
            val = float(str_x[i[0]])
            val += self._sigma * random.uniform(0, 1)
            if i[0] == 0:
                str_y = "".join(("", "{}".format(int(val)), str_x[1:]))
            else:
                str_y = "".join((str_x[:i[0] - 1], "{}".format(int(val)), str_x[i[0]:]))
            str_x = str_y
        y.position = int(str_y)
        return y

    def discover(self, ignore_support: bool = False, target_col: int | None = None, exclude_target: bool = False) -> dict:
        """
        Uses genetic algorithm to find GP candidates. The candidates are validated if their computed support is greater
        than or equal to the minimum support threshold specified by the user.

        :param ignore_support: Do not filter extracted GPs using a user-defined minimum support threshold.
        :param target_col: Target feature's column index.
        :param exclude_target: Only accept GP candidates that do not contain the target feature.

        :return: A dict object
        """

        start = time.time()
        s_space = self.init_search_space(self._parent_pop, self._max_iteration)
        if isinstance(s_space, str):
            return {"Error": s_space}

        num_children = int(np.round(self._children_pop * self._parent_pop / 2) * 2)  # Number of children np.round is used to get an even number
        repeated = 0
        while s_space.counter < self._max_iteration:

            c_pop = []  # Children population
            for _ in range(num_children // 2):
                # Select Parents
                q = np.random.permutation(self._parent_pop)
                p1 = s_space.pop[int(q[0])]
                p2 = s_space.pop[int(q[1])]

                # a. Perform Crossover
                c1, c2 = self._crossover(p1, p2)
                BaseGrad.evaluate_candidate(c1, s_space, self.valid_bins)
                BaseGrad.evaluate_candidate(c2, s_space, self.valid_bins)

                # b. Perform Mutation
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                BaseGrad.evaluate_candidate(c1, s_space, self.valid_bins)
                BaseGrad.evaluate_candidate(c2, s_space, self.valid_bins)

                # c. Add Offsprings to c_pop
                c_pop.append(c1)
                c_pop.append(c2)

            # Merge, Sort and Select
            s_space.pop += c_pop
            s_space.pop = sorted(s_space.pop, key=lambda x: x.cost)
            s_space.pop = s_space.pop[0:self._parent_pop]

            # Evaluate GP
            _, repeated = BaseGrad.evaluate_gradual_pattern(repeated, s_space, self, ignore_support, target_col, exclude_target)

        for gp in s_space.best_patterns:
            self.add_gradual_pattern(gp)

        duration = time.time() - start
        out_dict: dict[str, str | list] = {
            "Algorithm": "GA-GRAANK",
            # "Memory Usage (MiB)": f{mem_use)}"
            "Initial Population": f"{self._parent_pop}",
            "Children Proportion": f"{self._children_pop}",
            "Crossover Gamma": f"{self._gamma}",
            "Mutation Mu": f"{self._mu}",
            "Mutation Sigma": f"{self._sigma}",
            "Number of iterations": f"{s_space.iter_count}",
            "Run-time": f"{duration:.6f} seconds",
            "Invalid Count": f"{s_space.invalid_count}"}
        return out_dict
