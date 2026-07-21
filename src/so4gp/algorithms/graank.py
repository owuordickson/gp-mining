# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0.
# See the LICENSE file at the root of this
# repository for complete details.


import json
from .base.graank_alg import GRAANKAlg


class GRAANK:

    def __init__(self, data_source, min_sup=0.5, eq=False) -> None:
        """
        Extracts gradual patterns (GPs) from a numeric dataset using the GRAANK algorithm. The algorithm relies on the
        APRIORI approach for generating GP candidates. This work was proposed by Anne Laurent
        and published in: https://link.springer.com/chapter/10.1007/978-3-642-04957-6_33.

             A GP is a set of gradual items (GI), and its quality is measured by its computed support value. For example,
             given a data set with 3 columns (age, salary, cars) and 10 objects. A GP may take the form: {age+, salary-}
             with a support of 0.8. This implies that 8 out of 10 objects have the values of column age 'increasing' and
             column 'salary' decreasing.

        This class extends class DataGP which is responsible for generating the GP bitmaps.

        :param data_source: [required] a data source, it can either be a 'file in csv format' or a 'Pandas DataFrame'
        :type data_source: pd.DataFrame | str

        :param min_sup: [optional] minimum support threshold, the default is 0.5
        :type min_sup: float

        :param eq: [optional] encode equal values as gradual, the default is False
        :type eq: bool

        >>> from so4gp.algorithms import GRAANK
        >>> import pandas
        >>>
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = GRAANKAlg(data_source=dummy_df, min_sup=0.5, eq=False)
        >>> result_json = str(mine_obj.discover())
        >>> # print(result['Patterns'])
        >>> print(result_json) # doctest: +SKIP

        """
        self._data_src = data_source
        self._min_supp: float = min_sup
        self._eq: bool = eq
        self._mine_obj = GRAANKAlg(data_source, min_sup=min_sup, eq=eq)

    @property
    def mine_obj(self):
        return self._mine_obj

    def discover(self,
                 search_type: str = "apriori",
                 ignore_support: bool = False, max_iteration: int | None = None,
                 target_col: int | None = None, exclude_target: bool = False,
                 compute_descriptors: bool = True, save_results: bool = True) -> str:
        """
        Uses a search algorithm to find gradual pattern (GP) candidates. The candidates are validated if their computed
        support is greater than or equal to the minimum support threshold specified by the user. The search algorithm
        can either be: 'apriori', 'genetic algorithm (ga)', 'ant-colony-search (aco)', 'random-search (random)',
        'hill-climbing-search (hc)', 'particle-swarm-search (pso)'

        :param search_type: search algorithm to use [default: 'apriori'].
            Allowed values: ['apriori', 'ga', 'aco', 'random', 'hc', 'pso']
        :param ignore_support: Do not filter extracted GPs using a user-defined minimum support threshold.
        :param max_iteration: Maximum APRIORI/iteration level for generating candidates.
        :param target_col: Target feature's column index.
        :param exclude_target: Only accept GP candidates that do not contain the target feature.
        :param compute_descriptors: [optional] compute descriptors for each GP candidate.
        :param save_results: [optional] Save results to a csv file.

        :return: JSON string
        """

        if search_type == "apriori":
            pass
        elif search_type == "ga":
            from .base.graank_ga import GeneticGRAANK
            max_iteration = max_iteration if max_iteration is not None else 1
            self._mine_obj = GeneticGRAANK(self._data_src, min_sup=self._min_supp, eq=self._eq, max_iter=max_iteration)
        elif search_type == "aco":
            from .base.graank_aco import AntGRAANK
            max_iteration = max_iteration if max_iteration is not None else 1
            self._mine_obj = AntGRAANK(self._data_src, min_sup=self._min_supp, eq=self._eq, max_iter=max_iteration)
        elif search_type == "pso":
            from .base.graank_pso import ParticleGRAANK
            max_iteration = max_iteration if max_iteration is not None else 1
            self._mine_obj = ParticleGRAANK(self._data_src, min_sup=self._min_supp, eq=self._eq, max_iter=max_iteration)
        elif search_type == "hc":
            from .base.graank_hc import HillClimbingGRAANK
            max_iteration = max_iteration if max_iteration is not None else 1
            self._mine_obj = HillClimbingGRAANK(self._data_src, min_sup=self._min_supp, eq=self._eq, max_iter=max_iteration)
        elif search_type == "random":
            from .base.graank_rand import RandomGRAANK
            max_iteration = max_iteration if max_iteration is not None else 1
            self._mine_obj = RandomGRAANK(self._data_src, min_sup=self._min_supp, eq=self._eq, max_iter=max_iteration)

        if self._mine_obj is None:
            raise ValueError("Invalid search type!")

        if isinstance(self._mine_obj, GRAANKAlg):
            res_dict = self._mine_obj.discover(ignore_support=ignore_support, target_col=target_col, exclude_target=exclude_target, apriori_level=max_iteration, compute_descriptors=compute_descriptors)
        else:
            res_dict = self._mine_obj.discover(ignore_support=ignore_support, target_col=target_col, exclude_target=exclude_target)

        try:
            if save_results:
                self._mine_obj.generate_output_files(res_dict, target_col=target_col)
            res_dict.update({"Patterns": self._mine_obj.display_patterns})
        except Exception as e:
            res_dict.update({"Error": str(e)})
        out:str = json.dumps(res_dict,indent=4)
        return out

