# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0.
# See the LICENSE file at the root of this
# repository for complete details.


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
        >>> mine_obj = GRAANK(data_source=dummy_df, min_sup=0.5, eq=False)
        >>> result_json = str(mine_obj.discover())
        >>> # print(result['Patterns'])
        >>> print(result_json) # doctest: +SKIP

        """
        self._data_src = data_source
        self._min_supp: float = min_sup
        self._eq: bool = eq

    def discover(self, ignore_support: bool = False, apriori_level: int | None = None,
                 target_col: int | None = None, exclude_target: bool = False, compute_descriptors: bool = True,
                 save_results: bool = True) -> str:
        """
        Uses apriori algorithm to find gradual pattern (GP) candidates. The candidates are validated if their computed
        support is greater than or equal to the minimum support threshold specified by the user.

        :param ignore_support: Do not filter extracted GPs using a user-defined minimum support threshold.
        :param apriori_level: Maximum APRIORI level for generating candidates.
        :param target_col: Target feature's column index.
        :param exclude_target: Only accept GP candidates that do not contain the target feature.
        :param compute_descriptors: [optional] compute descriptors for each GP candidate.
        :param save_results: [optional] Save results to a csv file.

        :return: JSON string
        """
        pass