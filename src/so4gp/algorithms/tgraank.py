# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0.
# See the LICENSE file at the root of this
# repository for complete details.


import json
from .base.tgrad import TGrad

class TGRAANK:
    """
    TGRAANK is an algorithm used to extract temporal gradual patterns from numeric datasets. An algorithm for mining
        temporal gradual patterns using fuzzy membership functions. It uses a technique
        published in: https://ieeexplore.ieee.org/abstract/document/8858883.
    """

    def __init__(self, data_source, target_col: int, min_sup: float = 0.5, min_rep: float = 0.5, eq: bool = False):
        """

        >>> import so4gp.algorithms import TGRAANK
        >>> import pandas
        >>>
        >>> dummy_data = [["2021-03", 30, 3, 1, 10], ["2021-04", 35, 2, 2, 8], ["2021-05", 40, 4, 2, 7], ["2021-06", 50, 1, 1, 6], ["2021-07", 52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Date', 'Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = TGRAANK(dummy_df, min_sup=0.5, target_col=1, min_rep=0.5)
        >>> result_json = mine_obj.discover()
        >>> # print(result['Patterns'])
        >>> print(result_json)
        """
        self._data_src = data_source
        self._target_col: int = target_col
        self._min_supp: float = min_sup
        self._min_rep: float = min_rep
        self._eq: bool = eq
        self._mine_obj = TGrad(data_source, target_col=target_col, min_sup=min_sup, min_rep=min_rep, eq=eq)

    @property
    def mining_engine(self):
        return self._mine_obj

    def discover(self, transformation_algorithm: str='ami', transformation_steps: dict|None = None,
                 use_clustering: bool = False, error_margin: float = 0.0001,
                 num_cores: int = 1,
                 eval_mode: bool = False, save_results: bool = True) -> str:
        """"""

        if transformation_algorithm == 'all':
            pass
        elif transformation_algorithm == 'ami':
            from .base.tgrad_ami import TGradAMI
            self._mine_obj = TGradAMI(self._data_src, target_col=self._target_col, min_sup=self._min_supp, min_rep=self._min_rep, eq=self._eq)
        else:
            raise ValueError("Invalid transformation algorithm")

        if isinstance(self._mine_obj, TGrad):
            res_dict = self._mine_obj.discover_tgp(num_cores=num_cores)
        else:
            res_dict = self._mine_obj.discover_tgp(use_clustering=use_clustering, transformation_steps=transformation_steps, error_margin=error_margin, eval_mode=eval_mode)

        try:
            if save_results:
                self._mine_obj.generate_output_files(res_dict, target_col=self._target_col)
            res_dict.update({"Patterns": self._mine_obj.display_patterns})
        except Exception as e:
            res_dict.update({"Error": str(e)})
        out:str = json.dumps(res_dict,indent=4)
        return out
