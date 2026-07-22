# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0.
# See the LICENSE file at the root of this
# repository for complete details.


import json
from .base.tgrad import TGrad

class TGRAANK:
    """
    Mine Temporal Gradual Patterns (TGPs) from time-series datasets.

    TGRAANK discovers fuzzy temporal gradual patterns by combining data
    transformation, fuzzy logic, and gradual pattern mining. Unlike classical
    gradual pattern mining, TGRAANK estimates temporal delays between
    observations before extracting gradual relationships.

    The framework supports both the original TGrad algorithm and an improved
    Mutual Information (AMI)-based transformation algorithm.

    Supported transformation algorithms:

    * ``all`` — Classical TGrad fuzzy temporal mining.
    * ``ami`` — Mutual Information-based temporal transformation (recommended).

    References:
        * TGrad:
          https://ieeexplore.ieee.org/abstract/document/8858883

        * TGradAMI:
          https://ieeexplore.ieee.org/abstract/document/11197674/
    """

    def __init__(self, data_source, target_col: int, min_sup: float = 0.5, min_rep: float = 0.5, eq: bool = False):
        """
        Initialize a temporal gradual pattern miner.

        This constructor creates a default TGrad mining engine for discovering
        fuzzy temporal gradual patterns. Alternative temporal transformation
        algorithms can later be selected by calling `discover()`.

        Args:
            data_source:
                Input dataset.

                Supported inputs include:

                * ``pandas.DataFrame``
                * Path to a CSV file

                The first column typically contains timestamps, while the remaining
                columns contain numerical attributes.

            target_col:
                Zero-based index of the target attribute.

                Temporal transformations are estimated relative to this attribute.

            min_sup:
                Minimum gradual pattern support threshold.

            min_rep:
                Minimum representativity threshold used during temporal
                transformation.

            eq:
                Whether equal values should be treated as satisfying gradual
                comparisons.

                * ``False`` — strict comparisons.
                * ``True`` — allow equal values.

        Attributes:
            mining_engine:
                Active temporal mining engine.

        Example:
            >>> import pandas as pd
            >>> from so4gp.algorithms import TGRAANK
            >>>
            >>> df = pd.DataFrame(
            ...     [
            ...         ["2021-03",30,3,1,10],
            ...         ["2021-04",35,2,2,8],
            ...         ["2021-05",40,4,2,7],
            ...         ["2021-06",50,1,1,6],
            ...         ["2021-07",52,7,1,2],
            ...     ],
            ...     columns=["Date","Age","Salary","Cars","Expenses"]
            ... )
            >>>
            >>> miner = TGRAANK(df, target_col=1)
            >>> result = miner.discover()
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

    def discover(self, transformations: str= 'ami', transformation_steps: dict | None = None,
                 eval_mode: bool = False, save_results: bool = True, **kwargs) -> str:
        """
        Discover fuzzy temporal gradual patterns.

        The selected transformation algorithm first estimates temporal delays
        between observations and then performs gradual pattern mining on the
        transformed dataset.

        Data Transformations:
            ``all``:
                Classical TGRAANK algorithm generating all possible transformations.

                Performs a full transformation of the dataset yielding all possible transformations up to the maximum
                transformation step specified by `min_rep`. Then it applies fuzzy membership functions to estimate
                temporal lags before mining temporal gradual patterns.

            ``ami``:
                TGRAANK-AMI algorithm with improved AMI-based data transformation.

                Extends TGrad by estimating temporal delays using Average Mutual
                Information (AMI). Candidate transformations are evaluated by
                comparing their mutual information with that of the original
                dataset. The transformation whose mutual information differs by at
                most `error_margin` is selected as the optimal delay.

                Optionally, clustering can be used to reduce the number of
                candidate transformations that must be evaluated.

        Args:
            transformations:
                Type of data transformations to be performed.

                Supported values:

                * ``ami`` (recommended)
                * ``all``

            transformation_steps:
                User-defined transformation steps.

                If omitted, all possible transformations are considered.

            eval_mode:
                Enables evaluation mode.

                Intended for benchmarking and experimental studies.

            save_results:
                Whether to generate CSV output files.

            **kwargs:
                Additional transformation-specific hyperparameters.

                These parameters are only used by the corresponding data
                transformation. Unused parameters are ignored.

                **TGRAANK (`transformations="all"`):**

                * **num_cores** (*int*, default=1):
                  Number of CPU cores to be used for multiprocessing.

                **TGRAANK-AMI (`transformations="ami"`):**

                * **use_clustering** (*bool*, default=False):
                  Use a clustering algorithm (KMeans) to estimate the best time-delay value.

                * **error_margin** (*float*, default=0.0001):
                  Maximum acceptable mutual information difference between the transformed and original datasets.

        Returns:
            JSON-formatted string containing the discovered temporal gradual
            patterns together with estimated time delays, support values, and
            additional metadata.

        Raises:
            ValueError:
                If an unsupported transformation algorithm is requested.

        Notes:
            The classical TGrad algorithm estimates temporal delays using fuzzy
            representativity.

            The AMI algorithm estimates delays by preserving the mutual
            information between the transformed dataset and the original target
            attribute, typically producing more accurate temporal transformations.
        """

        try:

            if transformations == 'all':
                res_dict = self._mine_obj.discover_tgp(**kwargs)
            elif transformations == 'ami':
                from .base.tgrad_ami import TGradAMI
                self._mine_obj = TGradAMI(self._data_src, target_col=self._target_col, min_sup=self._min_supp, min_rep=self._min_rep, eq=self._eq)
                res_dict = self._mine_obj.discover_tgp(transformation_steps=transformation_steps, eval_mode=eval_mode, **kwargs)
            else:
                raise ValueError("Invalid transformation algorithm")

            if save_results:
                self._mine_obj.generate_output_files(res_dict, target_col=self._target_col)
            res_dict.update({"Patterns": self._mine_obj.display_patterns})
        except Exception as e:
            res_dict = {"Error": str(e)}

        out:str = json.dumps(res_dict, indent=4)
        return out
