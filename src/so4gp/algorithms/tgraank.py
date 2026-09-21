# SPDX-License-Identifier: MIT
# See the LICENSE file at the root of this
# repository for complete details.


import json

#import torch
import numpy as np
import pandas

from .base.tgrad import TGrad

#from .. import get_num_cores


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

    def __init__(self, data_source, min_sup: float = 0.5, min_rep: float = 0.5, eq: bool = False,
                 mf_shape: str="triangular", clustering_algorithm: str = "kmeans", inference_method: str = "mamdani", device: str = "cpu"):
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

            min_sup:
                Minimum gradual pattern support threshold.

            min_rep:
                Minimum representativity threshold used during temporal
                transformation.

            eq:
                Whether equal values should be treated as satisfying gradual
                comparisons.

                * ``False`` — (default) strict comparisons.
                * ``True`` — allow equal values.

            mf_shape:
                The shape of the fuzzy membership function to use to estimate time-lag.

                * ``triangular`` — (default) triangular membership function.
                * ``trapezoidal`` — trapezoidal membership function.
                * ``gaussian`` — gaussian membership function.

            clustering_algorithm:
                The clustering algorithm to use to estimate and build the membership functions
                according to the shape selected.

                * ``kmeans`` — (default) the KMeans algorithm.
                * ``fcm`` — The fuzzy-cluster-cmeans algorithm.

            inference_method:
                The inference method to use to evaluate the fuzzy rules to be aggregated and
                defuzzified to estimate the time-delay.

                * ``mamdani`` — (default) The Mamdani inference.
                * ``larsen`` — The Larsen inference.

            device:
                The device on which the computation is performed.

                * ``cpu`` (default) — will use Numpy operations.
                * ``gpu`` — will use GPU (Tensor) operations.

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
            >>> # device = "cuda" if torch.cuda.is_available() else "cpu"
            >>> miner = TGRAANK(df, target_col=1, device='cpu')
            >>> result = miner.discover()
        """
        self._data_src = data_source
        self._min_supp: float = min_sup
        self._min_rep: float = min_rep
        self._eq: bool = eq
        self._mf_shape = mf_shape
        self._clustering_alg = clustering_algorithm
        self._inference_method = inference_method
        self._device = device
        self._mine_obj = TGrad(data_source, min_sup=min_sup, min_rep=min_rep, eq=eq, mf_shape=mf_shape,
                               clustering_algorithm=clustering_algorithm, inference_method=inference_method, add_time=True, device=device)

    @property
    def mining_engine(self):
        return self._mine_obj

    def discover(self, target_col: int, transformations: str = 'ami', transformation_steps: dict | None = None,
                 search_algorithm: str = "apriori", max_iteration: int = 3,
                 eval_mode: bool = False, compute_causality: bool = False, save_results: bool = False, **kwargs) -> str:
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

        The gradual pattern mining algorithm can use the classical APRIORI (GRAANK) algorithm or
        one of several metaheuristic and alternative search strategies.

        Supported search algorithms are:

            ``apriori``:
                Classical GRAANK uses APRIORI level-wise search for exhaustive gradual pattern
                candidate generation.

            ``ga``:
                Genetic GRAANK. Uses a genetic algorithm to search the gradual pattern
                space.

            ``aco``:
                ACO-GRAANK. Uses Ant Colony Optimization and pheromone-guided search
                to identify promising gradual pattern candidates.

            ``pso``:
                PSO-GRAANK. Uses Particle Swarm Optimization to search for high-support
                gradual patterns.

            ``hc``:
                Hill-Climbing GRAANK. Iteratively searches neighboring candidates and
                moves toward patterns with improved support.

            ``random``:
                Random Search GRAANK. Randomly samples and evaluates gradual pattern
                candidates.

            ``clustergp``:
                ClusterGP. Uses clustering-based search to identify gradual patterns
                from the transformed dataset.

        Args:
            target_col:
                [required] Index of the target attribute/feature/column.

                Temporal transformations are estimated relative to this attribute.

            transformations:
                Type of data transformations to be performed.

                Supported values:

                * ``ami`` (recommended)
                * ``all``

            transformation_steps:
                User-defined transformation steps.

                If omitted, all possible transformations are considered.

            search_algorithm:
                Gradual pattern mining algorithm to apply to the transformed
                dataset. Supported values are ``apriori``, ``ga``, ``aco``,
                ``pso``, ``hc``, ``random``, and ``clustergp``.
                Defaults to ``"apriori"``.

            max_iteration:
                The maximum number of iterations to run the search algorithm.

            eval_mode:
                Enables evaluation mode.

                Intended for benchmarking and experimental studies.

            compute_causality:
                Whether to compute causal relations between attributes based on the valid extracted gradual pattern.

                The target column/attribute is taken as the "cause" and the other attributes the "effects".

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
                self._mine_obj = TGrad(self._data_src, min_sup=self._min_supp, min_rep=self._min_rep, eq=self._eq,
                                       mf_shape=self._mf_shape, clustering_algorithm=self._clustering_alg,
                                       inference_method=self._inference_method, add_time=True, device=self._device)
                # num_cores = kwargs.get("num_cores", 1)
                #if num_cores <= 1:
                #    num_cores = get_num_cores()
                ignore_time = kwargs.get("ignore_time", False)
                res_dict = self._mine_obj.discover_tgp(target_col=target_col, search_algorithm=search_algorithm,
                                                       max_iteration=max_iteration, ignore_time=ignore_time)
            elif transformations == 'ami':
                from .base.tgrad_ami import TGradAMI
                self._mine_obj = TGradAMI(self._data_src, min_sup=self._min_supp, min_rep=self._min_rep, eq=self._eq,
                                       mf_shape=self._mf_shape, clustering_algorithm=self._clustering_alg,
                                       inference_method=self._inference_method, add_time=True, device=self._device,)
                res_dict = self._mine_obj.discover_tgp_ami(target_col=target_col,
                                                            transformation_steps=transformation_steps,
                                                            max_iteration=max_iteration,
                                                            search_algorithm = search_algorithm,
                                                            eval_mode=eval_mode, **kwargs)
            else:
                raise ValueError("Invalid transformation algorithm")

            if save_results:
                self._mine_obj.generate_output_files(res_dict, target_col=target_col)
            res_dict.update({"Patterns": self._mine_obj.display_patterns})

            if compute_causality:
                # Causal Inference
                from ..gradual_patterns import TGP
                causal_relations = []
                for tgp in self._mine_obj.gradual_patterns or []:
                    if isinstance(tgp, TGP):
                        res = tgp.get_causal_relations(self._mine_obj.titles)
                        causal_relations.extend(res)
                res_dict.update({"Causality": causal_relations})
        except Exception as e:
            res_dict = {"Error": str(e)}

        out: str = json.dumps(res_dict, indent=4,
                              default=lambda obj: obj.item() if isinstance(obj, np.generic) else obj.tolist(),)
        return out

    def get_lagged_dependencies(self, max_lag: int = 0) -> pandas.DataFrame:
        """
        Compute the lagged dependency matrix between all features.

        Each feature is treated as the target attribute in turn, and temporal
        gradual patterns are mined using the selected transformation algorithm.
        For every discovered causal relationship, the support value is accumulated
        into an adjacency matrix representing the strength of the dependency from
        each cause feature to each effect feature.

        The Genetic GRAANK search is configured with a maximum of three
    optimization iterations for each target feature. This search algorithm is the
    fastest for mining gradual patterns with a large number of features.

        The resulting matrix is returned as a ``pandas.DataFrame`` whose:

        - Rows represent **effect** variables.
        - Columns represent **cause** variables.
        - Cell ``(i, j)`` contains the cumulative support of all temporal gradual
          patterns indicating that feature ``j`` influences feature ``i``.

        Args:
            max_lag:
                Maximum temporal lag (transformation step) considered during
                temporal pattern mining. This value determines the minimum
                representativity threshold used when generating transformed
                datasets.

        Returns:
            pandas.DataFrame:
                A square dependency matrix indexed by feature names, where rows
                correspond to effects and columns correspond to causes.

        Raises:
            ValueError:
                If ``max_lag`` is negative or greater than or equal to the number
                of observations in the dataset.

        Notes:
            Dependency strengths are obtained by summing the support values of
            all temporal gradual patterns supporting the same cause–effect
            relationship.
        """

        if not 0 <= max_lag < self._mine_obj.row_count:
            raise ValueError(f"'max_lag' must be between 0 and {self._mine_obj.row_count - 1}.")

        # Compute minimum representativity from the maximum lag
        if max_lag > 0:
            self._min_rep = (self._mine_obj.row_count - max_lag) / self._mine_obj.row_count

        self._min_supp = 0.005
        feature_cols = self._mine_obj.attr_cols

        # Full adjacency matrix indexed by original column numbers
        n = self._mine_obj.col_count
        dependency_matrix = np.zeros((n, n), dtype=np.float32)

        for target in feature_cols:
            result = json.loads(
                self.discover(target_col=target, transformations="all", search_algorithm='apriori', max_iteration=1, compute_causality=True, ignore_time=True)
            )
            #print(f"Target: {target}\n{result['Causality']}")

            max_corr = {}
            for item in result['Causality']:
                key = tuple(item["correlation"])

                if key not in max_corr or item["support"] > max_corr[key]["support"]:
                    max_corr[key] = item
            computed_relations = list(max_corr.values())
            #print(f"{computed_relations}\n")

            for relation in computed_relations:# result.get("Causality", []):
                cause_col, effect_col = relation["correlation"]
                support = relation["support"]

                if cause_col == target:
                    dependency_matrix[effect_col, cause_col] += support

        # Keep only feature columns (exclude time columns, etc.)
        dependency_matrix = dependency_matrix[np.ix_(feature_cols, feature_cols)]

        feature_titles = [self._mine_obj.titles[i] for i in feature_cols]

        return pandas.DataFrame(
            dependency_matrix,
            index=pandas.Index(feature_titles, name="Effect"),
            columns=pandas.Index(feature_titles, name="Cause"),
        )
