# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0.
# See the LICENSE file at the root of this
# repository for complete details.


import json
from .base.graank_alg import GRAANKAlg


class GRAANK:
    """
    Mine Gradual Patterns (GPs) from numerical datasets.

    GRAANK is a unified interface for discovering gradual patterns using either
    the classical APRIORI algorithm or several metaheuristic optimization
    algorithms. A gradual pattern (GP) is a set of gradual items (GIs) that
    describe co-variation among numerical attributes.

    A gradual item consists of an attribute and a direction:

    * ``Age+``  : attribute values increase
    * ``Salary-`` : attribute values decrease

    A gradual pattern is a collection of gradual items, for example::

        {Age+, Salary-}

    with a support of ``0.80`` indicates that approximately 80% of all object
    pairs satisfy the relationship "Age increases while Salary decreases."

    Supported search algorithms:

    * ``apriori`` — Classical exhaustive level-wise search (Laurent et al.).
    * ``ga`` — Genetic Algorithm.
    * ``aco`` — Ant Colony Optimization.
    * ``pso`` — Particle Swarm Optimization.
    * ``hc`` — Hill Climbing.
    * ``random`` — Random Search.

    References:
        Anne Laurent, et al.
        "Mining Gradual Patterns."
        https://link.springer.com/chapter/10.1007/978-3-642-04957-6_33
    """

    def __init__(self, data_source, min_sup: float = 0.5, eq: bool = False) -> None:
        """
        Initialize a gradual pattern mining session.

        This constructor prepares the mining engine and creates a default
        :class:`GRAANKAlg` instance that performs classical APRIORI-based gradual
        pattern mining. Alternative search algorithms can later be selected by
        calling :meth:`discover` with the appropriate search type.

        Args:
            data_source:
                Input dataset.

                Supported inputs include:

                * ``pandas.DataFrame``
                * Path to a CSV file

                The dataset must contain numerical attributes.

            min_sup:
                Minimum support threshold in the interval ``(0, 1]``.

                A candidate gradual pattern is considered frequent only if its
                computed support is greater than or equal to this value.

            eq:
                Whether equal values should be treated as satisfying gradual
                comparisons.

                * ``False`` (default): use strict comparisons
                  (``<`` and ``>``).

                * ``True``: equal values are considered gradual
                  (``<=`` and ``>=``).

        Attributes:
            mine_obj:
                Active mining engine used by :meth:`discover`.

        Raises:
            ValueError:
                If the supplied dataset cannot be loaded.

        Example:
            >>> import pandas as pd
            >>> from so4gp.algorithms import GRAANK
            >>>
            >>> df = pd.DataFrame(
            ...     [
            ...         [30, 3, 1, 10],
            ...         [35, 2, 2, 8],
            ...         [40, 4, 2, 7],
            ...         [50, 1, 1, 6],
            ...         [52, 7, 1, 2],
            ...     ],
            ...     columns=["Age", "Salary", "Cars", "Expenses"],
            ... )
            >>>
            >>> miner = GRAANK(df, min_sup=0.5)
            >>> result = miner.discover()
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
        Discover gradual patterns using the selected search strategy.

        The mining process searches the gradual pattern space using either the
        classical APRIORI algorithm or one of several metaheuristic optimization
        techniques.

        Search Strategies:
            ``apriori``:
                Exhaustive level-wise candidate generation using the APRIORI
                principle.

            ``ga``:
                Genetic Algorithm.

                Candidate gradual patterns are encoded as chromosomes.
                Support determines the fitness (or inverse cost), and evolutionary
                operators are used to search for high-quality patterns.

            ``aco``:
                Ant Colony Optimization.

                Gradual items form pheromone-guided solution paths.
                Valid gradual patterns reinforce pheromone trails, biasing future
                candidate generation toward promising regions of the search space.

            ``pso``:
                Particle Swarm Optimization.

                Each particle represents a gradual pattern candidate.
                Particle positions evolve according to personal and global best
                solutions based on support.

            ``hc``:
                Hill Climbing.

                Starts from an initial solution and repeatedly explores neighboring
                gradual patterns, moving toward candidates with improved support.

            ``random``:
                Random Search.

                Randomly samples gradual pattern candidates and evaluates them
                independently without maintaining search history.

        Args:
            search_type:
                Search algorithm to use.

                Supported values are:

                * ``apriori``
                * ``ga``
                * ``aco``
                * ``pso``
                * ``hc``
                * ``random``

            ignore_support:
                If ``True``, discovered patterns are returned regardless of the
                minimum support threshold.

            max_iteration:
                Maximum number of search iterations.

                For APRIORI, this limits the maximum candidate level.
                For metaheuristics, it specifies the optimization iterations.

            target_col:
                Index of the target attribute.

                When specified, candidate gradual patterns may be filtered to
                either include or exclude this attribute.

            exclude_target:
                Controls target attribute filtering.

                * ``False`` (default):
                  only patterns containing the target attribute are accepted.

                * ``True``:
                  only patterns **not** containing the target attribute are
                  accepted.

            compute_descriptors:
                Compute additional descriptors for each gradual pattern.

            save_results:
                Generate CSV output files after mining.

        Returns:
            JSON-formatted string containing the mining results, including the
            discovered gradual patterns, statistics, descriptors, and metadata.

        Raises:
            ValueError:
                If an unsupported search algorithm is requested.

        Notes:
            Metaheuristic algorithms provide approximate solutions and are often
            substantially faster than exhaustive APRIORI search on high-dimensional
            datasets, although they do not guarantee complete enumeration of all
            frequent gradual patterns.
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

