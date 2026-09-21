# SPDX-License-Identifier: MIT
# See the LICENSE file at the root of this
# repository for complete details.


import json

#import torch
import numpy as np

from .base.graank_alg import OrigGRAANK


class GRAANK:
    """
    Mine Gradual Patterns (GPs) from numerical datasets.

    GRAANK is a unified interface for discovering gradual patterns using either
    the classical APRIORI algorithm or several metaheuristic optimization
    algorithms. A gradual pattern (GP) is a set of gradual items (GIs) that
    describe co-variation among numerical attributes.

    A gradual item consists of an attribute and a direction:

    * ``Age+``: attribute values increase
    * ``Salary-``: attribute values decrease

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
        * GRAANK: https://link.springer.com/chapter/10.1007/978-3-642-04957-6_33

        * ACO-GRAANK: https://link.springer.com/article/10.1007/s13042-021-01390-w

        * GA-GRAANK: https://www.sciencedirect.com/science/article/abs/pii/S2210650222001717

        * PSO-GRAANK: https://www.sciencedirect.com/science/article/abs/pii/S2210650222001717

        * Random-GRAANK: https://www.sciencedirect.com/science/article/abs/pii/S2210650222001717

        * Hill Climbing-GRAANK: https://www.sciencedirect.com/science/article/abs/pii/S2210650222001717
    """

    def __init__(self, data_source, min_sup: float = 0.5, eq: bool = False, device: str = "cpu") -> None:
        """
        Initialize a gradual pattern mining session.

        This constructor prepares the mining engine and creates a default
        :class:`OrigGRAANK` instance that performs classical APRIORI-based gradual
        pattern mining. Alternative search algorithms can later be selected by
        calling `discover()` with the appropriate search type.

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

            device:
                The device on which the computation is performed.

                * ``cpu`` (default): will use Numpy operations.

                * ``gpu``: will use GPU (Tensor) operations.

        Attributes:
            mining_engine:
                Active mining engine used by discover()`.

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
            >>> # device = "cuda" if torch.cuda.is_available() else "cpu"
            >>> miner = GRAANK(df, min_sup=0.5, device='cpu')
            >>> result = miner.discover()
        """
        self._data_src = data_source
        self._min_supp: float = min_sup
        self._eq: bool = eq
        self._device = device
        self._mine_obj = OrigGRAANK(data_source, min_sup=min_sup, eq=eq, device=device)

    @property
    def mining_engine(self):
        return self._mine_obj

    def discover(self,
                 search_type: str = "apriori",
                 max_iteration: int | None = None, target_col: int | None = None, exclude_target: bool = False,
                 time_data: dict | None = None,
                 compute_descriptors: bool = True, save_results: bool = False,
                 **kwargs) -> str:
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

            ``clustergp``:
                ClusterGP.

                Uses clustering-based approach to approximate gradual patterns.

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

            time_data:
                Time delay data required to compute the time lag of the gradual patterns. It consists of a dictionary
                with the following keys:

                * ``"time_delay_data"``:
                The time delays after transformation of the data at specific steps. It is either a dictionary or NumPy array.

                * ``"tri_mf"``:
                The triangular membership function used to compute the time lag. It is either None or a NumPy array.

                * ``"use_gp"``:
                Whether to use the gradual items of the GP to compute the time lag. It is a boolean.


            compute_descriptors:
                Compute additional descriptors for each gradual pattern.

            save_results:
                Generate CSV output files after mining.

            **kwargs:
                Additional algorithm-specific hyperparameters.

                These parameters are only used by the corresponding search
                algorithm. Unused parameters are ignored.

                **Genetic Algorithm (`search_type="ga"`):**

                * **n_pop** (*int*, default=5):
                  Population size.

                * **pc** (*float*, default=0.5):
                  Crossover probability.

                * **gamma** (*float*, default=1.0):
                  Crossover expansion factor.

                * **mu** (*float*, default=0.9):
                  Mutation probability.

                * **sigma** (*float*, default=0.9):
                  Mutation step size (standard deviation).

                **Ant Colony Optimization (`search_type="aco"`):**

                * **e_factor** (*float*, default=0.5):
                  Pheromone evaporation factor.

                **Hill Climbing (`search_type="hc"`):**

                * **step_size** (*float*, default=0.5):
                  Neighborhood search step size.

                **Particle Swarm Optimization (`search_type="pso"`):**

                * **n_particle** (*int*, default=5):
                  Number of particles.

                * **vel** (*float*, default=0.9):
                  Initial particle velocity.

                * **coeff_p** (*float*, default=0.01):
                  Cognitive (personal best) acceleration coefficient.

                * **coeff_g** (*float*, default=0.9):
                  Social (global best) acceleration coefficient.

                **ClusterGP (`search_type="clustergp"`):**

                * **e_prob** (*float*, default=0.01):
                Erasure probability.

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
            self._mine_obj = OrigGRAANK(self._data_src, min_sup=self._min_supp, eq=self._eq, device=self._device, max_apriori_level=max_iteration, **kwargs)
        elif search_type == "ga":
            from .base.graank_ga import GeneticGRAANK
            max_iteration = max_iteration if max_iteration is not None else 1
            self._mine_obj = GeneticGRAANK(self._data_src, min_sup=self._min_supp, eq=self._eq, device=self._device, max_iter=max_iteration, **kwargs)
        elif search_type == "aco":
            from .base.graank_aco import AntGRAANK
            max_iteration = max_iteration if max_iteration is not None else 1
            self._mine_obj = AntGRAANK(self._data_src, min_sup=self._min_supp, eq=self._eq, device=self._device, max_iter=max_iteration, **kwargs)
        elif search_type == "pso":
            from .base.graank_pso import ParticleGRAANK
            max_iteration = max_iteration if max_iteration is not None else 1
            self._mine_obj = ParticleGRAANK(self._data_src, min_sup=self._min_supp, eq=self._eq, device=self._device, max_iter=max_iteration, **kwargs)
        elif search_type == "hc":
            from .base.graank_hc import HillClimbingGRAANK
            max_iteration = max_iteration if max_iteration is not None else 1
            self._mine_obj = HillClimbingGRAANK(self._data_src, min_sup=self._min_supp, eq=self._eq, device=self._device, max_iter=max_iteration, **kwargs)
        elif search_type == "random":
            from .base.graank_rand import RandomGRAANK
            max_iteration = max_iteration if max_iteration is not None else 1
            self._mine_obj = RandomGRAANK(self._data_src, min_sup=self._min_supp, eq=self._eq, device=self._device, max_iter=max_iteration)
        elif search_type == "clustergp":
            from .cluster_gp import ClusterGP
            max_iteration = max_iteration if max_iteration is not None else 1
            self._mine_obj = ClusterGP(self._data_src, min_sup=self._min_supp, eq=self._eq, device=self._device, max_iter=max_iteration, **kwargs)
        else:
            raise ValueError("Invalid search type!")

        if isinstance(self._mine_obj, OrigGRAANK):
            res_dict = self._mine_obj.discover(target_col=target_col, time_data=time_data,
                                               exclude_target=exclude_target,
                                               compute_descriptors=compute_descriptors)
        else:
            res_dict = self._mine_obj.discover(target_col=target_col, time_data=time_data,
                                               exclude_target=exclude_target)

        try:
            if save_results:
                self._mine_obj.generate_output_files(res_dict, target_col=target_col)
            res_dict.update({"Patterns": self._mine_obj.display_patterns})
        except Exception as e:
            res_dict.update({"Error": str(e)})
        out:str = json.dumps(res_dict,indent=4,
                             default=lambda obj: obj.item() if isinstance(obj, np.generic) else obj.tolist(),)
        return out

