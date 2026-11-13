# SPDX-License-Identifier: MIT
# See the LICENSE file at the root of this
# repository for complete details.


import json
from itertools import combinations, islice, product
from numbers import Integral

# import torch
import numpy as np
import pandas as pds

from ..gradual_patterns import GP, PairwiseMatrix
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

    def __init__(
        self, data_source, min_sup: float = 0.5, eq: bool = False, device: str = "cpu"
    ) -> None:
        """
        Initialize a gradual pattern mining session.

        This constructor prepares the mining engine and creates a default
        ``OrigGRAANK`` instance that performs classical APRIORI-based gradual
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
                Active mining engine used by ``discover()``.

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

    def discover(
        self,
        search_type: str = "apriori",
        max_iteration: int | None = None,
        target_col: int | None = None,
        exclude_target: bool = False,
        time_data: dict | None = None,
        compute_descriptors: bool = True,
        save_results: bool = False,
        **kwargs,
    ) -> str:
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
                Time delay data required to compute the time lag of the gradual
                patterns. It consists of a dictionary
                with the following keys:

                * ``"time_delay_data"``:
                  The time delays after transformation of the data at specific steps.
                  It is either a dictionary or NumPy array.

                * ``"tri_mf"``:
                  The triangular membership function used to compute the time lag.
                  It is either None or a NumPy array.

                * ``"use_gp"``:
                  Whether to use the gradual items of the GP to compute the time lag.
                  It is a boolean.


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
            self._mine_obj = OrigGRAANK(
                self._data_src,
                min_sup=self._min_supp,
                eq=self._eq,
                device=self._device,
                max_apriori_level=max_iteration,
                **kwargs,
            )
        elif search_type == "ga":
            from .base.graank_ga import GeneticGRAANK

            max_iteration = max_iteration if max_iteration is not None else 1
            self._mine_obj = GeneticGRAANK(
                self._data_src,
                min_sup=self._min_supp,
                eq=self._eq,
                device=self._device,
                max_iter=max_iteration,
                **kwargs,
            )
        elif search_type == "aco":
            from .base.graank_aco import AntGRAANK

            max_iteration = max_iteration if max_iteration is not None else 1
            self._mine_obj = AntGRAANK(
                self._data_src,
                min_sup=self._min_supp,
                eq=self._eq,
                device=self._device,
                max_iter=max_iteration,
                **kwargs,
            )
        elif search_type == "pso":
            from .base.graank_pso import ParticleGRAANK

            max_iteration = max_iteration if max_iteration is not None else 1
            self._mine_obj = ParticleGRAANK(
                self._data_src,
                min_sup=self._min_supp,
                eq=self._eq,
                device=self._device,
                max_iter=max_iteration,
                **kwargs,
            )
        elif search_type == "hc":
            from .base.graank_hc import HillClimbingGRAANK

            max_iteration = max_iteration if max_iteration is not None else 1
            self._mine_obj = HillClimbingGRAANK(
                self._data_src,
                min_sup=self._min_supp,
                eq=self._eq,
                device=self._device,
                max_iter=max_iteration,
                **kwargs,
            )
        elif search_type == "random":
            from .base.graank_rand import RandomGRAANK

            max_iteration = max_iteration if max_iteration is not None else 1
            self._mine_obj = RandomGRAANK(
                self._data_src,
                min_sup=self._min_supp,
                eq=self._eq,
                device=self._device,
                max_iter=max_iteration,
            )
        elif search_type == "clustergp":
            from .cluster_gp import ClusterGP

            max_iteration = max_iteration if max_iteration is not None else 1
            self._mine_obj = ClusterGP(
                self._data_src,
                min_sup=self._min_supp,
                eq=self._eq,
                device=self._device,
                max_iter=max_iteration,
                **kwargs,
            )
        else:
            raise ValueError("Invalid search type!")

        if isinstance(self._mine_obj, OrigGRAANK):
            res_dict = self._mine_obj.discover(
                target_col=target_col,
                time_data=time_data,
                exclude_target=exclude_target,
                compute_descriptors=compute_descriptors,
            )
        else:
            res_dict = self._mine_obj.discover(
                target_col=target_col,
                time_data=time_data,
                exclude_target=exclude_target,
            )

        try:
            if save_results:
                self._mine_obj.generate_output_files(res_dict, target_col=target_col)
            res_dict.update({"Patterns": self._mine_obj.display_patterns})
        except RuntimeError as e:
            res_dict.update({"Error": str(e)})
        out: str = json.dumps(
            res_dict,
            indent=4,
            default=lambda obj: (
                obj.item() if isinstance(obj, np.generic) else obj.tolist()
            ),
        )
        return out

    @classmethod
    def build_gradual_features(
        cls,
        sequences: np.ndarray,
        max_patterns: int | None = None,
        pattern_level: int = 2,
    ) -> pds.DataFrame:
        """Encode numerical arrays using support and GCS descriptors.

        For a two-dimensional array, return one row per pattern, indexed by
        zero-based column identifiers such as ``"0+"`` or ``"0+ 1-"``.
        The six columns, in order, are ``support``, ``density``,
        ``connectivity``, ``avg_separation``, ``rank_dispersion``, and
        ``singularity``. Values come directly from SO4GP's fitted bitmaps,
        packed intersections, and descriptor calculations without display
        rounding.

        For a three-dimensional array, return one row per recording in input
        order. Columns are a two-level pandas MultiIndex named ``pattern``
        and ``descriptor``. Each pattern contributes the same six descriptors.
        Comparisons stay within each recording, and pattern identities and
        column order are identical for inputs with the same channel count,
        pattern level, and cap. No labels are included.

        A level of one keeps both signs of each GI. Higher levels enumerate
        exactly that many distinct attributes, retaining one orientation
        from each globally reversed pair: the smallest column always has
        ``+`` direction. Columns are enumerated numerically, with ``+``
        before ``-``. This fixes the directional GCS orientation explicitly.

        All canonical combinations are encoded, including zero-support
        patterns; no minimum-support filter is applied. Undefined separation,
        dispersion, and singularity for an empty GCS remain NaN in the
        two-dimensional descriptive table. In three-dimensional classifier
        features they become zero, while connectivity still counts every
        isolated observation.

        Args:
            sequences: Real numerical array shaped
                ``(observations, channels)`` or
                ``(recordings, observations, channels)``. Require at least
                two observations, one channel, and one recording.
            max_patterns: Positive maximum number of patterns encoded per
                recording, or None to encode all canonical combinations.
                The cap takes a deterministic prefix. Values larger than
                the available pattern count do not introduce padding.
            pattern_level: Exact number of gradual items per pattern,
                between one and the channel count. Defaults to two.

        Returns:
            A pandas DataFrame containing the six descriptors per pattern,
            either as pattern rows (2D input) or pattern/descriptor columns
            (3D input).

        Raises:
            ValueError: Input dimensions, values, pattern level, or cap are
                invalid, or a concordance set cannot be described.

        Notes:
            The method fits packed GI bitmaps once per recording and only
            for attributes used by the selected patterns. It combines them
            using GP.perform_and and delegates structural measures to
            GP.compute_descriptors. Limiting patterns stops the combination
            enumeration early rather than building all possible combinations
            first.

        Examples:
            >>> data = np.array([[1., 1.], [2., 2.], [3., 3.]])
            >>> table = GRAANK.build_gradual_features(data)
            >>> table.index.tolist()
            ['0+ 1+', '0+ 1-']
            >>> features = GRAANK.build_gradual_features(np.stack((data, data)))
            >>> features.shape
            (2, 12)
        """
        data = np.asarray(sequences)
        if data.ndim not in (2, 3) or data.dtype.kind not in "biuf":
            raise ValueError("sequences must be a real numerical 2D or 3D array")
        recordings = data[None] if data.ndim == 2 else data
        num_recordings, num_observations, num_channels = recordings.shape
        if num_recordings == 0 or num_observations < 2 or num_channels < 1:
            raise ValueError(
                "Require a recording with at least two observations and one channel"
            )
        if not np.isfinite(data).all():
            raise ValueError("sequences must contain only finite values")
        if (
            isinstance(pattern_level, bool)
            or not isinstance(pattern_level, Integral)
            or not 1 <= pattern_level <= num_channels
        ):
            raise ValueError(
                "pattern_level must be an integer from 1 to the channel count"
            )
        if max_patterns is not None and (
            isinstance(max_patterns, bool)
            or not isinstance(max_patterns, Integral)
            or max_patterns < 1
        ):
            raise ValueError("max_patterns must be a positive integer or None")
        level = int(pattern_level)
        limit = None if max_patterns is None else int(max_patterns)

        def pattern_items():
            """Yield exact-size patterns with deterministic numeric identifiers."""
            if level == 1:
                for column in range(num_channels):
                    for symbol in ("+", "-"):
                        yield ((column, symbol),)
            else:
                for columns in combinations(range(num_channels), level):
                    for suffix in product(("+", "-"), repeat=level - 1):
                        yield tuple(zip(columns, ("+", *suffix), strict=True))

        patterns = list(islice(pattern_items(), limit))
        names = [" ".join(f"{column}{sign}" for column, sign in p) for p in patterns]
        needed_columns = sorted({column for p in patterns for column, _ in p})
        descriptors = (
            "support",
            "density",
            "connectivity",
            "avg_separation",
            "rank_dispersion",
            "singularity",
        )

        def encode_recording(recording: np.ndarray) -> np.ndarray:
            """Fit used GI bitmaps once and intersect each requested pattern."""
            values = recording[:, needed_columns]
            source = pds.DataFrame(
                values, columns=[f"channel_{i}" for i in needed_columns]
            )
            engine = cls(source, min_sup=0.0).mining_engine
            if len(needed_columns) == 1:
                # fit_bitmap requires two attributes for mining. GI feature
                # extraction also supports a single attribute.
                bitmap = engine._compute_pairwise_bitmap(
                    np.asarray(values[:, 0], dtype=np.float64)
                )
                forward = sum(
                    np.count_nonzero(row[i + 1 :]) for i, row in enumerate(bitmap)
                )
                reverse = sum(np.count_nonzero(row[:i]) for i, row in enumerate(bitmap))
                support = float(max(forward, reverse) / GP.pair_count(num_observations))
                bins = {
                    "0+": PairwiseMatrix(np.packbits(bitmap.ravel()), support, {"0+"}),
                    "0-": PairwiseMatrix(
                        np.packbits(bitmap.T.ravel()), support, {"0-"}
                    ),
                }
            else:
                engine.fit_bitmap()
                bins = engine.valid_bins or {}
            lookup = {
                (original, sign): bins[f"{position}{sign}"]
                for position, original in enumerate(needed_columns)
                for sign in ("+", "-")
            }
            result = np.empty((len(patterns), len(descriptors)), dtype=float)
            for index, pattern in enumerate(patterns):
                bitmap = lookup[pattern[0]]
                for item in pattern[1:]:
                    bitmap = GP.perform_and(bitmap, lookup[item], num_observations)
                    if bitmap.support == 0:
                        break
                pairs = GP.gen_gradual_concordance_set(
                    bitmap.packed_bin_mat, num_observations
                )
                gp = GP()
                if not gp.compute_descriptors(pairs, num_observations):
                    raise ValueError(f"Invalid concordance set for {names[index]}")
                result[index] = (
                    bitmap.support,
                    gp.density,
                    gp.graph_connectivity,
                    gp.average_separation,
                    gp.rank_dispersion,
                    gp.singularity_score,
                )
            return result

        if data.ndim == 2:
            return pds.DataFrame(
                encode_recording(data),
                index=pds.Index(names, name="pattern"),
                columns=pds.Index(descriptors, name="descriptor"),
            )
        features = np.empty(
            (num_recordings, len(patterns) * len(descriptors)), dtype=float
        )
        for index, recording in enumerate(recordings):
            values = encode_recording(recording)
            features[index] = np.nan_to_num(values, nan=0.0, copy=False).ravel()
        return pds.DataFrame(
            features,
            index=pds.RangeIndex(num_recordings, name="recording"),
            columns=pds.MultiIndex.from_product(
                (names, descriptors), names=("pattern", "descriptor")
            ),
        )
