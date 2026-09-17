# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# See the LICENSE file at the root of this
# repository for complete details.
import copy
import random
import numpy as np
from dataclasses import dataclass
from ...data_gp import DataGP
from ...gradual_patterns import GI, GP, TGP


class BaseGrad(DataGP):

    @dataclass
    class Candidate:
        """
        Represent a candidate solution in a gradual pattern search space.

        A candidate associates a gradual pattern with its position and
        corresponding cost in the search space. The class is used by
        optimization-based gradual pattern mining algorithms to represent
        and evaluate potential solutions.

        A candidate may represent either a regular gradual pattern (``GP``)
        or a temporal gradual pattern (``TGP``). The pattern, position, and
        cost can initially be undefined and are represented by ``None`` until
        they are assigned during the search process.

        Attributes:
            gp:
                Gradual pattern represented by the candidate. This can be a
                :class:`GP`, a :class:`TGP`, or ``None`` if the candidate has
                not yet been associated with a pattern.

            position:
                Numeric position of the candidate in the search space.
                ``None`` indicates that a position has not yet been assigned.

            cost:
                Objective or cost value associated with the candidate.
                ``None`` indicates that the candidate has not yet been
                evaluated.

        Notes:
            The interpretation of ``position`` and ``cost`` depends on the
            optimization algorithm using the candidate. For example,
            algorithms that maximize gradual-pattern support may represent
            higher-quality solutions using lower costs.
        """
        gp: GP|TGP|None=None
        position: float|None=None
        cost: float|None=None

    @dataclass
    class SearchSpace:
        """
            Represent the state of a gradual pattern optimization search space.

            This class stores the configuration, execution statistics, candidate
            solutions, and search results maintained during an optimization-based
            gradual pattern mining process. It provides a common representation of
            the search state used by algorithms such as Genetic GRAANK, ACO-GRAANK,
            PSO-GRAANK, Hill-Climbing GRAANK, and Random GRAANK.

            Attributes:
                var_min:
                    Minimum valid value or boundary of the search-space variable.

                var_max:
                    Maximum valid value or boundary of the search-space variable.

                iter_count:
                    Number of search iterations completed.

                eval_count:
                    Number of candidate solutions evaluated during the search.

                invalid_count:
                    Number of candidate solutions rejected because they do not
                    satisfy the constraints of the search space or mining process.

                best_candidate:
                    Candidate solution with the best objective value found so far.

                loser_gps:
                    Gradual patterns that were evaluated but did not qualify as
                    selected or competitive solutions during the search. Elements
                    may be either regular gradual patterns (``GP``) or temporal
                    gradual patterns (``TGP``).

                pop:
                    Current population of candidate solutions maintained by the
                    optimization algorithm. Each element is a
                    :class:`BaseGrad.Candidate`.

            Notes:
                The interpretation of ``var_min`` and ``var_max`` depends on the
                optimization algorithm. Likewise, the definition of a "best"
                candidate depends on the objective function used by the algorithm.
                For example, some algorithms minimize cost while others maximize
                gradual-pattern support.
        """
        var_min: int
        var_max: int
        iter_count: int
        eval_count: int
        invalid_count: int
        best_candidate: "BaseGrad.Candidate"
        loser_gps: list[GP|TGP]
        pop: list["BaseGrad.Candidate"]

    def __init__(self, *args, **kwargs):
        """
            Initialize the base gradual-pattern optimization algorithm.

            The constructor delegates dataset and gradual-pattern mining
            initialization to :class:`DataGP` and initializes the state required
            by optimization-based search algorithms.

            Args:
                *args:
                    Positional arguments forwarded to :class:`DataGP`.

                **kwargs:
                    Keyword arguments forwarded to :class:`DataGP`. These typically
                    include the data source and mining parameters such as minimum
                    support and equality handling.

            Attributes:
                _target_col:
                    Index of the target attribute used during target-oriented
                    gradual pattern mining. Initialized to ``None`` until a target
                    attribute is specified.

                _search_space:
                    Current optimization search-space state. Initialized to
                    ``None`` and created when an optimization-based mining
                    algorithm starts its search.

            Notes:
                This constructor is intended to initialize shared state for
                subclasses implementing optimization-based gradual pattern
                mining strategies such as Genetic GRAANK, ACO-GRAANK,
                PSO-GRAANK, Hill-Climbing GRAANK, and Random GRAANK.
        """
        super(BaseGrad, self).__init__(*args, **kwargs)
        self._target_col: int | None = None
        self._search_space: "BaseGrad.SearchSpace|None" = None

    @property
    def target_col(self):
        return self._target_col

    @target_col.setter
    def target_col(self, tgt_col: int):
        if tgt_col in self.time_cols:
            msg = "Target column should not be a 'date-time' attribute"
            raise ValueError(msg)
        elif (tgt_col < 0) or (tgt_col >= self.col_count):
            msg = "Target column does not exist\nselect a column value between: " \
                      "0 and " + str(self.col_count - 1)
            raise ValueError(msg)
        self._target_col = tgt_col

    @property
    def search_space(self) -> "BaseGrad.SearchSpace|None":
        return self._search_space

    def blank_search_space(self) -> "BaseGrad.SearchSpace|None":
        """Create a blank search space."""
        try:
            self.init_search_space(1)
            s_space = self.search_space
            return s_space
        except ValueError:
            return None

    def init_search_space(self, pop_size: int) -> bool:
        """
        Initialize the search space with pairwise matrices
        :param pop_size: population size

        :return: Search space or error message
        """
        # Prepare data set
        self.fit_bitmap()
        self.clear_gradual_patterns()
        if self.valid_bins is None:
            raise ValueError("Pairwise matrices not available!")

        if pop_size == 0:
            raise ValueError("Population size is zero!")

        # Initialize search space
        self._search_space = self._initialize_numeric_search_space(pop_size)
        if self._search_space is None:
            return False
        return True

    def _initialize_numeric_search_space(self, total_pop: int):
        """Create a population of candidate solutions."""
        valid_bins_dict = self.valid_bins
        if valid_bins_dict is None:
            return None

        attr_keys = list(valid_bins_dict.keys())

        # Empty Individual Template
        empty_candidate = BaseGrad.Candidate (
            position=None,
            cost=None
        )

        # Initialize Population
        var_min = 0
        var_max = int(''.join(['1'] * len(attr_keys)), 2)
        pop = [empty_candidate] * total_pop
        for i in range(total_pop):
            pop[i].position = random.randrange(var_min, var_max)
            pop[i].cost = 1

        # Initialize best candidate
        best_candidate = BaseGrad.Candidate(
            position=pop[0].position,
            cost = None
        )

        # Initialize SearchSpace parameters
        search_space = BaseGrad.SearchSpace(
            iter_count=0,
            eval_count=0,
            invalid_count=0,
            var_min=var_min,
            var_max=var_max,
            best_candidate=best_candidate,
            loser_gps=[],
            pop=pop,
        )
        return search_space

    def _cost_function(self, candidate: "BaseGrad.Candidate|None", exclude_target: bool = False, time_data: dict|None= None) -> bool:
        """Description

        Computes the fitness of a GP

        :param candidate: a candidate GP in the search space
        :param exclude_target: (optional) if True, candidates containing the target feature are rejected
        :param time_data: (optional) time data for estimating time lag
        """

        def _decode_gp(position: float | None) -> GP:
            """Description

            Decodes a numeric value (position) into a GP

            :param position: a value in the numeric search space
            :return: GP that is decoded from the position value
            """

            temp_gp: GP = GP()
            if position is None or valid_bins_dict is None:
                return temp_gp

            attr_keys = list(valid_bins_dict.keys())
            bin_str = bin(int(position))[2:]
            bin_arr = np.array(list(bin_str), dtype=int)

            for i in range(bin_arr.size):
                bin_val = bin_arr[i]
                if bin_val == 1:
                    temp_gi = GI.from_string(attr_keys[i])
                    #if not temp_gp.contains_attr(temp_gi):
                    temp_gp.add_gradual_item(temp_gi)
            return temp_gp

        s_space = self.search_space
        target_col = self.target_col
        valid_bins_dict = self.valid_bins
        if valid_bins_dict is None or candidate is None or s_space is None:
            return False

        candidate.cost = 1
        if candidate.position is None:
            return False

        # 1. Decode candidate position into GP
        rand_gp = _decode_gp(candidate.position)

        # 2. Check is target-column is present in the GP
        target_col_ok = self.check_target_feature(rand_gp, exclude_target=exclude_target)
        if not target_col_ok:
            return False

        # 3. Check if the GP is a duplicate candidate
        exists = rand_gp.is_duplicate(self.gradual_patterns, s_space.loser_gps)
        if exists:
            return False
        if not exists:
            # check for anti-monotony
            is_super = rand_gp.check_am(s_space.loser_gps, subset=False)
            is_sub = rand_gp.check_am(self.gradual_patterns, subset=True)
            if is_super or is_sub:
                return False

        # 4. validate the GP
        gen_gp: GP|TGP = rand_gp.validate_via_graank(self, target_col=target_col, time_data=time_data)

        # 5. Compute the cost of the GP
        candidate.cost = (1.0 - gen_gp.support) ** 2  # penalize low-support patterns more strongly
        candidate.gp = gen_gp
        return True

    def evaluate_candidate(self, candidate: "BaseGrad.Candidate|None", exclude_target: bool, time_data: dict|None= None):
        """
        Evaluate a gradual-pattern candidate against the mining constraints.

        The candidate is evaluated to determine whether it represents a valid
        gradual pattern for the current search. When target-oriented mining is
        enabled, the candidate can be required to either contain or exclude the
        target attribute according to ``exclude_target``.

        Optional temporal data can be supplied when evaluating candidates that
        require time-dependent information.

        Args:
            candidate:
                Candidate solution to evaluate. ``None`` indicates that there is
                no candidate available for evaluation.

            exclude_target:
                Determines how the target attribute is handled during evaluation.
                When ``True``, candidates containing the target attribute are
                excluded. When ``False``, target-oriented evaluation is applied
                according to the configured target attribute.

            time_data:
                Optional temporal data required for evaluating time-dependent
                candidates. The dictionary structure depends on the temporal
                mining algorithm.

        Returns:
            The result of evaluating the candidate. The returned value depends on
            the candidate evaluation strategy implemented by the subclass.

        Notes:
            This method is intended to be used by optimization-based gradual
            pattern mining algorithms to validate and evaluate candidate
            solutions during the search process.
        """

        s_space = self.search_space
        if candidate is None or s_space is None or self.valid_bins is None:
            return s_space

        def apply_bound() -> None:
            """
            Modifies x (a numeric value) if it exceeds the lower/upper bound of the numeric search space.
            :return: None
            """
            candidate.position = float(np.maximum(candidate.position if candidate else 0, s_space.var_min if s_space else 0))
            candidate.position = float(np.minimum(candidate.position if candidate else 0, s_space.var_max if s_space else 0))

        apply_bound()
        # Update: What about duplicate candidate (position already exists in the search-space)?
        
        self._cost_function(candidate, exclude_target=exclude_target, time_data=time_data)
        if candidate.cost == 1:
            s_space.invalid_count += 1
        if candidate.cost is not None:
            # 1. Check if the candidate is better than the current best candidate
            if s_space.best_candidate.cost is None:
                s_space.best_candidate = copy.deepcopy(candidate)
            elif candidate.cost < s_space.best_candidate.cost:
                s_space.best_candidate = copy.deepcopy(candidate)

            # 2. Check if it is a valid GP and is NOT a duplicate candidate
            if candidate.gp is not None:
                gen_gp = candidate.gp
                length_ok = (len(gen_gp.gradual_items) > 1)
                if (gen_gp.support >= self.thd_supp) and length_ok:
                    is_present = gen_gp.is_duplicate(self.gradual_patterns)
                    is_sub = gen_gp.check_am(self.gradual_patterns, subset=True)
                    if not is_present and not is_sub:
                        self.add_gradual_pattern(gen_gp)
                else:
                    s_space.invalid_count += 1
                    s_space.loser_gps.append(gen_gp)
        s_space.eval_count += 1
        return s_space

    def check_target_feature(self, gp_cand: set | GP, exclude_target: bool = False):
        """
        Applies the target-feature constraint to a gradual pattern candidate.

        Parameters
        ----------
        gp_cand : set
            Candidate gradual pattern.
        exclude_target : bool, default=False
            If True, candidates containing the target feature are rejected.
            If False, candidates must contain the target feature.

        Returns
        -------
        bool
            True if the candidate passes the target-feature constraint,
            otherwise False.
        """

        target_col = self.target_col
        if target_col is None:
            return True

        has_target = True
        if isinstance(gp_cand, set):
            has_target = np.any(
                np.array(
                    [GI.from_string(gi_str).attribute_col == target_col for gi_str in gp_cand],
                    dtype=bool,
                )
            )
        elif isinstance(gp_cand, GP):
            has_target = np.any(
                np.array(
                    [gi.attribute_col == target_col for gi in gp_cand.gradual_items],
                    dtype=bool,
                )
            )

        # Reject candidates containing the target feature.
        if exclude_target:
            return not has_target

        # Accept only candidates containing the target feature.
        return has_target

