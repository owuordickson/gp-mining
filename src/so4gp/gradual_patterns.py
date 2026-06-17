# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0.
# See the LICENSE file at the root of this
# repository for complete details.

"""
@author: Dickson Owuor
@credits: Thomas Runkler, Edmond Menya, and Anne Laurent
@license: GNU GPL v3
@email: owuordickson@gmail.com
@created: 21 July 2021
@modified: 02 August 2025

A collection of Gradual Pattern classes and methods.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class PairwiseMatrix:
    bin_mat: np.ndarray
    support: float


class GI:

    def __init__(self, attr_col, symbol):
        """
        GI (Gradual Item). A class that is used to create GI objects. A GI is a pair (i,v) where is a column, and v is a variation symbol -
        increasing/decreasing. Each column of a data set yields 2 GIs; for example, column age yields GI age+ or age-.

        >>> import so4gp as sgp
        >>> gradual_item = sgp.GI(1, "+")
        >>> print(gradual_item.to_string())
        1+

        :param attr_col: Column index
        :type attr_col: int

        :param symbol: Variation symbol either "+" or "-"
        :type symbol: str

        """
        self._attribute_col = attr_col
        """:type attribute_col: int"""
        self._symbol = ""
        """:type symbol: str"""
        if symbol == "-" or symbol == "+":
            self._symbol = symbol
        else:
            print(f"Invalid variation symbol: {symbol}")
            raise ValueError("Invalid variation symbol. It should be either '+' or '-'.")

    @property
    def attribute_col(self) -> int:
        """The column index of a GI"""
        return self._attribute_col

    @property
    def symbol(self) -> str:
        """The variation symbol of a GI"""
        return self._symbol

    @property
    def as_tuple(self) -> tuple[int, str]:
        """The Gradual Item (GI) in tuple format"""
        return tuple((self._attribute_col, self._symbol))

    def as_string(self, columns: list[str]) -> str:
        """
        The Gradual Item (GI) in string format col_pos|col_neg.

        :param columns: Column/feature titles/names of the dataset.
        :return: GI as a word
        """
        suffix = ""
        if self.symbol == "-":
            suffix = "_neg"
        elif self.symbol == "+":
            suffix = "_pos"
        col_title: str = columns[self.attribute_col]
        return f"{col_title.lower()}{suffix}"

    def to_string(self) -> str:
        """
        Returns a GI in string format
        :return: string
        """
        return f"{self._attribute_col}{self._symbol}"

    @classmethod
    def from_string(cls, gi_str: str) -> "GI":
        """Creates a GI from a string like '1+', '12-', or '125+'"""
        if not gi_str or gi_str[-1] not in ('+', '-'):
            print(f"Invalid GI format: '{gi_str}'. Must end with '+' or '-'.")
            raise ValueError(f"Invalid GI format: '{gi_str}'. Must end with '+' or '-'.")

        try:
            # Everything except the last char is the column
            attr_col = int(gi_str[:-1])
            # Only the last char is the symbol
            symbol = gi_str[-1]

            return cls(attr_col, symbol)
        except ValueError:
            print(f"Invalid GI format: '{gi_str}'. Must be in the format 'column_number+'.")
            raise ValueError(f"Invalid column number in: '{gi_str}'")

    @staticmethod
    def swap_gi_symbol(gi_obj: "GI") -> "GI":
        """
        Inverts a GI symbol to the opposite variation (i.e., from - to +; or, from + to -)
        :return: inverted GI object
        """
        if gi_obj.symbol == "+":
            sym = "-"
        else:
            sym = "+"
        return GI(gi_obj.attribute_col, sym)

    @staticmethod
    def parse_gi(gi_str: str) -> "GI":
        """
        Converts a stringified GI into normal GI. The accepted format is '1_neg' or 1_pos'.

        :param gi_str: A stringified GI
        :type gi_str: str

        :return: GI
        """
        txt = gi_str.split('_')
        attr_col = int(txt[0])
        if txt[1] == 'neg':
            symbol = "-"
        else:
            symbol = "+"
        return GI(attr_col, symbol)


class GP:

    def __init__(self):
        """
        GP (Gradual Pattern). A class that is used to create GP objects. A GP object is a set of gradual items (GI),
        and its quality is measured by its computed support value. For example, given a data set with 3 columns
        (age, salary, cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies
        that 8 out of 10 objects have the values of column age 'increasing' and column 'salary' decreasing.

        >>> import so4gp as sgp
        >>> gradual_pattern = sgp.GP()
        >>> gradual_pattern.add_gradual_item(sgp.GI(0, "+"))
        >>> gradual_pattern.add_gradual_item(sgp.GI(1, "-"))
        >>> gradual_pattern.support = 0.5
        >>> print(f"{gradual_pattern.to_string()}: {gradual_pattern.support}")

        """
        self._gradual_items: list[GI] = list()
        self._support: float = 0
        self._density: float = 0
        self._avg_dev_from_diag: float = 0
        self._rank_dispersion: float = 0
        self._graph_connectivity: int = 0
        self._singularity_score: float = 0

    @property
    def gradual_items(self) -> list[GI]:
        return self._gradual_items

    @property
    def support(self) -> float:
        return self._support

    @support.setter
    def support(self, support: float):
        self._support = round(support, 3) if support <= 1 else support

    @property
    def density(self) -> float:
        return self._density

    @property
    def avg_deviation_from_diagonal(self) -> float:
        return self._avg_dev_from_diag

    @property
    def rank_dispersion(self) -> float:
        return self._rank_dispersion

    @property
    def graph_connectivity(self) -> int:
        return self._graph_connectivity

    @property
    def singularity_score(self) -> float:
        return self._singularity_score

    def add_gradual_item(self, item: GI) -> bool:
        """
        Adds a gradual item (GI) into the gradual pattern (GP)
        :param item: gradual item

        :return: True if gradual item is added, None otherwise
        """
        if not isinstance(item, GI):
            raise TypeError("Invalid gradual item")
        self._gradual_items.append(item)
        return True

    @property
    def as_set(self) -> set[str]:
        """Returns the gradual pattern (GP) as a set of strings: {'1+', '2-'}"""
        return set(self.to_string())

    @property
    def as_swapped_set(self) -> set[str]:
        """Returns the gradual pattern (GP) as a set of strings: {'1-', '2+'}"""
        gp = GP.swap_gp_symbols(self)
        return set(gp.to_string())

    def get_computed_descriptors(self, descriptor_title) -> list[str] | list[dict]:
        """
        Returns the computed descriptors of the gradual pattern (GP)

        :param descriptor_title: If True, returns a dictionary with column names as keys and descriptors as values

        :return: List of descriptors
        """
        if self.density <= 0:
            params = [f"sup={self.support}"] if not descriptor_title else [{"Support": f"{self.support}"}]
        else:
            if not descriptor_title:
                params = [f"sup={self.support}",
                          f"density={self.density}",
                          f"avg_dev={self.avg_deviation_from_diagonal}",
                          f"dispersion={self.rank_dispersion}",
                          f"connect={self.graph_connectivity}",
                          f"singularity_scr={self.singularity_score}"]
            else:
                params = [{"Support": f"{self.support}"},
                          {"Density": f"{self.density}"},
                          {"Avg. Deviation from Diagonal": f"{self.avg_deviation_from_diagonal}"},
                          {"Rank Dispersion": f"{self.rank_dispersion}"},
                          {"Graph Connectivity": f"{self.graph_connectivity}"},
                          {"Singularity Score": f"{self.singularity_score}"}]
        return params

    def decompose(self) -> tuple[list[int], list[str]]:
        """
        Breaks down all the gradual items (GIs) in the gradual pattern into columns and variation symbols and returns
        them as separate variables. For instance, a GP {"1+", "3-"} will be returned as [1, 3], [1, -1]: where [1, 3] is
        the list of attributes/features and [1, -1] are their corresponding gradual variations (1 -> '+' and 1- -> '-').

        :return: Separate columns and variation symbols
        """
        attrs = list()
        syms = list()
        for item in self._gradual_items:
            gi = item.as_tuple
            attrs.append(gi[0])
            syms.append(gi[1])
        return attrs, syms

    def contains_attr(self, gi: GI) -> bool:
        """
        Checks if any gradual item (GI) in the gradual pattern (GP) is composed of the column
        :param gi: gradual item
        :type gi: GI

        :return: True if a column exists, False otherwise
        """
        if gi is None:
            return False
        for gi_obj in self._gradual_items:
            if gi.attribute_col == gi_obj.attribute_col:
                return True
        return False

    def to_string(self) -> list[str]:
        """
        Returns the GP in string format
        :return: string
        """
        pattern = list()
        for item in self._gradual_items:
            pattern.append(item.to_string())
        return pattern

    def print(self, columns: list[str], descriptor_title: bool = False) -> tuple[str, list[str] | list[dict]]:
        """
        A method that returns patterns with actual column names

        :param columns: Column names
        :param descriptor_title: If True, returns a dictionary with column names as keys and descriptors as values

        :return: GP with actual column names
        """

        # Pattern
        pattern = ""
        i = 0
        for item in self._gradual_items:
            col_title = columns[item.attribute_col]
            pat = str(col_title + item.symbol)
            # pattern.append(pat)  # (item.to_string())
            pattern += pat + ", " if i < len(self._gradual_items) - 1 else pat
            i += 1

        # Descriptors
        params = self.get_computed_descriptors(descriptor_title)
        return pattern, params

    def validate_graank(self, d_gp) -> "GP":
        """
        Validates a candidate gradual pattern (GP) based on support computation. A GP is invalid if its support value is
        less than the minimum support threshold set by the user. It uses a breath-first approach to compute support.

        :param d_gp: Data_GP object
        :type d_gp: so4gp.DataGP # noinspection PyTypeChecker

        :return: A valid GP or an empty GP
        """
        # pattern = [('2', "+"), ('4', "+")]
        min_supp = d_gp.thd_supp
        n = d_gp.attr_size
        gi_dict = d_gp.valid_bins.copy()
        gi_key_list = list(gi_dict.keys())

        gen_pattern: GP = GP()
        pw_mat_1: PairwiseMatrix | None = None
        for gi in self.gradual_items:
            arg = np.argwhere(np.isin(np.array(gi_key_list), gi.to_string()))
            if len(arg) > 0:
                i = arg[0][0]
                if pw_mat_1 is None:
                    pw_mat_1 = gi_dict[gi_key_list[i]]
                    gen_pattern.add_gradual_item(gi)
                else:
                    pw_mat_2 = gi_dict[gi_key_list[i]]
                    res_pw_mat = GP.perform_and(pw_mat_1, pw_mat_2, n)
                    if res_pw_mat.support >= min_supp:
                        pw_mat_1 = PairwiseMatrix(bin_mat=res_pw_mat.bin_mat.copy(), support=res_pw_mat.support)
                        gen_pattern.add_gradual_item(gi)
                        gen_pattern.support = res_pw_mat.support
        if len(gen_pattern.gradual_items) <= 1:
            return self
        else:
            return gen_pattern

    def validate_tree(self, d_gp):
        """
        Validates a candidate gradual pattern (GP) based on support computation. A GP is invalid if its support value is
        less than the minimum support threshold set by the user. It applies a depth-first (FP-Growth) approach
        to compute support.

        :param d_gp: Data_GP object
        :type d_gp: so4gp.DataGP # noinspection PyTypeChecker

        :return: A valid GP or an empty GP
        """
        if d_gp.warping_set is None:
            return self

        min_supp = d_gp.thd_supp
        n = d_gp.row_count
        gen_pattern = GP()
        """type gen_pattern: GP"""
        temp_tids = None
        for gi in self.gradual_items:
            node = gi.to_string()
            node_inv = GI.swap_gi_symbol(gi).to_string()
            for gi_str, gi_tids in d_gp.warping_set.items():
                if (node == gi_str) or (node_inv == gi_str):
                    if temp_tids is None:
                        temp_tids = set(gi_tids)
                        gen_pattern.add_gradual_item(gi)
                    else:
                        temp = (temp_tids or {}).copy()
                        temp = temp.intersection(set(gi_tids))
                        supp = float(len(temp)) / float(n * (n - 1.0) / 2.0)
                        if supp >= min_supp:
                            temp_tids = temp.copy()
                            gen_pattern.add_gradual_item(gi)
                            gen_pattern.support = supp
        if len(gen_pattern.gradual_items) <= 1:
            return self
        else:
            return gen_pattern

    def check_am(self, gp_list: list["GP"] | None, subset: bool = True) -> bool:
        """
        Anti-monotonicity check. Checks if a GP is a subset or superset of an already existing GP

        :param gp_list: A list of existing GPs
        :param subset: A check if it is a subset
        :return: True if superset/subset, False otherwise
        """
        result = False
        if gp_list is None:
            return result

        if subset:
            for pat in gp_list:
                result1 = set(self.as_set).issubset(set(pat.as_set))
                result2 = set(self.as_swapped_set).issubset(set(pat.as_set))
                if result1 or result2:
                    result = True
                    break
        else:
            for pat in gp_list:
                result1 = set(self.as_set).issuperset(set(pat.as_set))
                result2 = set(self.as_swapped_set).issuperset(set(pat.as_set))
                if result1 or result2:
                    result = True
                    break
        return result

    def is_duplicate(self, valid_gps: list["GP"] | None, invalid_gps: list["GP"]|None = None) -> bool:
        """
        Checks if a pattern is in the list of winner GPs or loser GPs

        :param valid_gps: list of GPs
        :param invalid_gps: list of GPs
        :return: True if a pattern is a list, False otherwise
        """
        if valid_gps is None:
            return False

        if invalid_gps is None:
            pass
        else:
            for pat in invalid_gps:
                if set(self.as_set) == set(pat.as_set) or \
                        set(self.as_swapped_set) == set(pat.as_set):
                    return True
        for pat in valid_gps:
            if set(self.as_set) == set(pat.as_set) or \
                    set(self.as_swapped_set) == set(pat.as_set):
                return True
        return False

    def compute_descriptors(self, warping_set: np.ndarray | None, obj_count: int) -> bool:
        """
        Computes gradual warping set (GWS) descriptors for a given gradual pattern.

        The descriptors are defined as follows:

        1. Density (ρ_g):
            Proportion of concordant index pairs relative to all possible pairs
            ρ_g = |W_g| / C(n, 2)

        2. Average Deviation from Diagonal (μ_g):
            Mean absolute distance |i - j| across all pairs in W_g.

        3. Rank Dispersion (σ_g):
            Standard deviation of |i - j|, capturing variability of index distances across all pairs in W_g.

        4. Graph Connectivity (κ_g):
            Number of connected components when W_g is interpreted as an undirected graph.

        5. Singularity Score (S_g):
            Measures concentration of index participation (node degree skewness).
            High values indicate dominance of certain indices.

        Path-like behavior (DTW-like) is approximated when:
            κ_g = 1, S_g is low, and σ_g is smooth (low variance).

        :param warping_set: np.ndarray of shape (k, 2), containing index pairs (i, j)
        :param obj_count: Total number of objects (n)

        :return: True if descriptors are computed successfully, False otherwise
        """

        if warping_set is None or len(warping_set) == 0 or obj_count < 2:
            return False

        # Ensure numpy array
        w_set = np.asarray(warping_set)
        i_vals = w_set[:, 0]
        j_vals = w_set[:, 1]

        pair_count = len(w_set)
        total_pairs = obj_count * (obj_count - 1) / 2.0

        def compute_density() -> float:
            """
            Warping set density

            ρ_g = |W_g| / C(n, 2)
            """
            return float(pair_count) / float(total_pairs)

        def compute_avg_dev_from_diagonal() -> float:
            """
            Average Deviation from Diagonal

            μ_g = (1 / |W_g|) * Σ |i - j|
            """
            deviations = np.abs(i_vals - j_vals)
            return float(np.mean(deviations))

        def compute_rank_dispersion() -> float:
            """
            Rank Dispersion

            σ_g = sqrt((1 / |W_g|) * Σ (|i - j| - μ_g)^2)
            """
            deviations = np.abs(i_vals - j_vals)
            return float(np.std(deviations))

        def compute_graph_connectivity(active_only: bool = True) -> int:
            """
            Computes the graph connectivity (number of connected components) of the gradual warping set W_g.

            The warping set W_g is interpreted as an undirected graph G = (V, E), where:
                - V is the set of object indices
                - E = W_g is the set of edges (i, j)

            Two modes of computation are supported:

            1. Global connectivity (active_only = False):
                - V = {0, 1, ..., n-1}
                - Includes all dataset objects, even those not present in W_g
                - Isolated nodes are counted as individual connected components
                - Captures global fragmentation of the dataset

            2. Active connectivity (active_only = True):
                - V = set of indices appearing in W_g
                - Ignores isolated nodes not participating in any pair
                - Captures structural connectivity of the gradual pattern itself

            Interpretation:
                - κ_g = 1 indicates a fully connected structure
                - Higher κ_g indicates fragmentation
                - Path-like behavior is approximated when κ_g = 1

            :param active_only: If True, compute connectivity using only nodes present in W_g;
                                otherwise, include all dataset nodes.
            :return: Number of connected components (κ_g)
            """
            nodes = np.unique(w_set) if active_only else range(obj_count)
            parent = {node: node for node in nodes}
            count = len(nodes)

            def find(i):
                if parent[i] == i: return i
                parent[i] = find(parent[i])
                return parent[i]

            for u, v in w_set:
                # Skip edges if nodes aren't in your predefined set
                if u in parent and v in parent:
                    root_u, root_v = find(u), find(v)
                    if root_u != root_v:
                        parent[root_u] = root_v
                        count -= 1
            return count

        def compute_singularity_score() -> float:
            """
            Singularity Score

            S_g = normalized variance of node degrees

            Steps:
            - Count degree of each index
            - Compute variance normalized by mean degree
            """
            degree = np.zeros(obj_count)

            for u, v in w_set:
                degree[int(u)] += 1
                degree[int(v)] += 1

            mean_deg = np.mean(degree)
            if mean_deg == 0:
                return 0.0

            return float(np.var(degree) / mean_deg)

        # Compute descriptors
        self._density = round(compute_density(), 3)
        self._avg_dev_from_diag = round(compute_avg_dev_from_diagonal(), 3)
        self._rank_dispersion = round(compute_rank_dispersion(), 3)
        self._graph_connectivity = compute_graph_connectivity()
        self._singularity_score = round(compute_singularity_score(), 3)
        return True

    @staticmethod
    def swap_gp_symbols(gp_obj: "GP") -> "GP":
        """
        Swaps the variation symbols of all the gradual items (GIs) in a gradual pattern (GP)
        """
        new_gp = GP()
        for gi in gp_obj.gradual_items:
            new_gp.add_gradual_item(GI.swap_gi_symbol(gi))
        return new_gp

    @staticmethod
    def perform_and(bin_data_1: "PairwiseMatrix|None", bin_data_2: "PairwiseMatrix|None", dim: int) -> "PairwiseMatrix":
        """
        Perform logical AND operation on two bitmaps.

        :param bin_data_1: Bitmap 1
        :param bin_data_2: bitmap 2
        :param dim: dimension of the bitmaps
        """
        if bin_data_1 is None or bin_data_2 is None:
            return PairwiseMatrix(bin_mat=np.zeros((dim, dim)), support=0)
        bin_mat = bin_data_1.bin_mat * bin_data_2.bin_mat
        sup = float(np.sum(bin_mat)) / float(dim * (dim - 1.0) / 2.0)
        return PairwiseMatrix(bin_mat=bin_mat, support=sup)


class TimeDelay:

    def __init__(self, tstamp=0, supp=0):
        """
            TimeDelay (Time Delay). A class used in Fuzzy Temporal Gradual Patterns to create the time-delay object.

        >>> import so4gp as sgp
        >>> t_delay = sgp.TimeDelay(3600, 0.75)
        >>> t_delay.to_string()

        :param tstamp: The time-delay value as a timestamp.
        :type tstamp: Float

        :param supp: The true value of the time-delay value.
        :type supp: Float
        """
        self._timestamp: float = tstamp
        self._support: float = round(supp, 3)
        self._valid: bool = False
        self._sign: str = ""
        self._formatted_time: dict = {}
        self._init_parameters()

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def support(self) -> float:
        return self._support

    @property
    def valid(self) -> bool:
        return self._valid

    @property
    def sign(self) -> str:
        return self._sign

    @property
    def formatted_time(self) -> dict:
        return self._formatted_time

    def _init_parameters(self):
        """Initializes the class parameters."""

        def delay_sign() -> str:
            """
            Checks and returns the sign of the time-delay value (later/before).

            :return: The sign of the time-delay value.
            """
            if self._timestamp < 0:
                return "-"
            else:
                return "+"

        def format_time() -> list:
            """
            Formats the time-delay value as a Date in string format (i.e., seconds/minutes/hours/days/weeks/months/years).

            :return: The formatted time-delay as a list.
            """
            stamp_in_seconds = abs(self._timestamp)
            years = stamp_in_seconds / 3.154e+7
            months = stamp_in_seconds / 2.628e+6
            weeks = stamp_in_seconds / 604800
            days = stamp_in_seconds / 86400
            hours = stamp_in_seconds / 3600
            minutes = stamp_in_seconds / 60
            if int(years) <= 0:
                if int(months) <= 0:
                    if int(weeks) <= 0:
                        if int(days) <= 0:
                            if int(hours) <= 0:
                                if int(minutes) <= 0:
                                    return [round(stamp_in_seconds, 0), "seconds"]
                                else:
                                    return [round(minutes, 0), "minutes"]
                            else:
                                return [round(hours, 0), "hours"]
                        else:
                            return [round(days, 0), "days"]
                    else:
                        return [round(weeks, 0), "weeks"]
                else:
                    return [round(months, 0), "months"]
            else:
                return [round(years, 0), "years"]

        self._sign: str = delay_sign()
        self._formatted_time: dict = {}
        if self._timestamp != 0:
            time_arr = format_time()
            self._formatted_time = {'value': time_arr[0], 'duration': time_arr[1]}
            self._valid = True

    def to_string(self) -> str:
        """
        Returns formated time-delay as a string.

        :return: The time-delay as a string.
        """
        if not self._formatted_time:
            txt = ("~ " + self._sign + str(self._formatted_time['value']) + " " + str(self._formatted_time['duration'])
                   + " : " + str(self._support))
        else:
            txt = "No time lag found!"
        return txt


class TGP(GP):
    @dataclass
    class TemporalGI:
        gradual_item: GI
        time_delay: TimeDelay

    def __init__(self):
        """
        A class that inherits an existing GP class to create Temporal GP objects. A TGP is a gradual pattern with a
        time-delay. It has a target gradual item (which is created from a user-defined attribute), and it is used as the
        anchor for mining patterns from a dataset. The class has the following attributes:

        target_gradual_item: the gradual item on which the pattern is based.

        temporal_gradual_items: gradual items which occur after specific time delays.

        >>> import so4gp as sgp
        >>> t_gp = sgp.TGP()
        >>> t_gp.target_gradual_item = sgp.GI(1, "+")
        >>> t_gp.add_temporal_gradual_item(sgp.GI(2, "-"), sgp.TimeDelay(7200, 0.8))
        >>> t_gp.to_string()
        """
        super(TGP, self).__init__()
        self._target_gradual_item: GI | None = None
        self._temporal_gradual_items: list[TGP.TemporalGI] = list()

    @property
    def target_gradual_item(self) -> GI | None:
        return self._target_gradual_item

    @target_gradual_item.setter
    def target_gradual_item(self, item: GI) -> None:
        """Adds a target gradual item (fTGI) into the fuzzy temporal gradual pattern (fTGP)"""
        if not isinstance(item, GI):
            raise TypeError("Target gradual item must be of type GI")
        self._target_gradual_item = item

    @property
    def temporal_gradual_items(self) -> list[TemporalGI]:
        return self._temporal_gradual_items

    def add_temporal_gradual_item(self, item: GI, time_delay: TimeDelay):
        """
            Adds a fuzzy temporal gradual item (fTGI) into the fuzzy temporal gradual pattern (fTGP)
            :param item: gradual item
            :type item: so4gp.GI

            :param time_delay: time delay
            :type time_delay: TimeDelay

            :return: void
        """
        if isinstance(item, GI) and isinstance(time_delay, TimeDelay):
            temp_gi = TGP.TemporalGI(gradual_item=item, time_delay=time_delay)
            self._temporal_gradual_items.append(temp_gi)
        else:
            raise TypeError("Invalid arguments - require GI and TimeDelay objects")

    def to_string(self) -> list:
        """
        Returns the Temporal-GP in string format as a list.
        """
        pattern = [self._target_gradual_item.to_string() if self._target_gradual_item else ""]
        for temp_gi in self._temporal_gradual_items:
            gi = temp_gi.gradual_item
            t_lag = temp_gi.time_delay
            str_time = f"{t_lag.sign}{t_lag.formatted_time['value']} {t_lag.formatted_time['duration']}"
            pattern.append(f"({gi.to_string()}) {str_time}")
        return pattern

    def print(self, columns: list[str], descriptor_title: bool = False) -> tuple[str, list[str] | list[dict]]:
        """
        A method that returns a fuzzy temporal gradual pattern (TGP) with actual column names

        :param columns: Column names
        :param descriptor_title: If True, prints the descriptor title

        :return: TGP with actual column names
        """

        target_gi = self._target_gradual_item
        col_title = columns[target_gi.attribute_col if target_gi else -1]
        pattern = f"{col_title}{target_gi.symbol if target_gi else ''}, "

        i = 0
        for temp_gi in self._temporal_gradual_items:
            gi = temp_gi.gradual_item
            t_lag = temp_gi.time_delay
            str_time = f"{t_lag.sign}{t_lag.formatted_time['value']} {t_lag.formatted_time['duration']}"
            col_title = columns[gi.attribute_col]
            pat = f"({col_title}{gi.symbol}) {str_time}"
            # pattern.append(pat)
            pattern += pat + ", " if i < len(self._temporal_gradual_items) - 1 else pat
            i += 1

        # Descriptors
        params = self.get_computed_descriptors(descriptor_title)
        return pattern, params

    def is_similar_to(self, ftgp) -> bool:
        """
        Checks if two fuzzy temporal gradual patterns are similar.

        :param ftgp: Fuzzy temporal gradual pattern to compare with.

        :return: True if the patterns are similar, False otherwise.
        """

        if not isinstance(ftgp, TGP):
            return False

        # Compare target gradual items
        tgt1 = self.target_gradual_item
        tgt2 = ftgp.target_gradual_item
        swapped = False
        if tgt1 is None or tgt2 is None:
            return False
        if tgt1 is not None and tgt2 is not None:
            if tgt1.to_string() != tgt2.to_string():
                if GI.swap_gi_symbol(tgt1).to_string() != tgt2.to_string():
                    return False
                else:
                    swapped = True

        # Compare temporal gradual items
        lst_tgi1 = self.temporal_gradual_items
        lst_tgi2 = ftgp.temporal_gradual_items
        if (len(lst_tgi1) != len(lst_tgi2)) and (len(lst_tgi1) <= 0) or (len(lst_tgi2) <= 0):
            return False

        gi_set1 = set([tgi.gradual_item.to_string() for tgi in lst_tgi1])
        gi_set1_swap = set([GI.swap_gi_symbol(tgi.gradual_item).to_string() for tgi in lst_tgi1])
        gi_set2 = set([tgi.gradual_item.to_string() for tgi in lst_tgi2])
        if gi_set1 != gi_set2:
            if swapped and gi_set1_swap != gi_set2:
                return False
            else:
                return False

        # Compare time delays
        td_set1 = set([f"{tgi.time_delay.sign}{tgi.time_delay.formatted_time['value']} {tgi.time_delay.formatted_time['duration']}" for tgi in lst_tgi1])
        td_set2 = set([f"{tgi.time_delay.sign}{tgi.time_delay.formatted_time['value']} {tgi.time_delay.formatted_time['duration']}" for tgi in lst_tgi2])
        if td_set1 != td_set2:
            return False

        # All checks passed, patterns are similar
        return True
