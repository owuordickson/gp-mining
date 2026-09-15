# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# See the LICENSE file at the root of this
# repository for complete details.

"""
@author: Dickson Owuor
@credits: Thomas Runkler, Edmond Menya, and Anne Laurent
@license: MIT
@email: owuordickson@gmail.com
@created: 21 July 2021

A collection of Gradual Pattern classes and methods.
"""


import copy
import torch
import numpy as np
from dataclasses import dataclass


NO_TIME_LABEL = "NoTime"

@dataclass
class PairwiseMatrix:
    """A data-class for storing pairwise (bitmap) matrix as packed-bits and its support value."""
    packed_bin_mat: np.ndarray|torch.Tensor
    support: float
    pattern: set[str]
    time_lag: "TimeDelay|None"=None


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

    def add_gradual_item(self, item: GI) -> None|bool:
        """
        Add a gradual item to this gradual pattern. First checks if the gradual item's column already exists in the GP.

        Args:
            item:
                Gradual item to add.

        Returns:
            True if the item was added successfully, None otherwise.

        Raises:
            TypeError:
                If ``item`` is not a :class:`GI`.
        """
        if not isinstance(item, GI):
            raise TypeError("item must be an instance of GI.")

        if not self.contains_attr(item):
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

    def contains_attr(self, gi: GI|None) -> bool:
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

    def validate_via_graank(self, data_gp, target_col: int | None, time_data: dict | None=None) -> "GP|TGP":
        """
        Validates a candidate gradual pattern (GP) based on support computation. A GP is invalid if its support value is
        less than the minimum support threshold set by the user. It uses a breath-first approach to compute support.

        :param data_gp: a :class:`so4gp.DataGP` object
        :type data_gp: so4gp.DataGP # noinspection PyTypeChecker
        :param target_col: (optional) target column for estimating time lag.
        :param time_data: (optional) time data for estimating time lag.

        :return: A valid GP or an empty GP
        """
        # pattern = [('2', "+"), ('4', "+")]
        min_supp = data_gp.thd_supp
        n = data_gp.attr_size
        gi_dict = copy.deepcopy(data_gp.valid_bins)

        gen_pattern: GP | TGP = TGP() if time_data is not None else GP()
        target_gi = self.gradual_items[0]
        if target_col is not None:
            if f"{target_col}+" in self.as_set:
                target_gi = GI(target_col, "+")
            elif f"{target_col}-" in self.as_set:
                target_gi = GI(target_col, "-")

        pw_mat_1: PairwiseMatrix = gi_dict[target_gi.to_string()]
        time_lag = gi_dict[target_gi.to_string()].time_lag
        GP.add_gradual_item_strict(gen_pattern, target_gi, target_col=target_col, time_lag=time_lag)

        for gi in self.gradual_items:
            if gi.to_string() == target_gi.to_string():
                continue
            else:
                pw_mat_2 = gi_dict[gi.to_string()]
                pw_mat_1 = GP.perform_and(pw_mat_1, pw_mat_2, n, time_data=time_data)
                if pw_mat_1.support >= min_supp:
                    GP.add_gradual_item_strict(gen_pattern, gi, target_col=target_col, time_lag=pw_mat_1.time_lag)
                    gen_pattern.support = pw_mat_1.support
        if len(gen_pattern.gradual_items) <= 1:
            return self
        else:
            # if compute_descriptors:
            #    warping_set_arr: np.ndarray = np.array(DataGP.gen_gradual_warping_set(pw_mat.bin_mat, as_array=True))
            #    rand_gp.compute_descriptors(warping_set_arr, obj_count=self.row_count)
            return gen_pattern

    def validate_via_tree(self, d_gp):
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
                        temp = set(copy.deepcopy(temp_tids or {}))
                        temp = temp.intersection(set(gi_tids))
                        supp = float(len(temp)) / GP.pair_count(n)
                        if supp >= min_supp:
                            temp_tids = copy.deepcopy(temp)
                            gen_pattern.add_gradual_item(gi)
                            gen_pattern.support = supp
        if len(gen_pattern.gradual_items) <= 1:
            return self
        else:
            return gen_pattern

    def check_am(self, gp_list: list["GP|TGP"] | None, subset: bool = True) -> bool:
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

    def is_duplicate(self, valid_gps: list["GP|TGP"]|None, invalid_gps: list["GP|TGP"]|None = None) -> bool:
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

    def compute_descriptors(self, warping_set: np.ndarray | torch.Tensor | None, obj_count: int,) -> bool:
        """
        Compute gradual warping set (GWS) descriptors.

        The descriptors are defined as:

        1. Density (ρ_g):
            Proportion of concordant index pairs relative to all possible pairs::

                ρ_g = |W_g| / C(n, 2)

        2. Average Deviation from Diagonal (μ_g):
            Mean absolute distance between paired indices::

                μ_g = mean(|i - j|)

        3. Rank Dispersion (σ_g):
            Standard deviation of the absolute index distances::

                σ_g = std(|i - j|)

        4. Graph Connectivity (κ_g):
            Number of connected components when ``W_g`` is interpreted as
            an undirected graph.

        5. Singularity Score (S_g):
            Normalized variance of node degrees::

                S_g = Var(degree) / mean(degree)

            Higher values indicate greater concentration of edge participation
            among a smaller number of nodes.

        When ``warping_set`` is a CUDA tensor, numerical calculations and
        graph connectivity are performed on the GPU. When it is a NumPy
        array or CPU tensor, NumPy/Python implementations are used.

        Args:
            warping_set:
                Array or tensor of shape ``(k, 2)`` containing index pairs
                ``(i, j)``.
            obj_count:
                Total number of objects.

        Returns:
            True if descriptors are computed successfully, otherwise False.
        """
        if warping_set is None or len(warping_set) == 0 or obj_count < 2:
            return False

        # ------------------------------------------------------------------
        # Prepare warping set
        # ------------------------------------------------------------------
        is_tensor = isinstance(warping_set, torch.Tensor)
        w_set_cpu = None
        w_set_gpu = None

        if is_tensor:
            w_set_gpu = warping_set
            if w_set_gpu.ndim != 2 or w_set_gpu.shape[1] != 2:
                return False

            # Edge indices must be integers.
            # w_set_gpu = w_set_gpu.long()

            i_vals = w_set_gpu[:, 0]
            j_vals = w_set_gpu[:, 1]

            pair_count = len(w_set_gpu)
        else:
            w_set_cpu = np.asarray(warping_set)
            if w_set_cpu.ndim != 2:# or w_set.shape[1] != 2:
                return False

            i_vals = w_set_cpu[:, 0]
            j_vals = w_set_cpu[:, 1]

            pair_count = len(w_set_cpu)
        total_pairs = (obj_count * (obj_count - 1)) / 2.0

        # ------------------------------------------------------------------
        # Density
        # ------------------------------------------------------------------
        def compute_density() -> float:
            """
            Compute warping set density.

            ρ_g = |W_g| / C(n, 2)
            """
            return float(pair_count) / total_pairs

        # ------------------------------------------------------------------
        # Average deviation from diagonal
        # ------------------------------------------------------------------
        def compute_avg_dev_from_diagonal() -> float:
            """
            Compute average absolute distance from the diagonal.
            """
            if isinstance(i_vals, torch.Tensor) and isinstance(j_vals, torch.Tensor):
                deviations = torch.abs(i_vals - j_vals)
                return deviations.float().mean().item()
            else:
                deviations = np.abs(i_vals - j_vals)
                return float(np.mean(deviations))

        # ------------------------------------------------------------------
        # Rank dispersion
        # ------------------------------------------------------------------
        def compute_rank_dispersion() -> float:
            """
            Compute standard deviation of index distances.
            """
            if isinstance(i_vals, torch.Tensor) and isinstance(j_vals, torch.Tensor):
                deviations = torch.abs(i_vals - j_vals).float()
                # correction=0 gives population standard deviation,
                # equivalent to np.std(..., ddof=0).
                return deviations.std(correction=0).item()

            deviations = np.abs(i_vals - j_vals)
            return float(np.std(deviations))

        # ------------------------------------------------------------------
        # Graph connectivity - GPU
        # ------------------------------------------------------------------
        def compute_graph_connectivity_gpu(edges: torch.Tensor|None, active_only: bool = True,) -> int:
            """
            Compute connected components using GPU label propagation.

            The graph is treated as undirected.

            Each node initially receives its own label. During each iteration,
            the minimum label is propagated across every edge in both
            directions. The process terminates when labels no longer change.

            Args:
                edges:
                    CUDA tensor of shape ``(k, 2)`` containing graph edges.
                active_only:
                    If True, only nodes appearing in ``edges`` are counted.
                    If False, all ``obj_count`` nodes are counted.

            Returns:
                Number of connected components.
            """
            if edges is None:
                return 0

            device = edges.device

            # --------------------------------------------------------------
            # Initialize node labels
            # --------------------------------------------------------------
            labels = torch.arange(
                obj_count,
                device=device,
                dtype=torch.long,
            )

            # --------------------------------------------------------------
            # Active nodes
            # --------------------------------------------------------------
            #if active_only:
            #    active_nodes = torch.unique(edges)

            # --------------------------------------------------------------
            # Edge endpoints
            # --------------------------------------------------------------
            u = edges[:, 0]
            v = edges[:, 1]

            # --------------------------------------------------------------
            # Iterative label propagation
            # --------------------------------------------------------------
            #
            # For an undirected edge (u, v):
            #
            #     label[u] <- min(label[u], label[v])
            #     label[v] <- min(label[v], label[u])
            #
            # scatter_reduce performs the operation for all edges in
            # parallel.
            #
            for _ in range(obj_count - 1):

                new_labels = labels.clone()

                # Propagate v -> u
                new_labels.scatter_reduce_(
                    dim=0,
                    index=u,
                    src=labels[v],
                    reduce="amin",
                    include_self=True,
                )

                # Propagate u -> v
                new_labels.scatter_reduce_(
                    dim=0,
                    index=v,
                    src=labels[u],
                    reduce="amin",
                    include_self=True,
                )

                # Stop when no labels changed.
                if torch.equal(
                        new_labels,
                        labels,
                ):
                    labels = new_labels
                    break

                labels = new_labels

            # --------------------------------------------------------------
            # Count components
            # --------------------------------------------------------------
            if active_only:
                active_nodes = torch.unique(edges)
                component_count = torch.unique(labels[active_nodes]).numel()
            else:
                component_count = torch.unique(labels).numel()
            return int(component_count)

        # ------------------------------------------------------------------
        # Graph connectivity - NumPy
        # ------------------------------------------------------------------
        def compute_graph_connectivity_cpu(edges: np.ndarray|None, active_only: bool = True,) -> int:
            """
            Compute connected components using CPU union-find.
            """
            if edges is None:
                return 0

            if active_only:
                nodes = np.unique(edges)
            else:
                nodes = range(obj_count)

            parent = {int(node): int(node) for node in nodes}
            count = len(parent)

            def find(node: int) -> int:
                while parent[node] != node:
                    parent[node] = parent[ parent[node]]
                    node = parent[node]
                return node

            for u, v in edges:
                u = int(u)
                v = int(v)

                if u not in parent or v not in parent:
                    continue

                root_u = find(u)
                root_v = find(v)

                if root_u != root_v:
                    parent[root_u] = root_v
                    count -= 1
            return count

        # ------------------------------------------------------------------
        # Select connectivity implementation
        # ------------------------------------------------------------------
        def compute_graph_connectivity(active_only: bool = True,) -> int:
            """
            Compute graph connectivity using the appropriate backend.
            """
            if isinstance(w_set_gpu, torch.Tensor):
                return compute_graph_connectivity_gpu(w_set_gpu, active_only=active_only,)
            else:
                return compute_graph_connectivity_cpu(w_set_cpu, active_only=active_only,)

        # ------------------------------------------------------------------
        # Singularity score
        # ------------------------------------------------------------------
        def compute_singularity_score() -> float:
            """
            Compute normalized variance of node degrees.

            S_g = Var(degree) / mean(degree)
            """
            if isinstance(i_vals, torch.Tensor) and isinstance(j_vals, torch.Tensor):
                # Each edge contributes one degree to each endpoint.
                degree = (
                        torch.bincount(i_vals, minlength=obj_count,)
                        +
                        torch.bincount(j_vals, minlength=obj_count,)
                ).float()

                mean_deg = degree.mean()
                if mean_deg.item() == 0.0:
                    return 0.0

                variance = degree.var(correction=0)
                return (variance / mean_deg).item()
            elif isinstance(i_vals, np.ndarray) and isinstance(j_vals, np.ndarray):
                # --------------------------------------------------------------
                # NumPy implementation
                # --------------------------------------------------------------
                degree = np.zeros(obj_count, dtype=np.int64,)

                np.add.at(degree, i_vals.astype(np.int64),1,)
                np.add.at(degree, j_vals.astype(np.int64),1,)

                mean_deg = np.mean(degree)
                if mean_deg == 0.0:
                    return 0.0

                return float(np.var(degree) / mean_deg)
            return float(0)

        # ------------------------------------------------------------------
        # Compute descriptors
        # ------------------------------------------------------------------
        self._density = round(compute_density(), 3,)
        self._avg_dev_from_diag = round(compute_avg_dev_from_diagonal(), 3,)
        self._rank_dispersion = round(compute_rank_dispersion(), 3,)
        self._graph_connectivity = compute_graph_connectivity(active_only=True)
        self._singularity_score = round(compute_singularity_score(), 3,)

        return True

    @staticmethod
    def pair_count(n: int) -> float:
        """
        Get the total number of pairs in a given number of objects (n)

        :param n: Number of objects (or attribute size)

        :return: Total number of pairs in the dataset
        """
        if n < 2:
            return 1.0
        return float(n * (n - 1.0) / 2.0)

    @staticmethod
    def add_gradual_item_strict(gp: "GP|TGP", gi: GI, target_col: int|None = None, time_lag: "TimeDelay|None" = None) -> "GP|TGP":
        """
        Add a gradual item to a gradual pattern using pattern-aware placement.

        Handles the structural differences between regular gradual patterns (GPs) and
        temporal gradual patterns (TGPs). For a TGP, the gradual item is
        assigned as the target gradual item when its attribute corresponds
        to ``target_col``; otherwise, it is added as a temporal gradual item
        together with its associated time lag.

        For a regular GP, the gradual item is added directly to the pattern.

        Args:
            gp:
                Gradual pattern to which the gradual item should be added.
                Must be an instance of :class:`GP` or :class:`TGP`.

            gi:
                Gradual item to add to the pattern.

            target_col:
                Column index of the target attribute. When ``gp`` is a
                :class:`TGP` and ``gi.attribute_col`` matches this value,
                ``gi`` is assigned as the TGP's target gradual item.

            time_lag:
                Temporal delay associated with ``gi`` when it is added to a
                TGP as a temporal gradual item.

        Returns:
            The modified gradual pattern with the added gradual item.

        Raises:
            TypeError:
                If ``gp`` is not a :class:`GP` or :class:`TGP`, or if ``gi``
                is not a valid gradual item.

            ValueError:
                If a temporal gradual item requires a time lag but
                ``time_lag`` is not provided.
        """
        if not isinstance(gp, (GP, TGP)):
            raise TypeError("gp must be an instance of GP or TGP.")

        if not isinstance(gi, GI):
            raise TypeError("gi must be an instance of GI.")

        if isinstance(gp, TGP) and target_col is not None:
            if gi.attribute_col == target_col:
                gp.target_gradual_item = gi
            else:
                if time_lag is None:
                    raise ValueError("time_lag must be provided for temporal gradual items.")
                gp.add_temporal_gradual_item(gi, time_lag)
        else:
            gp.add_gradual_item(gi)
        return gp

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
    def get_selected_rows(packed_bit_mat: np.ndarray|torch.tensor, dim: int) -> np.ndarray | torch.Tensor:
        """
        Get objects participating in at least one active warping relation.

        Returns:
            Unique object indices.
        """

        edge_list = GP.gen_gradual_warping_set(packed_bit_mat, dim, )
        if isinstance(edge_list, torch.Tensor):
            return torch.unique(edge_list.flatten())
        return np.unique(edge_list.flatten())

    @staticmethod
    def gen_gradual_warping_set(packed_pairwise_mat: np.ndarray | torch.Tensor, n: int) -> np.ndarray | torch.Tensor:
        """
        A method that decomposes the pairwise matrix of a gradual item/pattern into a warping set. Attributes that have
        strong correlation will produce a warping set with dense zigzag patterns when plotted as a graph. Those with weak
        correlation will produce a warping set with sparse zigzag patterns.

        :param packed_pairwise_mat: The pairwise matrix of a gradual item/pattern, reduced to packed bits.
        :param n: Number of rows/columns in the unpacked pairwise matrix.

        :return: A list array of the warping path (as an edge list) as a numpy array.
        """

        if isinstance(packed_pairwise_mat, torch.Tensor):
            return GP.gen_gradual_warping_set_gpu(packed_pairwise_mat, n)

        pairwise_mat = np.unpackbits(packed_pairwise_mat, count=n * n).reshape(n, n).astype(bool)
        edge_lst: list[tuple[int, int]] = [(i, j) for i, row in enumerate(pairwise_mat) for j, val in enumerate(row) if
                                           val]
        edge_lst = sorted(list(edge_lst), key=lambda x: x[0])
        return np.array(edge_lst)

    @staticmethod
    def gen_gradual_warping_set_gpu(packed: torch.Tensor, n: int, ) -> torch.Tensor:
        """Convert a packed CUDA bitmap directly to edge indices.

        Args:
            packed: 1-D uint8 CUDA tensor containing the packed bitmap.
            n: Number of rows/columns in the original pairwise matrix.

        Returns:
            CUDA tensor of shape (E, 2), where each row is ``(i, j)``.
        """

        bit_masks = torch.tensor(
            [128, 64, 32, 16, 8, 4, 2, 1],
            dtype=torch.uint8,
            device=packed.device,
        )

        # Determine which bits are set.
        bits = (packed[:, None] & bit_masks).flatten()

        # Remove np.packbits() padding.
        bits = bits[:n * n]

        # Get flattened positions of set bits.
        positions = torch.nonzero(bits, as_tuple=False).flatten()

        # Convert flattened indices to (row, column).
        rows = positions // n
        cols = positions % n

        return torch.stack((rows, cols), dim=1)

    @staticmethod
    def perform_and(bin_data_1: "PairwiseMatrix|None", bin_data_2: "PairwiseMatrix|None", dim: int, time_data: dict|None=None) -> "PairwiseMatrix":
        """
        Perform logical AND operation on two bitmaps.

        :param bin_data_1: Bitmap 1
        :param bin_data_2: bitmap 2
        :param dim: dimension of the bitmaps
        :param time_data: (optional) time data for estimating time lag
        """

        if bin_data_1 is None or bin_data_2 is None:
            return PairwiseMatrix(packed_bin_mat=np.zeros((dim, dim)), support=0, pattern=set())

        # Intersection of packed bitmaps -- Supports NumPy arrays and PyTorch tensors (CPU or CUDA)
        packed_1 = bin_data_1.packed_bin_mat
        packed_2 = bin_data_2.packed_bin_mat

        if isinstance(packed_1, torch.Tensor):
            if not isinstance(packed_2, torch.Tensor):
                raise TypeError("Both packed bitmaps must be either NumPy arrays or PyTorch tensors.")

            if packed_1.device != packed_2.device:
                raise ValueError("Packed tensors must be on the same device.")

            packed_bit_mat = torch.bitwise_and(packed_1, packed_2)
            bit_counts = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.int64, device=packed_bit_mat.device,)
            sup = (bit_counts[packed_bit_mat.long()].sum().item() / GP.pair_count(n=dim))
        else:
            if isinstance(packed_2, torch.Tensor):
                raise TypeError("Both packed bitmaps must be either NumPy arrays or PyTorch tensors.")

            packed_bit_mat = np.bitwise_and(packed_1, packed_2)
            bit_counts = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8,)
            sup = (bit_counts[packed_bit_mat].sum() / GP.pair_count(n=dim))

        # Combine gradual items
        gp = bin_data_1.pattern | bin_data_2.pattern

        # Time-delay computation
        if time_data is not None:
            selected_rows = GP.get_selected_rows(packed_bit_mat, dim,)
            t_lag = TimeDelay.approx_time_lag(selected_rows, time_data, gp_set=gp)
            return PairwiseMatrix(packed_bin_mat=packed_bit_mat, support=sup, time_lag=t_lag, pattern=gp,)

        return PairwiseMatrix(packed_bin_mat=packed_bit_mat, support=sup,pattern=gp,)


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
        self._formatted_time: dict = {'value': 0, 'duration': ''}
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

    @staticmethod
    def predict_time(crisp_inputs: np.ndarray, time_data: np.ndarray, fuzzy_mfs: list[dict], inference_method: str) -> float:
        """Predict time using a multi-antecedent fuzzy inference system.

        Each crisp input is first fuzzified against the same set of
        membership functions. The membership degree corresponding to each
        MF is then combined across *all* inputs using the fuzzy AND
        operator (minimum).

        The method therefore implements a multi-antecedent rule of the
        form:

            Input 1 is MF_i
            AND Input 2 is MF_i
            AND ...
            AND Input n is MF_i
            --------------------------------
            Output is Aggregate(MF_i, ..., MF_i)

        More generally, the activated membership functions identified by
        the inputs can be viewed as a single multi-input fuzzy rule. The
        implementation does **not** explicitly construct the combinatorial
        ``M^n`` rule base. Instead, it directly computes the AND activation
        of the membership functions selected by the crisp inputs.

        For ``n`` inputs and ``M`` membership functions, the fuzzification
        matrix has shape ``(n, M)``:

            Input 1 -> [mu_10, mu_11, ..., mu_1M]
            Input 2 -> [mu_20, mu_21, ..., mu_2M]
            ...
            Input n -> [mu_n0, mu_n1, ..., mu_nM]

        The fuzzy AND is then applied column-wise:

            firing_strength[i]
                = min(
                    mu_1i,
                    mu_2i,
                    ...,
                    mu_ni
                  )

        Thus, if three inputs activate ``MF_i``, ``MF_j`` and ``MF_k``,
        respectively, their membership degrees are combined as:

            w = AND(
                mu_i(x_1),
                mu_j(x_2),
                mu_k(x_3)
            )

        The corresponding output membership functions are aggregated
        before applying the rule firing strength:

            Output_MF(t)
                = Aggregate(
                    MF_i(t),
                    MF_j(t),
                    MF_k(t)
                  )

        where aggregation is performed using the maximum operator:

            mu_Output(t)
                = max(
                    mu_i(t),
                    mu_j(t),
                    mu_k(t)
                  )

        The resulting output is then inferred using either Mamdani or
        Larsen implication:

        Mamdani:

            mu_rule(t) = min(
                firing_strength,
                mu_Output(t)
            )

        Larsen:

            mu_rule(t) = firing_strength * mu_Output(t)

        Finally, the outputs are aggregated using the maximum operator
        and converted to a crisp prediction using centroid
        defuzzification.

        Importantly, the implementation avoids constructing all possible
        ``M^n`` combinations of membership functions. Only the membership
        functions actually activated by the supplied crisp inputs
        participate in the inference.

        Args:
            crisp_inputs:
                One-dimensional sequence containing the crisp input
                values. All supplied inputs participate in the same
                multi-antecedent fuzzy AND rule.

            time_data:
                One-dimensional array of observed time-delay values.
                These values define the universe of discourse for the
                output variable.

            fuzzy_mfs:
                Membership-function specifications generated by
                ``build_membership_functions()``. Each specification must
                contain:

                    {
                        "shape": "triangular"
                                 | "trapezoidal"
                                 | "gaussian",
                        "params": [...]
                    }

            inference_method:
                The technique to use for fuzzy inference. Either 'mamdani' or 'larsen'.

        Returns:
            The predicted time value obtained through centroid
            defuzzification.

        Raises:
            ValueError:
                If no crisp inputs, time values, or membership functions
                are provided; if the inputs contain non-finite values; if
                an unsupported membership-function shape is supplied; or
                if an invalid inference method is specified.

        Notes:
            The computational complexity of the inference stage is
            approximately ``O(n * M + M * U)``, where:

                n = number of crisp inputs,
                M = number of membership functions,
                U = number of output-universe samples.

            This avoids the ``O(M^n)`` combinatorial explosion that would
            result from explicitly constructing every possible
            multi-antecedent rule.
        """

        # Validate inputs
        inputs = np.asarray(crisp_inputs, dtype=np.float64, ).ravel()

        if inputs.size == 0:
            raise ValueError( "crisp_inputs must contain at least one value.")

        if not np.all(np.isfinite(inputs)):
            raise ValueError("crisp_inputs must contain only finite values.")

        time_values = np.asarray(time_data, dtype=np.float64,).ravel()
        time_values = time_values[np.isfinite(time_values)]

        if time_values.size == 0:
            raise ValueError("time_data must contain at least one finite value.")

        if not fuzzy_mfs:
            raise ValueError("fuzzy_mfs must contain at least one membership function.")

        def evaluate_mfs(values: np.ndarray) -> np.ndarray:
            """
            Evaluate all membership functions for the supplied values.

            Returns:
                shape = (number_of_values, number_of_MFs)
            """

            values = np.asarray(values, dtype=np.float64, ).ravel()
            num_values = values.size
            num_mfs = len(fuzzy_mfs)

            memberships = np.empty((num_values, num_mfs), dtype=np.float64, )
            eps = np.finfo(np.float64).eps

            for i, mf in enumerate(fuzzy_mfs):
                shape = mf["shape"].lower()
                params = np.asarray(mf["params"], dtype=np.float64,)

                # ------------------------------------------------------
                # Triangular MF
                # ------------------------------------------------------
                if shape == "triangular":
                    if params.size != 3:
                        raise ValueError( "Triangular membership functions require [left, center, right].")
                    left, center, right = params
                    rising = ((values - left) / max(center - left, eps))
                    falling = ( (right - values) / max(right - center, eps))

                    memberships[:, i] = np.maximum(0.0, np.minimum(rising, falling),)

                # ------------------------------------------------------
                # Trapezoidal MF
                # ------------------------------------------------------
                elif shape == "trapezoidal":
                    if params.size != 4:
                        raise ValueError("Trapezoidal membership functions require [left, left_peak, right_peak, right].")
                    left, left_peak, right_peak, right = params
                    rising = ((values - left) / max(left_peak - left, eps))
                    falling = ((right - values) / max(right - right_peak, eps))

                    memberships[:, i] = np.maximum(
                        0.0, np.minimum( np.minimum(rising, 1.0), falling,),)

                # ------------------------------------------------------
                # Gaussian MF
                # ------------------------------------------------------
                elif shape == "gaussian":
                    if params.size != 2:
                        raise ValueError("Gaussian membership functions require [center, sigma].")

                    center, sigma = params
                    sigma = max(abs(float(sigma)), eps,)

                    memberships[:, i] = np.exp(-0.5 * ((values - center) / sigma) ** 2)
                else:
                    raise ValueError(
                        f"Unsupported membership-function shape: {shape!r}. Expected 'triangular', 'trapezoidal', or 'gaussian'."
                    )

            return np.clip(memberships,0.0,1.0,)

        # Output universe
        universe = np.linspace(
            min(0.0, float(time_values.min()) - 5.0),
            float(time_values.max()) + 5.0,
            num=max(1000, time_values.size * 10),
        )

        # --------------------------------------------------------------
        # Fuzzify all crisp inputs.
        #
        # Example with 3 inputs and 5 MFs:
        #
        #     fuzzified =
        #
        #     [
        #       [mu_10, mu_11, mu_12, mu_13, mu_14],
        #       [mu_20, mu_21, mu_22, mu_23, mu_24],
        #       [mu_30, mu_31, mu_32, mu_33, mu_34]
        #     ]
        #
        # --------------------------------------------------------------
        fuzzified = evaluate_mfs(inputs)

        # --------------------------------------------------------------
        # Multi-input AND
        #
        # Combine every input's membership degree for the same MF.
        #
        #     firing_strength[i]
        #         = min_j fuzzified[j, i]
        #
        # This directly computes the multi-antecedent rule activation
        # without constructing M^n combinations.
        # --------------------------------------------------------------
        firing_strengths = np.min(fuzzified, axis=0,)

        # --------------------------------------------------------------
        # Evaluate the MFs over the output universe.
        #
        # output_mfs:
        #
        #     shape = (U, M)
        #
        #     output_mfs[t, i] = MF_i(t)
        # --------------------------------------------------------------
        output_mfs = evaluate_mfs(universe)

        # --------------------------------------------------------------
        # Aggregate the output MFs that participate in the activated
        # multi-input rule.
        #
        # Instead of constructing combinations, each MF is weighted/
        # activated directly according to its multi-input AND strength.
        #
        # For the common case where all inputs refer to the same MF
        # index, this gives:
        #
        #     MF_i AND MF_i AND ... MF_i
        #
        #     -> Aggregate(MF_i, MF_i, ..., MF_i)
        #
        # For inputs whose strongest memberships occur at different MFs,
        # the activated MFs are combined through their firing strengths.
        # --------------------------------------------------------------
        method = inference_method.lower()

        if method not in {"mamdani", "larsen"}:
            raise ValueError(f"Unsupported inference method: {inference_method!r}. Expected 'mamdani' or 'larsen'.")

        # --------------------------------------------------------------
        # Apply the multi-antecedent activation to every output MF.
        #
        # Mamdani:
        #
        #     min(w_i, MF_i(t))
        #
        # Larsen:
        #
        #     w_i * MF_i(t)
        #
        # --------------------------------------------------------------
        if method == "mamdani":
            rule_outputs = np.minimum(output_mfs, firing_strengths[None, :],)
        else:
            rule_outputs = (output_mfs * firing_strengths[None, :])

        # --------------------------------------------------------------
        # Aggregate all activated output membership functions.
        #
        #     mu_aggregated(t)
        #         = max_i rule_output_i(t)
        #
        # --------------------------------------------------------------
        aggregated_mf = np.max( rule_outputs, axis=1,)

        # Centroid defuzzification
        total_membership = np.sum(aggregated_mf,)

        if total_membership <= np.finfo(np.float64).eps:
            return float(np.mean(inputs))

        prediction = (
                np.sum(universe * aggregated_mf,) / total_membership
        )
        return float(prediction)

    @classmethod
    def approx_time_lag(cls, selected_rows: np.ndarray|torch.Tensor, time_data: dict|None, gp_set: set | None=None) -> "TimeDelay":
        """
        A method that uses a fuzzy membership function to select the most accurate time-delay value. We implement two
        methods: (1) uses classical slide and re-calculate dynamic programming to find the best time-delay value and,
        (2) uses metaheuristic hill-climbing to find the best time-delay value.

        :param selected_rows: Only the rows where the GP is respected.
        :param time_data: A dict that contains time-data, mined GP, MFs parameters and inference method.
        :param gp_set: Gradual item object.

        :return: TimeDelay object.
        """

        if time_data is None:
            return cls(-1, 0)

        t_data: np.ndarray|None = time_data["time_data"]
        use_gp: bool = time_data["use_gp"]
        mf_data: list[dict] = time_data["fuzzy_mfs"]
        inference: str = time_data["inference"]
        gp_set = gp_set if use_gp else None

        if t_data is None:
            return cls(-1, 0)

        # 2. Get TimeDelay Array
        lst_rows = selected_rows.cpu().tolist() if isinstance(selected_rows, torch.Tensor) else selected_rows.tolist()
        if gp_set is not None and isinstance(t_data, dict):
            ## t_data = {col1: [row time-lags], col2: [row time-lags]}
            t_lag_lst = []
            sel_cols: set = set(t_data.keys())
            for gi_str in gp_set:
                col = GI.from_string(gi_str).attribute_col
                if col in sel_cols:
                    t_lag_lst.append(t_data[col])
            t_lag_arr = np.array(t_lag_lst)
            t_lag_arr = t_lag_arr[:, lst_rows]
            print(f"w GPs: {t_lag_arr}")
        else:
            ## t_data = [row time-lags]
            t_lag_arr = np.ndarray([t_data[lst_rows]])
            print(f"w/o GPs: {t_lag_arr}")

        # 3. Approximate TimeDelay value
        time_val: float = TimeDelay.predict_time(crisp_inputs=t_lag_arr, time_data=t_data, fuzzy_mfs=mf_data, inference_method=inference)
        best_time_lag: TimeDelay = cls(time_val, 0.99)

        return best_time_lag


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
        self.add_gradual_item(item)

    @property
    def temporal_gradual_items(self) -> list[TemporalGI]:
        return self._temporal_gradual_items

    def add_temporal_gradual_item(self, item: GI, time_delay: TimeDelay|None):
        """
            Adds a fuzzy temporal gradual item (fTGI) into the fuzzy temporal gradual pattern (fTGP)
            :param item: gradual item
            :type item: so4gp.GI

            :param time_delay: time delay
            :type time_delay: TimeDelay

            :return: void
        """
        if item is None or time_delay is None:
            return

        if isinstance(item, GI) and isinstance(time_delay, TimeDelay):
            temp_gi = TGP.TemporalGI(gradual_item=item, time_delay=time_delay)
            self._temporal_gradual_items.append(temp_gi)
            self.add_gradual_item(item)
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

        :param columns: Column names of the dataset
        :param descriptor_title: If True, prints the descriptor title

        :return: TGP with actual column names
        """

        target_gi = self._target_gradual_item
        col_title = columns[target_gi.attribute_col if target_gi else -1]
        pattern = f"{col_title}{target_gi.symbol if target_gi else ''}, "
        has_no_time = True if NO_TIME_LABEL in columns else False

        i = 0
        for temp_gi in self._temporal_gradual_items:
            gi = temp_gi.gradual_item
            t_lag = temp_gi.time_delay
            str_time = f"{t_lag.sign}{t_lag.formatted_time['value']} {"lag" if has_no_time else t_lag.formatted_time['duration']}"
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

    def get_causal_relations(self, columns: list) -> list[dict[str, object]]:
        """
        Return the causal relationships represented by this temporal gradual pattern.

        Each temporal gradual item is interpreted as having a causal relationship
        with the target gradual item. The returned metadata includes the causal
        direction, estimated time lag, and the support of the temporal gradual
        pattern.

        :param columns: Column names of the dataset

        Returns:
            A list of dictionaries, where each dictionary contains:

            - ``causality``: A pair of attribute indices in the form
              ``[target_attribute, related_attribute]``.
            - ``direction``: ``"+"`` if both gradual items have the same trend
              (increasing/increasing or decreasing/decreasing), otherwise ``"-"``.
            - ``time_lag``: Human-readable estimated time lag.
            - ``support``: Support value of the temporal gradual pattern.

        Notes:
            The returned relationships describe inferred temporal associations.
            They should not be interpreted as proof of statistical or causal
            inference without additional domain validation.
        """
        relations: list[dict[str, object]] = []
        has_no_time = True if NO_TIME_LABEL in columns else False

        target = self.target_gradual_item
        if target is None:
            return relations

        for temporal_item in self.temporal_gradual_items:
            lag = temporal_item.time_delay

            time_lag = None
            if lag is not None:
                time_lag = (
                    f"{lag.sign}"
                    f"{lag.formatted_time['value']} "
                    f"{"lag" if has_no_time else lag.formatted_time['duration']}"
                )

            relations.append(
                {
                    "correlation": [
                        target.attribute_col,
                        temporal_item.gradual_item.attribute_col,
                    ],
                    "direction": (
                        "+"
                        if target.symbol == temporal_item.gradual_item.symbol
                        else "-"
                    ),
                    "time lag": time_lag,
                    "support": self.support,
                }
            )

        return relations
