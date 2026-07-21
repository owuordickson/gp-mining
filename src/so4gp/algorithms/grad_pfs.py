# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0.
# See the LICENSE file at the root of this
# repository for complete details.

import ntpath
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .base.graank_alg import OrigGRAANK
from .cluster_gp import ClusterGP
from .base.graank_aco import AntGRAANK
from .base.graank_ga import GeneticGRAANK


class GradPFS:
    """
    GradPFS is a filter-based algorithm for performing univariate or/and multivariate feature selection through gradual
    patterns for regression tasks. This algorithm is published in:
    """

    def __init__(self, data_src: str | pd.DataFrame, min_score: float = 0.75, target_col: int | None = None):
        """
        An algorithm based on the filter method for performing univariate or/and multivariate feature selection through
        gradual patterns for regression tasks (not suitable for classification tasks). The results are returned as a
        Pandas DataFrame.

        :param data_src: [required] the data in a CSV file or Pandas DataFrame.
        :param min_score: [optional] user-specified minimum correlation score for filtering redundant features, default=0.75.
        :param target_col: [optional] user-specified target column index, default=None.

        >>> import pandas
        >>> from so4gp.algorithms import GradPFS
        >>>
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> fs_obj = GradPFS(data_src=dummy_df)
        >>> gp_cor = fs_obj.univariate_fs()
        >>> fs_obj.generate_pdf_report(fs_type='U')
        >>>
        >>> # fs_obj.target_col = 2
        >>> # m_fs = fs_obj.multivariate_fs()
        >>> print(gp_cor)
                  Age  Salary  Cars  Expenses
        Age       1.0     0.6  -0.4      -1.0
        Salary    0.6     1.0  -0.3      -0.6
        Cars     -0.4    -0.3   1.0       0.4
        Expenses -1.0    -0.6   0.4       1.0

        """
        self.data_src = data_src
        """type data_src: str | pd.DataFrame"""
        self.file_path = ""
        """:type file_path: str"""
        if type(data_src) is str:
            self.file_path = data_src
        self.thd_score = min_score
        """:type thd_score: float"""
        self.target_col = target_col
        """:type target_col: int | None"""
        self.titles, self.data = None, None
        """:type titles: list | None"""
        """:type data: np.ndarray | None"""

    def univariate_fs(self) -> pd.DataFrame:
        """
        A method that runs the univariate GradPFS feature selection algorithm. The method that calculates the gradual
        correlation between each pair of attributes in the dataset. This is achieved by mining 2-attribute GPs and
        using their highest support values to show the correlation between them. The method returns a correlation
        matrix of feature similarities.

        :return: Correlation matrix as a pandas dataframe.
        """

        # 1. Instantiate GRAANK object and extract GPs
        grad = OrigGRAANK(self.data_src)
        self.titles = grad.titles
        self.data = grad.data
        grad.discover(ignore_support=True, apriori_level=2, target_col=self.target_col)

        # 2. Create a correlation matrix
        n = grad.col_count
        corr_mat = np.zeros((n, n), dtype=float)
        np.fill_diagonal(corr_mat, 1)

        # 3. Extract column names
        col_names = []
        for col_obj in grad.titles:
            # col_names[int(col_obj[0])] = col_obj[1].decode()
            col_names.append(col_obj[1].decode())
        col_names = np.array(col_names)

        # 4. Update correlation matrix with GP support
        for gp in (grad.gradual_patterns or []):
            score = gp.support
            i = int(gp.gradual_items[0].attribute_col)
            j = int(gp.gradual_items[1].attribute_col)
            i_symbol = str(gp.gradual_items[0].symbol)
            j_symbol = str(gp.gradual_items[1].symbol)

            if i_symbol != j_symbol:
                score = -score
            if abs(corr_mat[i][j]) < abs(score):
                corr_mat[i][j] = score
                corr_mat[j][i] = score

        # 5. Create Pandas DataFrame and return it as a result
        corr_mat = np.round(corr_mat, 4)
        corr_df = pd.DataFrame(corr_mat, columns=col_names)
        """:type corr_df: pd.DataFrame"""
        corr_df.index = col_names
        return corr_df

    def multivariate_fs(self, algorithm: str = 'GRAANK') -> pd.DataFrame | None:
        """
        Run the multivariate GradPFS feature selection algorithm.

        GradPFS performs gradual pattern-based feature selection in two stages:

        1. Gradual Pattern Mining:
           Mine gradual patterns that contain the specified target feature. These
           patterns represent gradual relationships that are relevant to the target
           variable.

        2. Feature Selection:
           Identify all features participating in the discovered gradual patterns.
           These features are returned as the selected (relevant) feature subset,
           while features that never participate in any target-related gradual pattern
           are considered irrelevant to the target variable.

        Args:
            algorithm:
                Gradual pattern mining algorithm used during feature selection.

                Supported values are:

                * ``"GRAANK"`` – Classical APRIORI-based gradual pattern mining.
                * ``"ACO"`` – Ant Colony Optimization.
                * ``"CLU"`` – Cluster-based gradual pattern mining.
                * ``"GA"`` – Genetic Algorithm.

                Defaults to ``"GRAANK"``.

        Returns:
            pandas.DataFrame:
                A DataFrame containing the selected features that are gradually
                associated with the target feature.

        Raises:
            ValueError:
                If no target feature has been specified before running the algorithm.

        Notes:
            GradPFS is a filter-based feature selection algorithm that exploits
            gradual dependencies between variables rather than conventional linear or
            statistical correlation measures.
        """

        if self.target_col is None:
            raise ValueError("You must specify a target feature (column index).")

        # 1. Instantiate GRAANK object and extract GPs
        algorithm += 'GRAANK'  # bypass for now (TO BE DELETED)
        if algorithm == 'CLU':
            grad = ClusterGP(self.data_src, min_sup=self.thd_score)
        elif algorithm == 'ACO':
            grad = AntGRAANK(self.data_src, min_sup=self.thd_score)
        elif algorithm == 'CLU':
            grad = GeneticGRAANK(self.data_src, min_sup=self.thd_score)
        else:
            grad = OrigGRAANK(self.data_src, min_sup=self.thd_score)
            grad.discover(target_col=self.target_col)
            # grad.discover(target_col=self.target_col, exclude_target=True)
        self.titles = grad.titles
        self.data = grad.data

        # 2. Extract column names
        col_names = []
        for col_obj in grad.titles:
            col_names.append(col_obj[1].decode())
        col_names = np.array(col_names)

        # 3a. Collect the irrelevant features (and redundant among themselves)
        rel_lst = []
        for gp in (grad.gradual_patterns or []):
             rel_attributes = gp.decompose()[0]
             for attr in rel_attributes:
                 rel_lst.append(attr)
        rel_set = set(rel_lst)
        rel_set = rel_set.difference({self.target_col})

        # # 4b. Identify irrelevant features by eliminating the relevant ones
        irr_set = set(grad.attr_cols.tolist()).difference(rel_set)
        irr_set = irr_set.difference({self.target_col})

        # # 3b. Collect the irrelevant features (and redundant among themselves)
        # irr_lst = []
        # for gp in grad.gradual_patterns:
        #     irr_attributes = gp.get_attributes()[0]
        #     for attr in irr_attributes:
        #         irr_lst.append(attr)
        # irr_set = set(irr_lst)
        #
        # # 4b. Identify relevant features by eliminating the irrelevant ones
        # rel_set = set(grad.attr_cols.tolist()).difference(irr_set)
        # rel_set = rel_set.difference({self.target_col})

        # # 5. Update the correlation list (relevant features w.r.t. target feature)
        irr_features = col_names[list(irr_set)]
        rel_features = col_names[list(rel_set)]
        corr_lst = [[{str(col_names[self.target_col])}, set(rel_features.tolist()), set(irr_features.tolist())],
                     [{self.target_col}, rel_set, irr_set]]

        # # 3c. Update correlation matrix with GP support
        # corr_lst = []
        # for gp in grad.gradual_patterns:
        #      score = gp.support
        #      lst_col = []
        #      lst_attr = []
        #      for gi in gp.gradual_items:
        #          att = gi.attribute_col
        #          att = -att if gi.symbol == '-' else att
        #          lst_col.append(att)
        #          lst_attr.append(col_names[att])
        #      corr_lst.append([set(lst_col), set(lst_attr), score])

        # 6. Create Pandas DataFrame and return it as a result
        if len(corr_lst) <= 0:
            return None
        corr_arr = np.array(corr_lst, dtype=object)
        # corr_df = pd.DataFrame(corr_arr, columns=[ "Attribute Indices", "Relevant Features", "GradPFS Score"])
        corr_df = pd.DataFrame(corr_arr, columns=["Target Feature", "Relevant Features", "Irrelevant Features"])
        """:type corr_df: pd.DataFrame"""
        return corr_df

    def generate_pdf_report(self, fs_type: str = 'U') -> bool:
        """
        A method that executes GradPFS algorithm for either Univariate Feature Selection ('U') or
        Multivariate Feature Selection ('M') and generates a PDF report.

        :param fs_type: Feature selection type: 'U' -> univariate or 'M' -> multivariate. Default is 'U'
        :return: True if a PDF report is generated.
        """

        # 2. Run a feature selection algorithm
        if fs_type == 'M':
            # 2a. Multivariate feature selection
            corr_df = self.multivariate_fs()
            fig_corr = None
            if corr_df is None:
                return False

            # Create table data
            tab_data = np.vstack([corr_df.columns, corr_df.to_numpy()])
            col_width = [1/3, 1/3, 1/3]
        else:
            # 2b. Univariate feature selection
            corr_mat_df = self.univariate_fs()
            lst_redundant = GradPFS.find_redundant_features(corr_mat_df.to_numpy(), self.thd_score)

            # Create a plot figure
            fig_corr = plt.Figure(figsize=(8.5, 8), dpi=300)
            ax_corr = fig_corr.add_subplot(1, 1, 1)
            sns.heatmap(corr_mat_df, annot=True, cmap="coolwarm", annot_kws={"size": 7}, ax=ax_corr)
            ax_corr.set_title("Univariate Feature Correlation Matrix")
            fig_corr.tight_layout(pad=3)  # Add padding to ensure the plot doesn't occupy the whole page

            # Create table data
            tab_data = [["Redundant Features", "GradPFS Score"]]
            for x in lst_redundant:
                feat = x[0]
                scores = np.round(x[1], 3)
                # Handle both single numbers and arrays/lists
                if isinstance(scores, np.ndarray):
                    score_val = tuple(scores.tolist())
                else:
                    # If it's just a single float/int
                    score_val = scores
                tab_data.append([str(feat), str(score_val)])
            tab_data = np.array(tab_data, dtype=object)
            col_width = [1/2, 1/2]

        # 3. Produce PDF report
        if isinstance(self.data_src, str):
            f_name = ntpath.basename(self.data_src)
            f_name = f_name.replace('.csv', '')
        else:
            f_name = ""

        if fs_type == 'M':
            out_info = [["Feature Selection Type", "Multivariate"]]
            pdf_file = f"{f_name}_multi_report.pdf"
        else:
            out_info = [["Feature Selection Type", "Univariate"]]
            pdf_file = f"{f_name}_uni_report.pdf"
        out_info.append(["Minimum Correlation Score", f"{self.thd_score}"])
        out_info = np.array(out_info, dtype=object)

        out_file = [["Encoding", "Feature Name"]]
        for txt in self.titles:
            col = int(txt[0])
            if (self.target_col is not None) and (col == self.target_col):
                out_file.append([f"{txt[0]}", f"{txt[1].decode()}** (target feature)"])
            else:
                out_file.append([f"{txt[0]}", f"{txt[1].decode()}"])
        # out_file.append(["File", f"{f_path}"])
        out_file = np.array(out_file, dtype=object)

        with (PdfPages(pdf_file)) as pdf:  # type: ignore
            pdf.savefig(GradPFS.generate_table("Gradual Pattern-based Feature Selection (GradPFS) Report",
                                               out_info, [2/3,1/3], xscale=0.5))
            if fig_corr is not None:
                pdf.savefig(fig_corr)
            pdf.savefig(GradPFS.generate_table("", out_file, [1/4, 3/4]))
            pdf.savefig(GradPFS.generate_table("", tab_data, col_width))
        return True

    @staticmethod
    def find_redundant_features(corr_arr: np.ndarray, thd_score: float) -> list:
        """
        A method that identifies features that are redundant using their correlation score.

        :param corr_arr: A correlation matrix as a numpy array.
        :param thd_score: A user-specified minimum correlation score for filtering redundant features.
        :return: Redundant features with the corresponding similarity/correlation score.
        """
        lst_redundant = []
        """:type lst_redundant: list"""
        lst_info = []
        """:type lst_info: list"""

        for i in range(corr_arr.shape[0]):  # row index
            lst_sim = []
            cor_scores = []
            for j in range(i, corr_arr.shape[1]):  # col index
                cor_score = corr_arr[i, j]
                if abs(cor_score) > thd_score:
                    lst_sim.append((-j if cor_score < 0 else j))
                    cor_scores.append(round(float(abs(cor_score)), 3))
            if len(lst_sim) <= 1:
                continue
            is_subset = False
            for item in lst_redundant:
                is_subset = set(lst_sim).issubset(item)
                if is_subset:
                    break
            if not is_subset:
                lst_redundant.append(set(lst_sim))
                lst_info.append([set(lst_sim), cor_scores])
        return lst_info

    @staticmethod
    def find_similar(corr_set: dict, cor_arr: np.ndarray):
        """
        A method that searches a correlation matrix for a specific set of features.

        :param corr_set: A set of features.
        :param cor_arr: A correlation matrix as a numpy array.
        :return: Found a set of features and correlation score.
        """

        row_idx = list(corr_set)[0]
        lst_sim = []
        cor_scores = []
        """:type lst_sim: list"""
        for j in list(corr_set):
            cor_score = cor_arr[row_idx, j]
            cor_scores.append(round(float(cor_score), 3))
            lst_sim.append(j)

        sim_set = set(lst_sim)
        """:type sim_set: set"""
        return [sim_set, cor_scores]

    @staticmethod
    def generate_table(title: str, data: np.ndarray, col_width: list, xscale: float = 1, yscale: float = 1.5):
        """
        A method that represents data in a table format using the matplotlib library.

        :param title: The title of the table.
        :param data: The data to be displayed.
        :param col_width: The width size of each column.
        :param xscale: The width of the table.
        :param yscale: The length of the table.
        :return: A matplotlib table.
        """
        #fig_tab = plt.Figure(figsize=(8.5, 11), dpi=300)
        # ax_tab = fig_tab.add_subplot(1, 1, 1)
        fig_tab, ax_tab = plt.subplots(figsize=(8.5, 11), dpi=300)

        ax_tab.set_axis_off()
        ax_tab.set_title(f"{title}")
        tab = ax_tab.table(cellText=data[:, :], loc='upper center', colWidths=col_width, cellLoc='left', ax=ax_tab)
        tab.scale(xscale, yscale)

        fig_tab.tight_layout()
        return fig_tab
