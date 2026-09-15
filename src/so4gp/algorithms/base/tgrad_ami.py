# SPDX-License-Identifier: MIT
# See the LICENSE file at the root of this
# repository for complete details.

import time
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from .tgrad import TGrad


class TGradAMI(TGrad):

    def __init__(self, *args, **kwargs):
        """
        Algorithm for estimating time-lag using Average Mutual Information (AMI) and KMeans clustering which is
        extended to mining gradual patterns. The average mutual information I(X; Y) is a measure of the “information”
        amount that the random variables X and Y provide about one another.

        This algorithm extends the work published in: https://ieeexplore.ieee.org/abstract/document/8858883. TGradAMI
        is an algorithm that improves the classical TGrad algorithm for extracting more accurate temporal gradual
        patterns.  It computes Mutual Information (MI) with respect to target-column with original dataset to get
        the actual relationship between variables: by computing MI for every possible time-delay and if the transformed
        dataset has the same almost identical MI to the original dataset, then it selects that as the best time-delay.
        Instead of min-representativity value, the algorithm relies on the error-margin between MIs.

        :param args: [required] data source path of Pandas DataFrame, [optional] minimum-support, [optional] eq
        :param kwargs: [required] target-column or attribute or feature, [optional] minimum representativity,
        [optional] MF shape, [optional] clustering algorithm, [optional] inference method.

        """
        super(TGradAMI, self).__init__(*args, **kwargs)
        self._mi_error: float = 0
        self._transformation_data: dict = {}

    @property
    def mi_error(self):
        return self._mi_error

    @property
    def transformation_data(self):
        return self._transformation_data

    def find_best_mutual_info(self, error_margin: float, feature_cols: np.ndarray) -> tuple[dict[int, int], int]:
        """
        Estimate the optimal time transformation for each feature using
        Average Mutual Information (AMI).

        For each feature, this method computes the mutual information (MI)
        between the target attribute and:

        1. The original (untransformed) dataset.
        2. All candidate time-transformed datasets that satisfy the minimum
           representativity constraint.

        The optimal transformation is the one whose MI differs from the
        original dataset by at most the specified error margin. This approach
        assumes that the best time delay preserves the information shared
        between the feature and the target attribute.

        To simplify comparison during optimization, an MI value of zero
        (indicating no mutual information) is internally encoded as ``-1``.
        This sentinel value allows the algorithm to distinguish the absence
        of mutual information from very small positive MI values while
        preserving equality comparisons between the original and transformed
        datasets.

        Args:
            error_margin:
                Maximum allowable absolute difference between the mutual
                information of the original dataset and that of a transformed
                dataset.

            feature_cols:
                Feature matrix excluding the target attribute. The feature matrix
                contains the indices of these attributes. Each column is
                independently evaluated to determine its optimal temporal
                transformation.

        Returns:
            A dictionary mapping each feature column index to its selected
            transformation step (estimated time delay) and the maximum transformation step.

        Notes:
            Only transformed datasets satisfying the minimum representativity
            threshold are considered during the search for the optimal time
            transformation.
        """

        # 1. Compute MI for original dataset w.r.t. target-col

        y = np.array(self.full_attr_data[self._target_col], dtype=float).T
        x_data = np.array(self.full_attr_data[feature_cols], dtype=float).T
        init_mi_info = np.array(mutual_info_regression(x_data, y), dtype=float)

        # 2. Compute all the MI for every time-delay and compute error
        mi_list = []
        for step in range(1, self.max_step):
            # Compute MI
            attr_data, _ = self.transform_and_mine(step, return_patterns=False)
            y = np.array(attr_data[self._target_col], dtype=float).T
            x_data = np.array(attr_data[feature_cols], dtype=float).T
            try:
                mi_vals = np.array(mutual_info_regression(x_data, y), dtype=float)
            except ValueError:
                optimal_dict = {int(feature_cols[i]): step for i in range(len(feature_cols))}
                self._mi_error = -1
                self.min_rep = round(((self.row_count - step) / self.row_count), 5)
                return optimal_dict, step

            # Compute MI error
            squared_diff = np.square(np.subtract(mi_vals, init_mi_info))
            mse_arr = np.sqrt(squared_diff)
            is_mi_preserved = np.all(mse_arr <= error_margin)
            if is_mi_preserved:
                optimal_dict = {int(feature_cols[i]): step for i in range(len(feature_cols))}
                self._mi_error = round(np.min(mse_arr), 5)
                self.min_rep = round(((self.row_count - step) / self.row_count), 5)
                return optimal_dict, step
            mi_list.append(mi_vals)
        mi_info_arr = np.array(mi_list, dtype=float)

        # 3. Standardize MI array
        mi_info_arr[mi_info_arr == 0] = -1

        # 4. Identify steps (for every feature w.r.t. target) with minimum error from initial MI
        squared_diff = np.square(np.subtract(mi_info_arr, init_mi_info))
        mse_arr = np.sqrt(squared_diff)
        # mse_arr[mse_arr < self.error_margin] = -1
        optimal_steps_arr = np.argmin(mse_arr, axis=0)
        max_step = int(np.max(optimal_steps_arr)) + 1

        # 5. Integrate feature indices with the computed steps
        optimal_dict = {int(feature_cols[i]): int(optimal_steps_arr[i] + 1) for i in range(len(feature_cols))}

        self._mi_error = round(np.min(mse_arr), 5)
        self.min_rep = round(((self.row_count - max_step) / self.row_count), 5)
        return optimal_dict, max_step

    def gather_delayed_data(self, optimal_dict: dict, max_step: int) -> tuple[np.ndarray|None, dict]:
        """
        A method that combined attribute data with different data transformations and computes the corresponding
        time-delay values for each attribute.

        :param optimal_dict: Raw transformed dataset.
        :param max_step: Largest data transformation step.
        :return: Combined transformed dataset with corresponding time-delay values.
        """

        delayed_data: np.ndarray|None = None
        time_data: dict = {}  # {col1: [time-lags], col2: [time-lags]}
        n = self.row_count
        k = (n - max_step)  # Number of rows created by the largest step-delay
        for col_index in range(self.col_count):
            if (col_index == self._target_col) or (col_index in self.time_cols):
                # date-time column OR target column
                temp_col = self.full_attr_data[col_index][0: k]
            else:
                # other attributes
                step = optimal_dict[col_index]
                temp_col = self.full_attr_data[col_index][step: n]
                _, _, time_diffs_arr = self.get_time_diffs(step)
                time_data[col_index] = time_diffs_arr

                # Get first k items for delayed data
                temp_col = temp_col[0: k]


                # for i in range(k):
                #    if i in time_dict:
                #        time_dict[i].append(time_diffs[i])
                #    else:
                #        time_dict[i] = [time_diffs[i]]
                # print(f"{time_diffs}\n")
                # WHAT ABOUT TIME DIFFERENCE/DELAY? It is different for every step!!!
            delayed_data = temp_col if (delayed_data is None) \
                else np.vstack((delayed_data, temp_col))
        return delayed_data, time_data

    def discover_tgp_ami(self, target_col: int, search_algorithm: str = "apriori",
                         max_iteration: int=3, transformation_steps: dict|None = None,
                         error_margin: float = 0.0001, eval_mode: bool = False) -> dict:
        """
        A method that applies mutual information concept, clustering, and hill-climbing algorithm to find the best data
        transformation that maintains MI and estimate the best time-delay value of the mined Fuzzy Temporal Gradual
        Patterns (FTGPs).
    
        :param target_col: [required] Index of the target attribute/feature/column. Temporal transformations are
        estimated relative to this attribute.
        :param search_algorithm: Gradual pattern mining algorithm to apply to the transformed dataset. Supported values
        are ``apriori``, ``ga``, ``aco``, ``pso``, ``hc``, ``random``, and ``clustergp``. Defaults to ``"apriori"``.
        :param max_iteration: Maximum number of iterations to run the search algorithm.
        :param transformation_steps: Data transformation steps (used to override the computed transformation steps).
        :param error_margin: [optional] minimum Mutual Information error margin.
        :param eval_mode: Run algorithm in evaluation mode.

        :return: List of (FTGPs as DICT object) or (FTGPs and evaluation data as a Python dict) when executed in evaluation mode.
        """

        start = time.time()
        self._target_col = target_col
        self._search_algorithm = search_algorithm
        self._algorithm_max_iter = max_iteration
        self.clear_gradual_patterns()
        # 1. Compute and find the lowest mutual information
        if transformation_steps is not None:
            optimal_dict = transformation_steps
            max_step = 0
            for _, v in optimal_dict.items():
                if v > max_step:
                    max_step = v
        else:
            feature_cols: np.ndarray = np.setdiff1d(self.attr_cols, self._target_col)
            optimal_dict, max_step = self.find_best_mutual_info(error_margin=error_margin, feature_cols=feature_cols)

        # 2. Create a final (and dynamic) delayed dataset
        delayed_data, time_data = self.gather_delayed_data(optimal_dict, max_step)

        # 3. Discover temporal-GPs from time-delayed data
        lst_tgp = self._mine_gps_at_step(time_delay_data=time_data, attr_data=delayed_data)

        # 4. Organize FTGPs into a single list
        if lst_tgp:
            for tgp in lst_tgp:
                self.add_gradual_pattern(tgp)

        # 5. Check if the algorithm is in evaluation mode
        if eval_mode:
            title_row = []
            time_title = []
            for col, txt in enumerate(self.titles):
                title_row.append(txt)
                if (col != self._target_col) and (col not in self.time_cols):
                    time_title.append(txt)
            str_time_data = {"".join(self.titles[k]): v for k, v in time_data.items()}
            self._transformation_data = {
                'Patterns': self.display_patterns,
                'Transformation Steps': optimal_dict,
                'Time Data': str_time_data,
                'Transformed Data': np.vstack(
                    (np.array(title_row), delayed_data.T if delayed_data is not None else np.array([]))),
            }

        duration = time.time() - start
        out_dict: dict[str, str | list | np.ndarray | None | dict] = {
            "Algorithm": "TGradAMI",
            # "Memory Usage (MiB)": f{mem_use)}",
            "GP Search Algorithm": f"{self._search_algorithm}",
            "Maximum Iteration for Search Algorithm": f"{self._algorithm_max_iter}",
            "Minimum Representation": f"{self.min_rep:.2f}",
            "MI Minimum Error": f"{error_margin:.2f}",
            "MI Error": f"{self.mi_error:.2f}",
            "Target Column": f"{self._target_col}",
            "Run-time": f"{duration:.6f} seconds"}
        return out_dict
