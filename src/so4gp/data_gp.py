# SPDX-License-Identifier: MIT
# See the LICENSE file at the root of this
# repository for complete details.

"""
@author: Dickson Owuor
@credits: Thomas Runkler, Edmond Menya, and Anne Laurent
@license: MIT
@email: owuordickson@gmail.com
@created: 21 July 2021

A collection of classes for pre-processing data for mining gradual patterns.
"""

import csv
import os
import statistics
import time

import numpy as np
import pandas as pd
import torch
from dateutil.parser import parse
from tabulate import tabulate

from .gradual_patterns import GI, GP, NO_TIME_LABEL, TGP, FatalError, PairwiseMatrix
from .utils import write_file


class DataGP:

    def __init__(self, data_source, min_sup=0.5, eq=False, add_time: bool = False, device: str = 'cpu') -> None:
        """
        A class for creating data-gp objects. A data-gp object is meant to store all the parameters required by GP
        algorithms to extract gradual patterns (GP). It takes a numeric file (in CSV format) as input and converts it
        into an object whose attributes are used by algorithms to extract GPs.

        :param data_source: [required] a data source, it can either be a 'file in csv format' or a 'Pandas DataFrame'
        :type data_source: pd.DataFrame | str

        :param min_sup: [optional] minimum support threshold, the default is 0.5
        :type min_sup: float

        :param eq: [optional] encode equal values as gradual, the default is False
        :type eq: bool

        :param add_time: [optional] add a dummy time column if the dataset is not a time-series

        :param device: [required] computing device to be used, 'cpu' or 'gpu', default is 'cpu'

        """
        self._data_src = data_source
        self._thd_supp: float = min_sup
        self._include_equal_values: bool = eq
        self._device: str = device
        self._titles, self._data = DataGP.read(data_source)
        """:type _titles: list"""
        """:type _data: np.ndarray"""
        self._row_count: int = 0
        self._col_count: int = 0
        self._time_cols: np.ndarray = np.array([])
        self._attr_cols: np.ndarray = np.array([])
        self._valid_bins: dict | None = None
        self._warping_set: dict | None = None
        self._attr_size: int = 0
        self._gradual_patterns: list[GP | TGP] | None = None
        self._init_attributes(create_time_index=add_time)

    @property
    def thd_supp(self) -> float:
        return self._thd_supp

    @property
    def titles(self) -> list:
        return self._titles

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def col_count(self) -> int:
        return self._col_count

    @property
    def time_cols(self) -> np.ndarray:
        return self._time_cols

    @property
    def attr_cols(self) -> np.ndarray:
        return self._attr_cols

    @property
    def valid_bins(self) -> dict | None:
        return self._valid_bins

    @property
    def warping_set(self) -> dict[str, list] | None:
        return self._warping_set

    @property
    def attr_size(self) -> int:
        return self._attr_size

    @property
    def gradual_patterns(self) -> list[GP | TGP] | None:
        return self._gradual_patterns

    @property
    def display_patterns(self) -> list:
        str_gps = []
        if self._gradual_patterns is None:
            return str_gps
        for gp in self._gradual_patterns:
            str_gp, gp_params = gp.print(self.titles)
            str_gps.append([str_gp, *gp_params])
        return str_gps

    @property
    def display_patterns_as_df(self) -> pd.DataFrame:
        if not self._gradual_patterns:
            return pd.DataFrame(columns=['Pattern'])

        all_rows = []
        for gp in self._gradual_patterns:
            str_gp, gp_params = gp.print(self.titles, descriptor_title=True)
            # Create a clean dictionary for this row
            row_data = {"Pattern": str_gp}
            for param_dict in gp_params:
                row_data.update(param_dict)
            all_rows.append(row_data)
        return pd.DataFrame(all_rows)

    def _init_attributes(self, create_time_index: bool) -> None:
        """
        Initializes the attributes of the data-gp object.

        :param create_time_index: adds a time index column if none exists.

        """

        def get_attr_cols() -> np.ndarray:
            """
            Returns indices of all columns with non-datetime objects

            :return: ndarray
            """
            all_cols = np.arange(self._col_count)
            attr_cols = np.setdiff1d(all_cols, self._time_cols)
            return attr_cols

        def get_time_cols() -> np.ndarray:
            """
            Tests each column's objects for date-time values. Returns indices of all columns with date-time objects

            :return: A ndarray object containing the indices of the time columns.
            """
            # Retrieve the first column only
            time_cols = []
            n = self._col_count
            for i in range(n):  # check every column/attribute for time format
                row_data = str(self._data[0][i])
                try:
                    time_ok, _ = DataGP.test_time(row_data)
                    if time_ok:
                        time_cols.append(i)
                except ValueError:
                    continue
            return np.array(time_cols)

        self._row_count, self._col_count = self._data.shape
        self._time_cols = get_time_cols()

        # Add Dummy Time
        if self._time_cols.size == 0 and create_time_index:
            self._titles.append(NO_TIME_LABEL)
            no_time = np.arange(self._data.shape[0])
            # d.index = pd.DatetimeIndex(d.index.values, freq=d.index.inferred_freq)

            self._data = np.column_stack((self._data, no_time))
            self._time_cols = np.append(self._time_cols, [len(self._titles) - 1]).astype(int)
            self._row_count, self._col_count = self._data.shape
        self._attr_cols = get_attr_cols()

    def _compute_pairwise_bitmap( self, attr_values: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute a pairwise bitmap and its support for one attribute.

        Args:
            attr_values:
                One-dimensional numeric array containing the values of a single
                attribute.

        Returns:
            A tuple containing:

            - ``bitmap``: Boolean pairwise comparison matrix with shape
              ``(n_samples, n_samples)``.
            - ``support``: Proportion of valid object pairs satisfying the
              gradual relationship.

        Notes:
            The bitmap requires O(n²) memory, where n is the number of samples.
            Only one attribute bitmap is generated at a time to avoid the
            O(n² × m) memory requirement of constructing a bitmap tensor for all
            m attributes simultaneously.
        """
        n_samples = len(attr_values)
        pair_cnt = GP.pair_count(n_samples)

        # Pairwise comparison:
        #
        #   bitmap[i, j] = True
        #       when attr_values[j] > attr_values[i]
        #
        # This represents the positive gradual item (attribute+).
        lhs = attr_values[:, None]
        rhs = attr_values[None, :]

        with np.errstate(invalid="ignore"):
            if self._include_equal_values:
                bitmap = rhs >= lhs

                # Exclude self-comparisons.
                indices = np.arange(n_samples)
                bitmap[indices, indices] = False
            else:
                bitmap = rhs > lhs

        support = float(bitmap.sum() / pair_cnt)
        return bitmap, support

    def _compute_pairwise_bitmap_gpu(self, attr_values: np.ndarray, chunk_size: int = 2048,) -> tuple[np.ndarray, float]:

        values = torch.as_tensor(attr_values, dtype=torch.float32, device="cuda",)
        n = len(values)
        bitmap = np.empty((n, n), dtype=np.bool_,)
        total_true = 0

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)

            lhs = values[start:end, None]
            rhs = values[None, :]

            if self._include_equal_values:
                chunk = lhs <= rhs
                rows = torch.arange(start, end, device="cuda",)
                # cols = torch.arange(n,device="cuda",)
                # Remove diagonal.
                chunk[torch.arange(end - start, device="cuda"), rows] = False
            else:
                chunk = rhs > lhs

            total_true += int(chunk.sum().item())
            bitmap[start:end] = chunk.cpu().numpy()

        support = total_true / GP.pair_count(n)
        return bitmap, support

    def add_gradual_pattern(self, pattern) -> None:
        """
        Adds a gradual pattern to the list of gradual patterns.

        :param pattern: A gradual pattern
        """
        if not isinstance(pattern, (GP, TGP)):
            raise TypeError("Pattern must be of type GP, ExtGP, or TGP")

        if self._gradual_patterns is None:
            self._gradual_patterns = [pattern]
        else:
            self._gradual_patterns.append(pattern)

    def clear_gradual_patterns(self) -> None:
        """Clears the list of gradual patterns."""
        self._gradual_patterns = []

    def remove_subsets(self, gi_arr: set, gradual_patterns: list[GP] | None = None) -> None:
        """
        Remove subset GPs from the list.

        :param gi_arr: Gradual items in an array
        :param gradual_patterns: List of gradual patterns (if None, use the object's GPs)
        :return: List of GPs
        """
        gps = self._gradual_patterns if gradual_patterns is None else gradual_patterns
        if gps is None:
            return

        for gp in gps:
            result1 = set(gp.as_set).issubset(gi_arr)
            result2 = set(gp.as_swapped_set).issubset(gi_arr)
            if result1 or result2:
                gps.remove(gp)

    def fit_bitmap(self, attr_data: np.ndarray | None = None) -> None:
        """Generate valid gradual-item bitmaps one attribute at a time.

        For each selected attribute, a pairwise comparison matrix is generated
        independently. Its support is evaluated against the minimum support
        threshold. Valid positive and negative gradual items are then stored in
        ``self._valid_bins``.

        Generating attributes independently limits peak bitmap memory to a single
        ``(n_samples, n_samples)`` matrix rather than a
        ``(n_samples, n_samples, n_attributes)`` tensor.

        Args:
            attr_data:
                Optional stepped/transformed attribute data. If ``None``, the
                original dataset is transposed and used. When stepped data is
                provided, its first selected attribute determines ``_attr_size``.

        Returns:
            None.

        Notes:
            Bitmap generation has O(n²) time and memory complexity per attribute.
            Only attributes satisfying ``self._thd_supp`` are retained.
        """
        # ------------------------------------------------------------------
        # 1. Prepare attribute data
        # ------------------------------------------------------------------
        if attr_data is None:
            attr_data = self._data.T
            self._attr_size = self._row_count
        else:
            self._attr_size = len(attr_data[self._attr_cols[0]])
        self._valid_bins = {}

        # ------------------------------------------------------------------
        # 2. Generate one bitmap at a time
        # ------------------------------------------------------------------
        for col in self._attr_cols:
            # Convert only the current attribute to a numeric NumPy array.
            attr_values = np.asarray(attr_data[col], dtype=np.float64,)

            # Generate the bitmap and calculate its support.
            if self._device == "cuda":
                bin_mat, support = self._compute_pairwise_bitmap_gpu(attr_values)
            else:
                bin_mat, support = self._compute_pairwise_bitmap(attr_values)

            # Discard unsupported attributes immediately.
            if support < self._thd_supp or self._valid_bins is None:
                del bin_mat
                continue

            # --------------------------------------------------------------
            # Positive gradual item
            # --------------------------------------------------------------
            conv_bits = np.packbits(bin_mat.ravel())
            packed_bits = torch.from_numpy(conv_bits).cuda() if self._device == "cuda" else conv_bits
            self._valid_bins[f"{col}+"] = PairwiseMatrix(
                packed_bin_mat=packed_bits,
                support=support,
                pattern={f"{col}+"},
            )

            # --------------------------------------------------------------
            # Negative gradual item
            # --------------------------------------------------------------
            conv_bits_t = np.packbits(bin_mat.T.ravel())
            packed_bits_t = torch.from_numpy(conv_bits_t).cuda() if self._device == "cuda" else conv_bits_t
            self._valid_bins[f"{col}-"] = PairwiseMatrix(
                packed_bin_mat=packed_bits_t,
                support=support,
                pattern={f"{col}-"},
            )

        # ------------------------------------------------------------------
        # 3. Require a minimum number of valid gradual items
        # ------------------------------------------------------------------
        valid_bins_len = len(self._valid_bins) if self._valid_bins is not None else 0
        if valid_bins_len < 3:
            self._valid_bins = None

    def fit_warpingset(self) -> None:
        """
        Generates transaction ids (tids) for each column/feature with numeric objects. It stores the tids in attribute
        valid_tids (those tids whose computed support values are greater or equal to the minimum support threshold
        value).

        The method decomposes the pairwise matrix of a gradual item/pattern into a warping set. Attributes that have
        strong correlation will produce a warping set with dense zigzag patterns when plotted as a graph. Those with weak
        correlation will produce a warping set with sparse zigzag patterns.

        """

        if self._valid_bins is None:
            self.fit_bitmap()
            if self._valid_bins is None:
                return

        n = self._row_count
        self._warping_set = {}
        for gi_str, gi_data in self._valid_bins.items():
            edge_list: np.ndarray|torch.Tensor = GP.gen_gradual_warping_set(gi_data.packed_bin_mat, n)

            tids_len = edge_list.shape[0]
            supp = ((tids_len * 0.5) * (tids_len - 1) / GP.pair_count(n))
            if supp >= self._thd_supp and self._warping_set is not None:
                lst_ij = edge_list.cpu().tolist() if isinstance(edge_list, torch.Tensor) else edge_list.tolist()
                self._warping_set[gi_str] = lst_ij

    def generate_output_files(self, alg_data: dict, target_col: int | None = None, save_to_file: bool = True):
        """
        Generates output of results (as files) for the GP mining algorithm.

        :param alg_data: Dictionary of algorithm parameters.
        :param target_col: Index of the target column.
        :param save_to_file: If True, saves the output to files.
        """

        list_gp = self.gradual_patterns
        num_patterns = len(list_gp) if list_gp is not None else 0
        f_name = str(str(alg_data['Algorithm']) + '_' + str(time.time()).replace('.', '', 1))

        out_txt = ""
        for key, val in alg_data.items():
            out_txt += f"{key}: {val}\n"

        out_txt += f"No. of (dataset) attributes: {self.col_count}\n"
        out_txt += f"No. of (dataset) objects: {self.row_count} \n"
        out_txt += f"Minimum support: {self.thd_supp}\n"
        # out_txt += f"Number of cores: {num_cores}\n"
        out_txt += f"Number of patterns: {num_patterns}\n"

        out_txt += "\nAttributes:\n"
        tgt_col = target_col if target_col is not None else -1
        for i, txt in enumerate(self.titles):
            if i == tgt_col:
                out_txt += f"{i}. {txt}**\n"
            else:
                out_txt += f"{i}. {txt}\n"

        file = 'a dataframe'
        if isinstance(self._data_src, str):
            file = self._data_src
        out_txt += f"\nFile: {file}\n"
        out_txt += str("\nPattern : Support" + '\n')

        list_tgp = self.gradual_patterns
        if list_tgp is not None:
            for tgp in list_tgp:
                gp_str = f"{tgp.to_string()} :  {tgp.support}"
                if len(gp_str) > 100:
                    gp_str = gp_str[:100] + '\n' + gp_str[100:]
                out_txt += f"{gp_str}\n"
        if not save_to_file:
            print(out_txt)

        if save_to_file:
            gp_df = self.display_patterns_as_df
            gp_df.to_csv(str(f_name + '.csv'), index=False)
            write_file(out_txt, str(f_name + '.txt'), wr=True)

    @classmethod
    def save_pairwise_data(cls, data_src: pd.DataFrame | str, min_sup: float = 0.5, out_dir: str = "") -> bool:
        """
        Given a numeric dataset, this method generates all the pairwise matrices for the all the gradual items (GI)
        which are obtained from the dataset's features/columns.

        >>> import so4gp as sgp
        >>> import pandas
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> columns = ['Age', 'Salary', 'Cars', 'Expenses']
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>> sgp.save_pairwise_data(dummy_df, out_dir="mat_data")

        :param data_src: Pandas dataframe or CSV file name as a string
        :param min_sup: Minimum support value for the pairwise matrices.
        :param out_dir: Optional directory path where CSV files should be saved.

        :return: True if data saved to CSV files, False otherwise.
        """

        d_set = cls(data_src, min_sup=min_sup)
        d_set.fit_bitmap()

        if d_set._valid_bins is None:
            return False

        # Create the directory if it doesn't exist (does nothing if out_dir is "")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        for gi_str, gp_data in (d_set._valid_bins or {}).items():
            gi: GI = GI.from_string(gi_str)
            file_name = f"{gi.as_string(d_set.titles)}.csv"
            file = os.path.join(out_dir, file_name)

            # Note: Changed fmt to '%.0f' to support floats without decimal places safely
            np.savetxt(file, gp_data.packed_bin_mat, delimiter=',', fmt='%.0f')
        return True

    @classmethod
    def analyze_gps(cls, data_src: pd.DataFrame | str, min_sup: float, est_gps: list[GP], approach: str = 'bfs') -> str:
        """
        For each estimated GP, computes its true support using the GRAANK approach and returns the statistics (% error,
        and standard deviation).

        >>> import so4gp as sgp
        >>> import pandas
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> columns = ['Age', 'Salary', 'Cars', 'Expenses']
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> estimated_gps = list()
        >>> temp_gp = sgp.GP()
        >>> for gi_str in ['0+', '1-']:
        >>>    temp_gp.add_gradual_item(sgp.GI.from_string(gi_str))
        >>> temp_gp.support = 0.5
        >>> estimated_gps.append(temp_gp)
        >>> temp_gp = sgp.GP()
        >>> for gi_str in ['1+', '3-', '0+']:
        >>>    temp_gp.add_gradual_item(sgp.GI.from_string(gi_str))
        >>> temp_gp.support = 0.48
        >>> estimated_gps.append(temp_gp)
        >>> res = sgp.analyze_gps(dummy_df, min_sup=0.4, est_gps=estimated_gps, approach='bfs')
        >>> print(res)
        Gradual Pattern       Estimated Support    True Support  Percentage Error      Standard Deviation
        ['0+', '1-']                       0.5              0.4             25.0%                   0.071
        ['1+', '3-', '0+']                 0.48             0.6            -20.0%                   0.085

        :param data_src: Data set file
        :param min_sup: Minimum support (set by user)
        :param est_gps: Estimated GPs
        :param approach: 'Bfs' (default) or 'dfs'

        :return: Tabulated results
        """
        if approach == 'dfs':
            d_set = cls(data_src, min_sup)
            d_set.fit_warpingset()
        else:
            d_set = cls(data_src, min_sup)
            d_set.fit_bitmap()
        headers = ["Gradual Pattern", "Estimated Support", "True Support", "Percentage Error", "Standard Deviation"]
        data = []
        for est_gp in est_gps:
            est_sup = est_gp.support
            est_gp.support = 0
            if approach == 'dfs':
                true_gp = est_gp.validate_via_tree(d_set)
            else:
                true_gp = est_gp.validate_via_graank(d_set, target_col=None)
            true_sup = true_gp.support

            if true_sup == 0:
                percentage_error = np.inf
                st_dev = np.inf
            else:
                percentage_error = ((est_sup - true_sup) / true_sup) * 100
                st_dev = statistics.stdev([est_sup, true_sup])

            if len(true_gp.gradual_items) == len(est_gp.gradual_items):
                data.append(
                    [est_gp.to_string(), round(float(est_sup), 3), round(float(true_sup), 3),
                     str(round(float(percentage_error), 3)) + '%',
                     round(float(st_dev), 3)])
            else:
                data.append([est_gp.to_string(), round(est_sup, 3), -1, np.inf, np.inf])
        return tabulate(data, headers=headers)

    @staticmethod
    def read(data_src) -> tuple[list, np.ndarray]:
        """
        Reads all the contents of a file (in CSV format) or a data-frame. Checks if its columns have numeric values. It
        separates its column headers (titles) from the objects.

        :param data_src: A data source, it can either be a 'file in csv format' or a 'Pandas DataFrame'
        :type data_src: pd.DataFrame | str

        :return: The title, column objects
        """
        # 1. Retrieve data set from source
        if isinstance(data_src, pd.DataFrame):
            # a. DataFrame source
            # Check column names
            try:
                # Check data type
                _ = data_src.columns.astype(float)

                # Add column values
                data_src.loc[-1] = data_src.columns.to_numpy(dtype=float)  # adding a row
                data_src.index = data_src.index + 1  # shifting index
                data_src.sort_index(inplace=True)

                # Rename column names
                header_vals = ['col_' + str(k) for k in range(data_src.shape[1])]
                data_src.columns = header_vals
            except ValueError:
                pass
            except TypeError:
                pass
            # print ("Data fetched from DataFrame")
            return DataGP.clean_data(data_src)
        else:
            # b. CSV file
            file = data_src if isinstance(data_src, str) else ""
            try:
                with open(file, 'r') as f:
                    dialect = csv.Sniffer().sniff(f.readline(), delimiters=";,' '\t")
                    f.seek(0)
                    reader = csv.reader(f, dialect)
                    raw_data = list(reader)
                    f.close()

                if len(raw_data) <= 1:
                    raise FatalError("CSV file read error. File has little or no data")
                else:
                    # print ("Data fetched from CSV file")
                    # 2. Get table headers
                    keys = range(len(raw_data[0]))
                    if raw_data[0][0].replace('.', '', 1).isdigit() or raw_data[0][0].isdigit():
                        header_vals = [f'col_{k}' for k in keys]
                    else:
                        if raw_data[0][1].replace('.', '', 1).isdigit() or raw_data[0][1].isdigit():
                            header_vals = ['col_' + str(k) for k in keys]
                        else:
                            header_vals = raw_data[0]
                            del raw_data[0]
                    d_frame = pd.DataFrame(raw_data, columns=header_vals)
                    return DataGP.clean_data(d_frame)
            except Exception as error:
                raise FatalError("Error: " + str(error))

    @staticmethod
    def test_time(date_str: str) -> tuple[bool, float | None]:
        """
        Tests if a str represents a date-time variable.

        :param date_str: A string
        :return: bool (True if it is a date-time variable, False otherwise)
        """
        # add all the possible formats
        # Exclude numeric values
        try:
            int(date_str)
            return False, None
        except ValueError:
            pass

        try:
            float(date_str)
            return False, None
        except ValueError:
            pass

        try:
            dt = parse(date_str)
            return True, time.mktime(dt.timetuple())
        except ValueError as exc:
            raise ValueError("No valid date-time format found.") from exc

    @staticmethod
    def clean_data(df) -> tuple[list, np.ndarray]:
        """
        Cleans a data frame by removing missing values and non-numeric columns,
        while preserving valid time columns.

        Args:
            df: Input pandas DataFrame.

        Returns:
            A tuple containing:

            - List of column names.
            - NumPy array of the cleaned data.

        Raises:
            Exception:
                If the cleaned dataset contains no remaining columns or rows.
        """
        # 1. Remove rows containing missing values
        df = df.dropna()

        # 2. Remove non-numeric columns (except valid time columns)
        cols_to_remove = []

        for col in df.columns:
            series = df[col]

            try:
                series.astype(float)
            except ValueError:
                # Preserve valid time columns
                first_valid_idx = series.first_valid_index()
                if first_valid_idx is None:
                    cols_to_remove.append(col)
                    continue

                sample_val = str(series.at[first_valid_idx])

                try:
                    is_time, _ = DataGP.test_time(sample_val)
                    if not is_time:
                        cols_to_remove.append(col)
                except ValueError:
                    cols_to_remove.append(col)

            except TypeError:
                cols_to_remove.append(col)

        # Keep only numeric and valid time columns
        df = df.drop(columns=cols_to_remove)

        # 3. Return titles and data
        if df.empty:
            raise ValueError("Dataset is empty after cleaning.")

        return list(df.columns), df.to_numpy()
