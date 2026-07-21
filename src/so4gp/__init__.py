from .data_gp import DataGP
from .gradual_patterns import GI
from .gradual_patterns import GP
from .gradual_patterns import TGP
from .gradual_patterns import TimeDelay
from .gradual_patterns import PairwiseMatrix

from .utils import get_num_cores
from .utils import get_slurm_cores


import pandas as pd
from functools import wraps
@wraps(DataGP.analyze_gps)
def analyze_gps(data_src: pd.DataFrame|str, min_sup: float, est_gps: list[GP], approach: str = 'bfs') -> str:
    return DataGP.analyze_gps(data_src=data_src, min_sup=min_sup, est_gps=est_gps, approach=approach)

@wraps(DataGP.save_pairwise_data)
def save_pairwise_data(data_src: pd.DataFrame|str, min_sup: float = 0.5, out_dir: str = "") -> bool:
    # Explicitly call the class-method using the class name
    return DataGP.save_pairwise_data(data_src=data_src, min_sup=min_sup, out_dir=out_dir)

# Project Details
__version__ = "0.9.4"
__title__ = f"so4gp (v{__version__})"
__author__ = "Dickson Owuor"
__credits__ = "Montpellier University"


__all__ = [
    "DataGP",
    "GI",
    "GP",
    "TGP",
    "TimeDelay",
    "PairwiseMatrix",
    "get_num_cores",
    "get_slurm_cores",
    "save_pairwise_data",
    "analyze_gps"
]
