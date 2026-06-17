from .data_gp import DataGP
from .gradual_patterns import GI
from .gradual_patterns import GP
from .gradual_patterns import TGP
from .gradual_patterns import TimeDelay
from .gradual_patterns import PairwiseMatrix

from .utils import get_num_cores
from .utils import get_slurm_cores


from functools import wraps
@wraps(DataGP.analyze_gps)
def analyze_gps(*args, **kwargs):
    return DataGP.analyze_gps(*args, **kwargs)

@wraps(DataGP.save_pairwise_data)
def save_pairwise_data(*args, **kwargs):
    return DataGP.save_pairwise_data(*args, **kwargs)

# Project Details
__version__ = "0.9.0"
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
