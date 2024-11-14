__all__ = [
    "sigma0_detrend",
    "dir_meteo_to_sample",
    "dir_sample_to_meteo",
    "dir_meteo_to_oceano",
    "dir_oceano_to_meteo",
    "dir_to_180",
    "dir_to_360",
    "get_test_file",
    "read_sarwing_owi",
]

from xsarsea.detrend import (
    dir_meteo_to_oceano,
    dir_meteo_to_sample,
    dir_oceano_to_meteo,
    dir_sample_to_meteo,
    dir_to_180,
    dir_to_360,
    read_sarwing_owi,
    sigma0_detrend,
)
from xsarsea.utils import get_test_file

try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata
__version__ = metadata.version("xsarsea")
