__all__ = ['sigma0_detrend', 'dir_geo_to_xtrack', 'dir_xtrack_to_geo', 'get_test_file']

from .utils import get_test_file
from .xsarsea import dir_geo_to_xtrack, dir_xtrack_to_geo, sigma0_detrend, read_sarwing_owi
from .cross_spectra_core import *
try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata
__version__ = metadata.version('xsarsea')

