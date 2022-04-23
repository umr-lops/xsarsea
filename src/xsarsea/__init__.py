__all__ = ['sigma0_detrend', 'geo_dir_to_xtrack', 'get_test_file']

from .utils import get_test_file
from .xsarsea import geo_dir_to_xtrack, sigma0_detrend, read_sarwing_owi

try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata
__version__ = metadata.version('xsarsea')

