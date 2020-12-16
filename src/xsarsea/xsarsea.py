__all__ = ['__version__']

import logging
from pkg_resources import get_distribution

__version__ = get_distribution('xsarsea').version

logger = logging.getLogger('xsarsea')
logger.addHandler(logging.NullHandler())
