import os
import warnings
import fsspec
import aiohttp
import zipfile
from pathlib import Path
import yaml
import time
import logging
import warnings

from importlib_resources import files


logger = logging.getLogger('xsarsea')
logger.addHandler(logging.NullHandler())


mem_monitor = True

try:
    from psutil import Process
except ImportError:
    logger.warning("psutil module not found. Disabling memory monitor")
    mem_monitor = False



def _load_config():
    """
    load config from default xsar/config.yml file or user ~/.xsar/config.yml
    Returns
    -------
    dict
    """
    user_config_file = Path('~/.xsarsea/config.yml').expanduser()
    default_config_file = files('xsarsea').joinpath('config.yml')

    if user_config_file.exists():
        config_file = user_config_file
    else:
        config_file = default_config_file

    config = yaml.load(
        config_file.open(),
        Loader=yaml.FullLoader)
    return config


def get_test_file(fname, iszip = True):
    """
    get test file from  https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsardata/
    file is unzipped if needed and extracted to `config['data_dir']`

    This function is for examples only, it should not be not used in production environments.

    Parameters
    ----------
    fname: str
        file name to get (without '.zip' extension)
    iszip: boolean
        true if file have to be unzipped ;

    Returns
    -------
    str
        path to file, relative to `config['data_dir']`

    """
    config = _load_config() 
    res_path = config['data_dir']
    base_url = 'https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsardata'
    file_url = '%s/%s.zip' % (base_url, fname)

    if not iszip:
        import urllib
        file_url = '%s/%s' % (base_url, fname)
        warnings.warn("Downloading %s" % file_url)
        urllib.request.urlretrieve(file_url,os.path.join(config['data_dir'],fname))

    else : 
        if not os.path.exists(os.path.join(res_path, fname)):
            warnings.warn("Downloading %s" % file_url)
            with fsspec.open(
                    'filecache::%s' % file_url,
                    https={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}},
                    filecache={'cache_storage': os.path.join(os.path.join(config['data_dir'], 'fsspec_cache'))}
            ) as f:

                    warnings.warn("Unzipping %s" % os.path.join(res_path, fname))
                    with zipfile.ZipFile(f, 'r') as zip_ref:
                        zip_ref.extractall(res_path)
        
    return os.path.join(res_path, fname)

def timing(logger=logger.debug):
    """provide a @timing decorator() for functions, that log time spent in it"""

    def decorator(f):
        #@wraps(f)
        def wrapper(*args, **kwargs):
            mem_str = ''
            process = None
            if mem_monitor:
                process = Process(os.getpid())
                startrss = process.memory_info().rss
            starttime = time.time()
            result = f(*args, **kwargs)
            endtime = time.time()
            if mem_monitor:
                endrss = process.memory_info().rss
                mem_str = 'mem: %+.1fMb' % ((endrss - startrss) / (1024 ** 2))
            logger(
                'timing %s : %.2fs. %s' % (f.__name__, endtime - starttime, mem_str))
            return result
        wrapper.__doc__ = f.__doc__
        return wrapper
    return decorator
