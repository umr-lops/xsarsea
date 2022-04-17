import os
import xarray as xr
import pickle as pkl
from ..xsarsea import logger
import numpy as np

registered_sarwing_luts = {}

def _load_sarwing_lut(pathin):
    if not os.path.isdir(pathin):
        raise FileNotFoundError(pathin)

    logger.info('load sarwing lut from %s' % pathin)

    sigma0_db_path = os.path.join(pathin, 'sigma.npy')
    sigma0_db = np.ascontiguousarray(np.transpose(np.load(sigma0_db_path)))
    inc = pkl.load(open(os.path.join(pathin, 'incidence_angle.pkl'), 'rb'), encoding='iso-8859-1')
    try:
        phi, wspd = pkl.load(open(os.path.join(pathin, 'wind_speed_and_direction.pkl'), 'rb'), encoding='iso-8859-1')
    except FileNotFoundError:
        phi = None
        wspd = pkl.load(open(os.path.join(pathin, 'wind_speed.pkl'), 'rb'), encoding='iso-8859-1')

    if phi is not None:
        dims = ['wspd', 'phi', 'incidence']
        coords = {'incidence': inc, 'phi': phi, 'wspd': wspd}
    else:
        dims = ['wspd', 'incidence']
        coords = {'incidence': inc, 'wspd': wspd}

    da_sigma0_db = xr.DataArray(sigma0_db, dims=dims, coords=coords)

    da_sigma0_db.name = 'sigma0'
    da_sigma0_db.attrs['unit'] = 'dB'

    return da_sigma0_db

def lut_from_sarwing(name):
    return _load_sarwing_lut(registered_sarwing_luts[name]['sarwing_lut_path'])


def register_sarwing_lut(name, path):
    registered_sarwing_luts[name] = {}
    registered_sarwing_luts[name]['sarwing_lut_path'] = path


def register_all_sarwing_luts(topdir):
    for path in os.listdir(topdir):
        sarwing_name = os.path.basename(path)
        path = os.path.abspath(os.path.join(topdir, path))
        name = sarwing_name.replace('GMF_', 'sarwing_lut_')
        register_sarwing_lut(name, path)
