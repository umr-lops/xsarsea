import os
import xarray as xr
import pickle as pkl
from ..xsarsea import logger
import numpy as np
from .utils import register_cmod


def load_sarwing_lut(pathin):
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



def register(path):
    register_cmod('cmodms1ahw', inc_range=[17., 50.], wspd_range=[3., 80.])(
        os.path.join(path, 'GMF_cmodms1ahw'))
    register_cmod('cmod5', inc_range=[17., 50.], wspd_range=[0.2, 50.], phi_range=[0., 180.])(
        os.path.join(path, 'GMF_cmod5'))
    register_cmod('cmod5n', inc_range=[17., 50.], wspd_range=[0.2, 50.], phi_range=[0., 180.])(
        os.path.join(path, 'GMF_cmod5n'))
    register_cmod('cmod5h', inc_range=[17., 50.], wspd_range=[0.2, 50.], phi_range=[0., 180.])(
        os.path.join(path, 'GMF_cmod5h'))
