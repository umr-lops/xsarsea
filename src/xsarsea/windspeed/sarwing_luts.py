import os
import xarray as xr
import pickle as pkl
from ..xsarsea import logger
import numpy as np
from .models import Model, available_models


class SarwingLutModel(Model):
    def __init__(self, name, path, **kwargs):
        super().__init__(name, **kwargs)
        self.path = path

    def _raw_lut(self):
        if not os.path.isdir(self.path):
            raise FileNotFoundError(self.path)

        logger.info('load sarwing lut from %s' % self.path)

        sigma0_db_path = os.path.join(self.path, 'sigma.npy')
        sigma0_db = np.ascontiguousarray(np.transpose(np.load(sigma0_db_path)))
        inc = pkl.load(open(os.path.join(self.path, 'incidence_angle.pkl'), 'rb'), encoding='iso-8859-1')
        try:
            phi, wspd = pkl.load(open(os.path.join(self.path, 'wind_speed_and_direction.pkl'), 'rb'),
                                 encoding='iso-8859-1')
        except FileNotFoundError:
            phi = None
            wspd = pkl.load(open(os.path.join(self.path, 'wind_speed.pkl'), 'rb'), encoding='iso-8859-1')

        if phi is not None:
            dims = ['wspd', 'phi', 'incidence']
            coords = {'incidence': inc, 'phi': phi, 'wspd': wspd}
        else:
            dims = ['wspd', 'incidence']
            coords = {'incidence': inc, 'wspd': wspd}

        da_sigma0_db = xr.DataArray(sigma0_db, dims=dims, coords=coords)

        da_sigma0_db.name = 'sigma0'
        da_sigma0_db.attrs['units'] = 'dB'

        return da_sigma0_db


def register_all_sarwing_luts(topdir):
    # TODO: polratio not handled
    for path in os.listdir(topdir):
        sarwing_name = os.path.basename(path)
        path = os.path.abspath(os.path.join(topdir, path))
        name = sarwing_name.replace('GMF_', 'sarwing_lut_')

        # guess available pols from filenames
        if os.path.exists(os.path.join(path, 'wind_speed_and_direction.pkl')):
            pols = ['VV']
        elif os.path.exists(os.path.join(path, 'wind_speed.pkl')):
            pols = ['VH']
        else:
            pols = None

        sarwing_model = SarwingLutModel(name, path, pols=pols)
        available_models[sarwing_model.name] = sarwing_model
