import os
import xarray as xr
import pickle as pkl
from .utils import logger
import numpy as np
from .models import LutModel


class SarwingLutModel(LutModel):
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
            final_dims = ['incidence', 'wspd', 'phi']
            coords = {'incidence': inc, 'phi': phi, 'wspd': wspd}
        else:
            dims = ['wspd', 'incidence']
            final_dims = ['incidence', 'wspd']
            coords = {'incidence': inc, 'wspd': wspd}

        da_sigma0_db = xr.DataArray(sigma0_db, dims=dims, coords=coords)

        da_sigma0_db.name = 'sigma0_gmf'
        da_sigma0_db.attrs['units'] = 'dB'
        da_sigma0_db.attrs['comment'] = 'from model %s' % self.name

        return da_sigma0_db.transpose(*final_dims)


def register_all_sarwing_luts(topdir):
    """
    Register all sarwing luts found under `topdir`.

    This function return nothing. See `xsarsea.windspeed.available_models` to see registered models.



    Parameters
    ----------
    topdir: str
        top dir path to sarwing luts.

    Examples
    --------
    register a subset of sarwing luts

    >>> xsarsea.windspeed.register_all_sarwing_luts(xsarsea.get_test_file('sarwing_luts_subset'))

    register all sarwing lut from ifremer path

    >>> xsarsea.windspeed.register_all_sarwing_luts('/home/datawork-cersat-public/cache/project/sarwing/GMFS/v1.6')

    Notes
    _____
    Sarwing lut can be downloaded from https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsardata/sarwing_luts

    See Also
    --------
    xsarsea.windspeed.available_models
    xsarsea.windspeed.gmfs.GmfModel.register

    """
    # TODO: polratio not handled
    for path in os.listdir(topdir):
        sarwing_name = os.path.basename(path)
        path = os.path.abspath(os.path.join(topdir, path))
        name = sarwing_name.replace('GMF_', 'sarwing_lut_')

        # guess available pols from filenames
        if os.path.exists(os.path.join(path, 'wind_speed_and_direction.pkl')):
            pol = 'VV'
        elif os.path.exists(os.path.join(path, 'wind_speed.pkl')):
            pol = 'VH'
        else:
            pol = None

        sarwing_model = SarwingLutModel(name, path, pol=pol)

