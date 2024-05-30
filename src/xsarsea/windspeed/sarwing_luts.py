import os
import xarray as xr
import pickle as pkl
from .utils import logger
import numpy as np
from .models import LutModel


class SarwingLutModel(LutModel):

    _name_prefix = 'sarwing_lut_'
    _priority = 1

    def __init__(self, name, path, **kwargs):
        super().__init__(name, **kwargs)
        self.path = path

    def _raw_lut(self, **kwargs):
        if not os.path.isdir(self.path):
            raise FileNotFoundError(self.path)

        logger.info('load sarwing lut from %s' % self.path)

        sigma0_db_path = os.path.join(self.path, 'sigma.npy')
        sigma0_db = np.ascontiguousarray(np.transpose(np.load(sigma0_db_path)))
        inc = pkl.load(open(os.path.join(
            self.path, 'incidence_angle.pkl'), 'rb'), encoding='iso-8859-1')
        try:
            phi, wspd = pkl.load(open(os.path.join(self.path, 'wind_speed_and_direction.pkl'), 'rb'),
                                 encoding='iso-8859-1')
        except FileNotFoundError:
            phi = None
            wspd = pkl.load(
                open(os.path.join(self.path, 'wind_speed.pkl'), 'rb'), encoding='iso-8859-1')

        self.wspd_step = np.round(np.unique(np.diff(wspd)), decimals=2)[0]
        self.inc_step = np.round(np.unique(np.diff(inc)), decimals=2)[0]
        self.inc_range = [np.round(np.min(inc), decimals=2), np.round(
            np.max(inc), decimals=2)]
        self.wspd_range = [np.round(np.min(wspd), decimals=2), np.round(
            np.max(wspd), decimals=2)]

        if phi is not None:
            dims = ['wspd', 'phi', 'incidence']
            final_dims = ['incidence', 'wspd', 'phi']
            coords = {'incidence': inc, 'phi': phi, 'wspd': wspd}
            self.phi_step = np.round(np.unique(np.diff(phi)), decimals=2)[0]
            # low res parameters, for downsampling
            self.inc_step_lr = 1.
            self.wspd_step_lr = 0.4
            self.phi_step_lr = 2.5
            self.phi_range = [np.round(np.min(phi), decimals=2), np.round(
                np.max(phi), decimals=2)]
        else:
            dims = ['wspd', 'incidence']
            final_dims = ['incidence', 'wspd']
            coords = {'incidence': inc, 'wspd': wspd}
            # low res parameters, for downsampling. those a close to high res, as crosspol lut has quite small
            self.inc_step_lr = 1.
            self.wspd_step_lr = 0.1
            self.phi_step_lr = 1

        da_sigma0_db = xr.DataArray(sigma0_db, dims=dims, coords=coords)

        da_sigma0_db.name = 'sigma0_gmf'
        da_sigma0_db.attrs['units'] = 'dB'
        da_sigma0_db.attrs['model'] = self.name
        da_sigma0_db.attrs['resolution'] = 'high'

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
        name = sarwing_name.replace('GMF_', SarwingLutModel._name_prefix)

        # guess available pols from filenames
        if os.path.exists(os.path.join(path, 'wind_speed_and_direction.pkl')):
            pol = 'VV'
        elif os.path.exists(os.path.join(path, 'wind_speed.pkl')):
            pol = 'VH'
        else:
            pol = None

        sarwing_model = SarwingLutModel(name, path, pol=pol)


def register_one_sarwing_lut(path):
    """
    Register a single sarwing lut.

    This function return nothing. See `xsarsea.windspeed.available_models` to see registered models.

    Parameters
    ----------
    path: str
        path to sarwing lut.

    Examples
    --------
    register a single sarwing lut

    >>> xsarsea.windspeed.register_one_sarwing_lut(xsarsea.get_test_file('sarwing_luts_subset/GMF_cmodms1ahw'))

    Notes
    _____
    Sarwing lut can be downloaded from https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsardata/sarwing_luts

    See Also
    --------
    xsarsea.windspeed.available_models
    xsarsea.windspeed.gmfs.GmfModel.register

    """
    sarwing_name = os.path.basename(path)
    name = sarwing_name.replace('GMF_', SarwingLutModel._name_prefix)

    # guess available pols from filenames
    if os.path.exists(os.path.join(path, 'wind_speed_and_direction.pkl')):
        pol = 'VV'
    elif os.path.exists(os.path.join(path, 'wind_speed.pkl')):
        pol = 'VH'
    else:
        pol = None

    sarwing_model = SarwingLutModel(name, path, pol=pol)
    return sarwing_model
