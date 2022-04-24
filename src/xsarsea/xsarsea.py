import numpy as np
from .utils import timing, logger, get_test_file
from xsarsea.windspeed.models import get_model
import xarray as xr


@timing(logger=logger.info)
def sigma0_detrend(sigma0, inc_angle, wind_speed_gmf=10., wind_dir_gmf=45., model='gmf_cmodIfr2'):
    """compute `sigma0_detrend` from `sigma0` and `inc_angle`

    Parameters
    ----------
    sigma0 : numpy-like
        linear sigma0
    inc_angle : numpy-like
        incidence angle (deg). Must be same shape as `sigma0`
    wind_speed_gmf : int, optional
        wind speed (m/s) used by gmf, by default 10
    wind_dir_gmf : int, optional
        wind dir (deg relative to antenna) used, by default 45

    Returns
    -------
    numpy-like
        sigma0 detrend.
    """
    model = get_model(model)

    # get model for one atrack (all incidences)
    sigma0_gmf_xtrack = model(inc_angle.isel(atrack=0), wind_speed_gmf, wind_dir_gmf, broadcast=True)

    gmf_ratio_xtrack = sigma0_gmf_xtrack / np.nanmean(sigma0_gmf_xtrack)
    detrended = sigma0 / gmf_ratio_xtrack.broadcast_like(sigma0)

    detrended.attrs['comment'] = 'detrended with model %s' % model.name

    return detrended


def read_sarwing_owi(owi_file):
    """
    read sarwing owi file, compatible with xsar.

    Parameters
    ----------
    owi_file: str

    Returns
    -------
    xarray.Dataset
        in xsar like format
    """

    sarwing_ds = xr.open_dataset(owi_file)
    sarwing_ds = xr.merge([sarwing_ds, xr.open_dataset(owi_file, group='owiInversionTables_UV')])
    sarwing_ds = sarwing_ds.rename_dims({'owiAzSize': 'atrack', 'owiRaSize': 'xtrack'})
    sarwing_ds = sarwing_ds.drop_vars(['owiCalConstObsi', 'owiCalConstInci'])
    sarwing_ds = sarwing_ds.assign_coords(
        {'atrack': np.arange(len(sarwing_ds.atrack)), 'xtrack': np.arange(len(sarwing_ds.xtrack))})

    return sarwing_ds


def dir_geo_to_xtrack(geo_dir, ground_heading):
    """
    Convert geographical N/S direction to image convention

    Parameters
    ----------
    geo_dir: geographical direction in degrees north
    ground_heading: azimuth at position, in degrees north

    Returns
    -------
    np.float64
        same shape as input. angle in radian, relative to xtrack, anticlockwise
    """

    return np.pi / 2 - np.deg2rad(geo_dir - ground_heading)


def dir_xtrack_to_geo(xtrack_dir, ground_heading):
    """
    Convert image direction relative to antenna to geographical direction

    Parameters
    ----------
    xtrack_dir: geographical direction in degrees north
    ground_heading: azimuth at position, in degrees north

    Returns
    -------
    np.float64
        same shape as input. angle in radian, relative to xtrack, anticlockwise
    """

    return 90 - np.rad2deg(xtrack_dir) + ground_heading
