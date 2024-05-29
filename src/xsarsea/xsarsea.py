import numpy as np
from .utils import timing, logger, get_test_file
from xsarsea.windspeed.models import get_model
import xarray as xr


@timing(logger=logger.info)
def sigma0_detrend(sigma0, inc_angle, wind_speed_gmf=10., wind_dir_gmf=45., model='gmf_cmodifr2'):
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

    # get model for one line (all incidences)
    try:
        # if using dask, model is unpicklable. The workaround is to use map_blocks
        sigma0_gmf_sample = inc_angle.isel(line=0).map_blocks(
            model, (wind_speed_gmf, wind_dir_gmf),
            template=inc_angle.isel(line=0),
            kwargs={'broadcast': True}
        )
    except AttributeError:
        # this should be the standard way
        # see https://github.com/dask/distributed/issues/3450#issuecomment-585255484
        sigma0_gmf_sample = model(inc_angle.isel(
            line=0), wind_speed_gmf, wind_dir_gmf, broadcast=True)

    gmf_ratio_sample = sigma0_gmf_sample / np.nanmean(sigma0_gmf_sample)
    detrended = sigma0 / gmf_ratio_sample.broadcast_like(sigma0)

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
    sarwing_ds = xr.merge([sarwing_ds, xr.open_dataset(
        owi_file, group='owiInversionTables_UV')])
    sarwing_ds = sarwing_ds.rename_dims(
        {'owiAzSize': 'line', 'owiRaSize': 'sample'})
    sarwing_ds = sarwing_ds.drop_vars(['owiCalConstObsi', 'owiCalConstInci'])
    sarwing_ds = sarwing_ds.assign_coords(
        {'line': np.arange(len(sarwing_ds.line)), 'sample': np.arange(len(sarwing_ds.sample))})

    return sarwing_ds


def dir_geo_to_sample(geo_dir, ground_heading):
    """
    Convert geographical N/S direction to image convention

    Parameters
    ----------
    geo_dir: geographical direction in degrees north
    ground_heading: azimuth at position, in degrees north

    Returns
    -------
    np.float64
        same shape as input. angle in radian, relative to sample, anticlockwise
    """

    return np.pi / 2 - np.deg2rad(geo_dir - ground_heading)


def dir_sample_to_geo(sample_dir, ground_heading):
    """
    Convert image direction relative to antenna to geographical direction

    Parameters
    ----------
    sample_dir: geographical direction in degrees north
    ground_heading: azimuth at position, in degrees north

    Returns
    -------
    np.float64
        same shape as input. angle in degrees, relative to sample, anticlockwise
    """

    return 90 - sample_dir + ground_heading
