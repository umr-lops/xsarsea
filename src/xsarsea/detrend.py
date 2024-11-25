import numpy as np
import xarray as xr

from xsarsea.utils import logger, timing
from xsarsea.windspeed.models import get_model


@timing(logger=logger.info)
def sigma0_detrend(
    sigma0,
    inc_angle,
    wind_speed_gmf=np.array([10.0]),
    wind_dir_gmf=np.array([45.0]),
    model="gmf_cmod5n",
):
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

    if wind_speed_gmf.ndim > 1 or wind_dir_gmf.ndim > 1:
        raise ValueError("wind_speed_gmf and wind_dir_gmf must be 0D or 1D")
    for var in [wind_speed_gmf, wind_dir_gmf]:
        if var.ndim == 1 and var.size > 1:
            raise ValueError("wind_speed_gmf and wind_dir_gmf size must be 1 or 0")

    # get model for one line (all incidences)
    """
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
    """
    sigma0_gmf_sample = model(inc_angle.isel(line=0), wind_speed_gmf, wind_dir_gmf, broadcast=True)

    for var in ["wspd", "phi", "incidence"]:
        if var in sigma0_gmf_sample.dims:
            sigma0_gmf_sample = sigma0_gmf_sample.squeeze(var)
        if var in sigma0_gmf_sample.coords:
            sigma0_gmf_sample = sigma0_gmf_sample.drop_vars(var)

    gmf_ratio_sample = sigma0_gmf_sample / np.nanmean(sigma0_gmf_sample)
    detrended = sigma0 / gmf_ratio_sample.broadcast_like(sigma0)

    detrended.attrs["comment"] = f"detrended with model {model.name}"

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
    sarwing_ds = xr.merge([sarwing_ds, xr.open_dataset(owi_file, group="owiInversionTables_UV")])
    sarwing_ds = sarwing_ds.rename_dims({"owiAzSize": "line", "owiRaSize": "sample"})
    sarwing_ds = sarwing_ds.drop_vars(["owiCalConstObsi", "owiCalConstInci"])
    sarwing_ds = sarwing_ds.assign_coords(
        {"line": np.arange(len(sarwing_ds.line)), "sample": np.arange(len(sarwing_ds.sample))}
    )

    return sarwing_ds


def dir_meteo_to_sample(meteo_dir, ground_heading):
    """
    Convert meteorological N/S direction to image convention

    Parameters
    ----------
    meteo_dir: meteorological direction in degrees north
    ground_heading: azimuth at position, in degrees north

    Returns
    -------
    np.float64
        same shape as input. angle in radian, relative to sample, anticlockwise
    """

    return np.pi / 2 - np.deg2rad(meteo_dir - ground_heading)


def dir_sample_to_meteo(sample_dir, ground_heading):
    """
    Convert image direction relative to antenna to meteorological direction

    Parameters
    ----------
    sample_dir: angle in degrees, relative to sample, anticlockwise
    ground_heading: azimuth at position, in degrees north

    Returns
    -------
    np.float64
        same shape as input. meteorological direction in degrees north
    """

    return 90 - sample_dir + ground_heading


def dir_meteo_to_oceano(meteo_dir):
    """
    Convert meteorological direction to oceanographic direction

    Parameters
    ----------
    meteo_dir: float
        Wind direction in meteorological convention (clockwise, from), ex: 0°=from north, 90°=from east

    Returns
    -------
    float
        Wind direction in oceanographic convention (clockwise, to), ex: 0°=to north, 90°=to east
    """
    oceano_dir = (meteo_dir + 180) % 360
    return oceano_dir


def dir_oceano_to_meteo(oceano_dir):
    """
    Convert oceanographic direction to meteorological direction

    Parameters
    ----------
    oceano_dir: float
        Wind direction in oceanographic convention (clockwise, to), ex: 0°=to north, 90°=to east

    Returns
    -------
    float
        Wind direction in meteorological convention (clockwise, from), ex: 0°=from north, 90°=from east
    """
    meteo_dir = (oceano_dir - 180) % 360
    return meteo_dir


def dir_to_180(angle):
    """
    Convert angle to [-180;180]

    Parameters
    ----------
    angle: float
        angle in degrees

    Returns
    -------
    float
        angle in degrees
    """
    angle_180 = (angle + 180) % 360 - 180
    return angle_180


def dir_to_360(angle):
    """
    Convert angle to [0;360]

    Parameters
    ----------
    angle: float
        angle in degrees

    Returns
    -------
    float
        angle in degrees
    """
    angle_360 = (angle + 360) % 360
    return angle_360
