import numpy as np
from .utils import timing, logger, get_test_file, read_sarwing_owi, geo_dir_to_xtrack
from xsarsea.windspeed.models import get_model

@timing
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
