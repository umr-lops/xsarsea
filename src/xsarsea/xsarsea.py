import logging
from pkg_resources import get_distribution
import numpy as np
from xsarsea.utils import timing, get_test_file

# allow nan without warnings
# some dask warnings are still non filtered: https://github.com/dask/dask/issues/3245
np.errstate(invalid='ignore')

__version__ = get_distribution("xsarsea").version

logger = logging.getLogger("xsarsea")
logger.addHandler(logging.NullHandler())


def cmodIfr2(wind_speed, wind_dir, inc_angle, subsample=True):
    """get nrcs from wind speed, dir, and inc angle

    Parameters
    ----------
    wind_speed : float or numpy-like
        wind speed (m/s)
    wind_dir : float or numpy-like
        wind dir, relative to antenna ?? (deg)
    inc_angle : float or numpy-like
        inc angle (deg)
    subsample:
        if True (by default), and inc_angle is an xarray.DataArray with dim ('atrack','xtrack'),
        inc_angle will be subsampled, then cmodIfr2 reinterpolated to original size.
        subsampling give good quality cmod if inc_angle are close from each other.

    Returns
    -------
    float or numpy-like
        simulated nrcs (linear)
    """
    # wind_speed = vent_fictif(wind_speed)

    if subsample:
        inc_angle_ori = inc_angle

        try:
            # cmodIfr2 doesn't need high resolution.
            # lower resolution is faster
            inc_angle_lr = inc_angle.interp(
                atrack=np.linspace(inc_angle.atrack[0], inc_angle.atrack[-1], 400),
                xtrack=np.linspace(inc_angle.xtrack[0], inc_angle.xtrack[-1], 400))
        except:
            inc_angle_lr = None

        if inc_angle_lr is not None:
            inc_angle = inc_angle_lr

    C = np.zeros(26)

    # init CMOD-IFR2 coef
    C[0] = 0.0
    C[1] = -2.437597
    C[2] = -1.5670307
    C[3] = 0.3708242
    C[4] = -0.040590
    C[5] = 0.404678
    C[6] = 0.188397
    C[7] = -0.027262
    C[8] = 0.064650
    C[9] = 0.054500
    C[10] = 0.086350
    C[11] = 0.055100
    C[12] = -0.058450
    C[13] = -0.096100
    C[14] = 0.412754
    C[15] = 0.121785
    C[16] = -0.024333
    C[17] = 0.072163
    C[18] = -0.062954
    C[19] = 0.015958
    C[20] = -0.069514
    C[21] = -0.062945
    C[22] = 0.035538
    C[23] = 0.023049
    C[24] = 0.074654
    C[25] = -0.014713

    T = inc_angle
    wind = wind_speed

    tetai = (T - 36.0) / 19.0
    xSQ = tetai * tetai
    # P0 = 1.0
    P1 = tetai
    P2 = (3.0 * xSQ - 1.0) / 2.0
    P3 = (5.0 * xSQ - 3.0) * tetai / 2.0
    ALPH = C[1] + C[2] * P1 + C[3] * P2 + C[4] * P3
    BETA = C[5] + C[6] * P1 + C[7] * P2
    ang = wind_dir
    cosi = np.cos(np.deg2rad(ang))
    cos2i = 2.0 * cosi * cosi - 1.0
    tetamin = 18.0
    tetamax = 58.0
    tetanor = (2.0 * T - (tetamin + tetamax)) / (tetamax - tetamin)
    vmin = 3.0
    vmax = 25.0
    vitnor = (2.0 * wind - (vmax + vmin)) / (vmax - vmin)
    pv0 = 1.0
    pv1 = vitnor
    pv2 = 2 * vitnor * pv1 - pv0
    pv3 = 2 * vitnor * pv2 - pv1
    pt0 = 1.0
    pt1 = tetanor
    pt2 = 2 * tetanor * pt1 - pt0
    # pt3 = 2 * tetanor * pt2 - pt1
    b1 = C[8] + C[9] * pv1 \
         + (C[10] + C[11] * pv1) * pt1 \
         + (C[12] + C[13] * pv1) * pt2
    tetamin = 18.0
    tetamax = 58.0
    tetanor = (2.0 * T - (tetamin + tetamax)) / (tetamax - tetamin)
    vmin = 3.0
    vmax = 25.0
    vitnor = (2.0 * wind - (vmax + vmin)) / (vmax - vmin)
    pv0 = 1.0
    pv1 = vitnor
    pv2 = 2 * vitnor * pv1 - pv0
    pv3 = 2 * vitnor * pv2 - pv1
    pt0 = 1.0
    pt1 = tetanor
    pt2 = 2 * tetanor * pt1 - pt0
    # pt3 = 2 * tetanor * pt2 - pt1
    result = (
            C[14]
            + C[15] * pt1
            + C[16] * pt2
            + (C[17] + C[18] * pt1 + C[19] * pt2) * pv1
            + (C[20] + C[21] * pt1 + C[22] * pt2) * pv2
            + (C[23] + C[24] * pt1 + C[25] * pt2) * pv3
    )
    b2 = result

    b0 = np.power(10.0, (ALPH + BETA * np.sqrt(wind)))
    sig = b0 * (1.0 + b1 * cosi + np.tanh(b2) * cos2i)

    if subsample and inc_angle_lr is not None:
        # retinterp to original resolution
        sig = sig.interp(atrack=inc_angle_ori.atrack, xtrack=inc_angle_ori.xtrack)
    return sig

@timing
def sigma0_detrend(sigma0, inc_angle, wind_speed_gmf=10, wind_dir_gmf=45):
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
        wind dir (deg relative to antenna) used by , by default 45

    Returns
    -------
    numpy-like
        sigma0 detrend.
    """

    sigma0_gmf = cmodIfr2(wind_speed_gmf, wind_dir_gmf, inc_angle)

    return np.sqrt(sigma0 / (sigma0_gmf / np.nanmean(sigma0_gmf)))
