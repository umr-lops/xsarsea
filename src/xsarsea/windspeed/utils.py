import logging
import os
import warnings

import numpy as np

import yaml
try:
    from importlib_resources import files
except:
    from importlib.resources import files  # new syntaxe

import logging
logger = logging.getLogger('xsarsea.windspeed.utils')
logger.setLevel(logging.INFO)


def get_dsig_wspd(name, U_crosspol, SNR_cr):
    def alpha(U_crosspol, SNR_cr, b, c0_base, gamma, k, Umax=30):
        c0 = c0_base - gamma * SNR_cr
        x = (U_crosspol - c0)
        alpha_core = 1 / (1 + np.exp(-b * x))
        drop = 1 / (1 + np.exp((U_crosspol - Umax) * k))
        return np.clip(alpha_core * drop, 0, 1)

    if name == "dsig_wspd_rs2_v3":
        b_opt = -0.4908643753212401
        c0_opt = 16.763199934792965
        gamma_opt = 1.3891445172991084
        k_opt = 20.616914824394343

    elif name == "dsig_wspd_s1_ew_rec_v3":
        b_opt = -0.5858970325653666
        c0_opt = 16.50039320910609
        gamma_opt = 1.1032031322520397
        k_opt = 7.434663633997121

    elif name == "dsig_wspd_rcm_v3":
        b_opt = -0.7920301376936547
        c0_opt = 15.8288289109038
        gamma_opt = 0.24040294696606557
        k_opt = 0.2538177092195224

    return alpha(U_crosspol, SNR_cr, b_opt, c0_opt, gamma_opt, k_opt)


def get_dsig(name, inc, sigma0_cr, nesz_cr):
    """
    get dsig_cr value(s) by name

    Parameters
    ----------
    name: str
        dsig_cr name
    inc: xarray.DataArray
        incidence angle
    sigma0_cr : xarray.DataArray
        sigma0 in cross pol for dualpol invertion
    nesz_cr: xarray.DataArray
        nesz in cross pol

    Returns
    -------
    float | xarray.DataArray
    """
    if name == "gmf_s1_v2":
        b = 1

        def sigmoid(x, c0, c1, d0, d1):
            sig = d0 + d1 / (1 + np.exp(-c0 * (x - c1)))
            return sig

        poptsig = np.array([1.57952257, 25.61843791, 1.46852088, 1.4058646])
        c = sigmoid(inc, *poptsig)
        return 1 / np.sqrt(b * (sigma0_cr / nesz_cr) ** c)

    elif name == "gmf_rs2_v2":
        b = 1
        c = 8
        return 1 / np.sqrt(b * (sigma0_cr / nesz_cr) ** c)

    elif name == "sarwing_lut_cmodms1ahw":
        return (1.25 / (sigma0_cr / nesz_cr)) ** 4.0

    elif name == "nc_lut_cmodms1ahw":
        return (1.25 / (sigma0_cr / nesz_cr)) ** 4.0

    else:
        raise ValueError(
            "dsig names different than 'gmf_s1_v2' or 'gmf_rs2_v2' or 'sarwing_lut_cmodms1ahw' or 'nc_lut_cmodms1ahw' are not handled. You can compute your own dsig_cr."
        )


def nesz_flattening(noise, inc):
    """
    Noise flatten by polynomial fit (order 1)

    Parameters
    ----------
    noise: array-like
        linear noise array (nesz), with shape (line, sample)
    inc: array-like
        incidence array

    Returns
    -------
    array-like
        flattened noise

    Examples
    --------
    Compute `dsig_cr` keyword for `xsarsea.windspeed.invert_from_model`

    >>> nesz_flat = nesz_flattening(nesz_cr, inc)
    >>> dsig_cr = (1.25 / (sigma0_cr / nesz_flat)) ** 4.0

    See Also
    --------
    xsarsea.windspeed.invert_from_model

    """

    if noise.ndim != 2:
        raise IndexError("Only 2D noise allowed")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*empty.*", category=RuntimeWarning)
        noise_mean = np.nanmean(noise, axis=0)

    try:
        # unlike np.mean, np.nanmean return a numpy array, even if noise is an xarray
        # if this behaviour change in the future, we want to be sure to have a numpy array
        noise_mean = noise_mean.values
    except AttributeError:
        pass

    def _noise_flattening_1row(noise_row, inc_row):

        noise_flat = noise_row.copy()

        # replacing nan values by nesz mean value for concerned incidence
        noise_flat[np.isnan(noise_flat)] = noise_mean[np.isnan(noise_flat)]

        # to dB
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            noise_db = 10.0 * np.log10(noise_flat)

        try:
            _coef = np.polyfit(
                inc_row[np.isfinite(noise_db)], noise_db[np.isfinite(noise_db)], 1)
        except TypeError:
            # noise is all nan
            return np.full(noise_row.shape, np.nan)

        # flattened, to linear
        noise_flat = 10.0 ** ((inc_row * _coef[0] + _coef[1] - 1.0) / 10.0)

        return noise_flat

    # incidence is almost constant along line dim, so we can make it 1D
    return np.apply_along_axis(_noise_flattening_1row, 1, noise, np.nanmean(inc, axis=0))
