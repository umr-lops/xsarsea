import numpy as np
from numba import vectorize, float64, complex64
from numba.experimental import jitclass
from gmfs import *


@njit
def cmod5(u10, phi, inc, neutral=True):
    """
    """
    # Coefficients and constants
    if neutral is True:  # CMOD5.n coefficients
        c = [0., -0.6878, -0.7957, 0.338, -0.1728, 0., 0.004, 0.1103,
             0.0159, 6.7329, 2.7713, -2.2885, 0.4971, -0.725, 0.045, 0.0066,
             0.3222, 0.012, 22.7, 2.0813, 3., 8.3659, -3.3428, 1.3236,
             6.2437, 2.3893, 0.3249, 4.159, 1.693]
    else:  # CMOD5 coefficients
        c = [0., -0.688, -0.793, 0.338, -0.173, 0., 0.004, 0.111,
             0.0162, 6.34, 2.57, -2.18, 0.4, -0.6, 0.045, 0.007,
             0.33, 0.012, 22., 1.95, 3., 8.39, -3.44, 1.36, 5.35,
             1.99, 0.29, 3.80, 1.53]
    zpow = 1.6
    thetm = 40.
    thethr = 25.
    y0 = c[19]
    pn = c[20]
    a = y0 - (y0 - 1.) / pn
    b = 1. / (pn * (y0 - 1.) ** (pn - 1.))

    # Angles
    cosphi = np.cos(np.deg2rad(phi))
    x = (inc - thetm) / thethr
    x2 = x ** 2.

    # B0 term
    a0 = c[1] + c[2] * x + c[3] * x2 + c[4] * x * x2
    a1 = c[5] + c[6] * x
    a2 = c[7] + c[8] * x
    gam = c[9] + c[10] * x + c[11] * x2
    s0 = c[12] + c[13] * x
    s = a2 * u10
    a3 = 1. / (1. + np.exp(-s0))
    slts0 = s < s0
    a3[~slts0] = 1. / (1. + np.exp(-s[~slts0]))
    a3[slts0] = a3[slts0] * (s[slts0] / s0[slts0]
                             ) ** (s0[slts0] * (1. - a3[slts0]))
    b0 = (a3 ** gam) * 10. ** (a0 + a1 * u10)

    # B1 term
    b1 = c[15] * u10 * (0.5 + x - np.tanh(4. * (x + c[16] + c[17] * u10)))
    b1 = (c[14] * (1. + x) - b1) / (np.exp(0.34 * (u10 - c[18])) + 1.)

    # B2 term
    v0 = c[21] + c[22] * x + c[23] * x2
    d1 = c[24] + c[25] * x + c[26] * x2
    d2 = c[27] + c[28] * x
    v2 = (u10 / v0 + 1.)
    v2lty0 = v2 < y0
    v2[v2lty0] = a + b * (v2[v2lty0] - 1.) ** pn
    b2 = (-d1 + d2 * v2) * np.exp(-v2)

    # Sigma0 according to Fourier terms
    sig = b0 * (1. + b1 * cosphi + b2 * (2. * cosphi ** 2. - 1.)) ** zpow
    return sig


du = 2
dv = 2
dsig = 0.1
du10_fg = 2

my_luts_co = "/home/vincelhx/Documents/ifremer/data/gmfs/lut_co.nc"
my_luts_cr = "/home/vincelhx/Documents/ifremer/data/gmfs/lut_cr.nc"
my_luts_cr2 = "/home/vincelhx/Documents/ifremer/data/gmfs/gmfs_mouche_cr/new_lut_cr_1.nc"
# my_luts_co_cmod5n_decim1___ = "/home/vincelhx/Documents/ifremer/data/gmfs/GMF_cmod5n/_cmod5n_decim1___.nc"


lut_co_zon, lut_co_mer, lut_co_spd, lut_co_nrcs, lut_co_inc = get_LUTs_co_arrays(
    my_luts_co)
lut_cr_spd, lut_cr_nrcs, lut_cr_inc = get_LUTs_cr_arrays(
    my_luts_cr)

"""
wspd_min = 0.2
wspd_max = 50
wspd_step = 1
wspd_1d = np.arange(wspd_min, wspd_max+wspd_step, wspd_step)

phi_min = 0
phi_max = 360
phi_step = 1
phi_1d = np.arange(phi_min, phi_max+phi_step, phi_step)
"""


@vectorize([float64(float64, float64, complex64)], forceobj=False, nopython=True, target="parallel")
def perform_copol_inversion_1pt_guvect(sigco, incid, ancillary_wind):
    """

    Parameters
    ----------
    sigco: float64
    incid: float64
    ancillary_wind: complex64

    Returns
    -------
    float64
    """

    if np.isnan(sigco) or np.isneginf(sigco) or np.isnan(ancillary_wind):
        return np.nan

    mu = np.real(ancillary_wind)
    mv = -np.imag(ancillary_wind)

    arg = np.argmin(np.abs(lut_co_inc-incid))

    Jwind = ((lut_co_zon-mu)/du)**2 + \
        ((lut_co_mer-mv)/dv)**2

    lut_ncrs__ = lut_co_nrcs[arg, :, :]
    Jsig = ((lut_ncrs__-sigco)/dsig)**2

    J = Jwind+Jsig

    __min = 99999999
    i_min = 0
    j_min = 0

    for i in range(0, J.shape[0]):
        j = (np.argmin(J[i, :]) % J.shape[-1])
        # np.where(J[i, :] == J[i, :].min())[0][0]
        min_t = J[i, j]
        if min_t < __min:
            __min = min_t
            i_min = i
            j_min = j

    return lut_co_spd[i_min, j_min]


@vectorize([float64(float64, float64, float64, float64, complex64)], forceobj=False, nopython=True, target="parallel")
def perform_dualpol_inversion_1pt_guvect(sigco, sigcr, nesz_cr, incid, ancillary_wind):
    """

    Parameters
    ----------
    sigco: float64
    sigcr: float64
    nesz_cr: float64
    incid: float64
    ancillary_wind: complex64

    Returns
    -------
    float64
    """
    if np.isnan(sigco) or np.isneginf(sigco) or np.isnan(sigcr) or np.isneginf(sigcr) or np.isnan(nesz_cr) or np.isneginf(nesz_cr) or np.isnan(ancillary_wind):
        return np.nan

    # co pol solution
    mu = np.real(ancillary_wind)
    mv = -np.imag(ancillary_wind)
    arg = np.argmin(np.abs(lut_co_inc-incid))
    Jwind = ((lut_co_zon-mu)/du)**2 + \
        ((lut_co_mer-mv)/dv)**2
    lut_ncrs__ = lut_co_nrcs[arg, :, :]
    Jsig = ((lut_ncrs__-sigco)/dsig)**2
    J = Jwind+Jsig
    __min = 99999999
    i_min = 0
    j_min = 0
    for i in range(0, J.shape[0]):
        j = (np.argmin(J[i, :]) % J.shape[-1])
        # j = np.where(J == J.min())[0][0]
        min_t = J[i, j]
        if min_t < __min:
            __min = min_t
            i_min = i
            j_min = j
    wsp_first_guess = lut_co_spd[i_min, j_min]

    index_cp_inc = np.argmin(np.abs(lut_cr_inc-incid))
    J_wind = ((lut_cr_spd-wsp_first_guess)/du10_fg)**2.

    nrcslin = 10.**(sigcr/10.)
    dsigcrpol = 1./(1.25/(nrcslin/nesz_cr))**4.
    J_sigcrpol2 = (
        (lut_cr_nrcs[index_cp_inc, :]-sigcr)*dsigcrpol)**2

    J_final2 = J_sigcrpol2 + J_wind

    # min__ = np.where(J_final2 == J_final2.min())[0][0]
    min__ = (np.argmin(J_final2) % J_final2.shape[-1])

    wsp_mouche = lut_cr_spd[min__]

    if (wsp_mouche < 5 or wsp_first_guess < 5 or np.isnan(wsp_mouche)):
        return wsp_first_guess
    return wsp_mouche


@guvectorize([(float64[:], float64[:], float64[:], float64[:, :])], '(n),(m),(p)->(m,p)')
def gmf_ufunc_inc(inc_1d, phi_1d, wspd_1d, sigma0_out):
    # return sigma 0 values of cmod5n for a given incidence (°)
    for i_spd, one_wspd in enumerate(wspd_1d):
        sigma0_out[:, i_spd] = cmod5(
            one_wspd, phi_1d, inc_1d, neutral=True)


@vectorize([float64(float64, float64, complex64)], forceobj=False, nopython=True, target="parallel")
def perform_copol_inversion_1pt_guvect(sigco, incid, ancillary_wind):
    """

    Parameters
    ----------
    sigco: float64
    incid: float64
    ancillary_wind: complex64

    Returns
    -------
    float64
    """

    if np.isnan(sigco) or np.isneginf(sigco) or np.isnan(ancillary_wind):
        return np.nan

    mu = np.real(ancillary_wind)
    mv = -np.imag(ancillary_wind)

    arg = np.argmin(np.abs(lut_co_inc-incid))

    Jwind = ((lut_co_zon-mu)/du)**2 + \
        ((lut_co_mer-mv)/dv)**2

    lut_ncrs__ = lut_co_nrcs[arg, :, :]
    Jsig = ((lut_ncrs__-sigco)/dsig)**2

    J = Jwind+Jsig

    __min = 99999999
    i_min = 0
    j_min = 0

    for i in range(0, J.shape[0]):
        j = (np.argmin(J[i, :]) % J.shape[-1])
        # np.where(J[i, :] == J[i, :].min())[0][0]
        min_t = J[i, j]
        if min_t < __min:
            __min = min_t
            i_min = i
            j_min = j

    return lut_co_spd[i_min, j_min]


@vectorize([float64(float64, float64, float64, float64, complex64)], forceobj=False, nopython=True, target="parallel")
def perform_dualpol_inversion_1pt_guvect(sigco, sigcr, nesz_cr, incid, ancillary_wind):
    """

    Parameters
    ----------
    sigco: float64
    sigcr: float64
    nesz_cr: float64
    incid: float64
    ancillary_wind: complex64

    Returns
    -------
    float64
    """
    if np.isnan(sigco) or np.isneginf(sigco) or np.isnan(sigcr) or np.isneginf(sigcr) or np.isnan(nesz_cr) or np.isneginf(nesz_cr) or np.isnan(ancillary_wind):
        return np.nan

    # co pol solution
    mu = np.real(ancillary_wind)
    mv = -np.imag(ancillary_wind)
    arg = np.argmin(np.abs(lut_co_inc-incid))
    Jwind = ((lut_co_zon-mu)/du)**2 + \
        ((lut_co_mer-mv)/dv)**2
    lut_ncrs__ = lut_co_nrcs[arg, :, :]
    Jsig = ((lut_ncrs__-sigco)/dsig)**2
    J = Jwind+Jsig
    __min = 99999999
    i_min = 0
    j_min = 0
    for i in range(0, J.shape[0]):
        j = (np.argmin(J[i, :]) % J.shape[-1])
        # j = np.where(J == J.min())[0][0]
        min_t = J[i, j]
        if min_t < __min:
            __min = min_t
            i_min = i
            j_min = j
    wsp_first_guess = lut_co_spd[i_min, j_min]

    index_cp_inc = np.argmin(np.abs(lut_cr_inc-incid))
    J_wind = ((lut_cr_spd-wsp_first_guess)/du10_fg)**2.

    nrcslin = 10.**(sigcr/10.)
    dsigcrpol = 1./(1.25/(nrcslin/nesz_cr))**4.
    J_sigcrpol2 = (
        (lut_cr_nrcs[index_cp_inc, :]-sigcr)*dsigcrpol)**2

    J_final2 = J_sigcrpol2 + J_wind

    # min__ = np.where(J_final2 == J_final2.min())[0][0]
    min__ = (np.argmin(J_final2) % J_final2.shape[-1])

    wsp_mouche = lut_cr_spd[min__]

    if (wsp_mouche < 5 or wsp_first_guess < 5 or np.isnan(wsp_mouche)):
        return wsp_first_guess
    return wsp_mouche


# , nopython=True, target="parallel")
@vectorize([float64(float64, float64, complex64)], forceobj=True)
def perform_copol_inversion_1pt_guvect_analytique(sigco, incid, ancillary_wind):
    """

    Parameters
    ----------
    sigco: float64
    incid: float64
    ancillary_wind: complex64

    Returns
    -------
    float64
    """

    if np.isnan(sigco) or np.isneginf(sigco) or np.isnan(ancillary_wind):
        return np.nan

    mu = np.real(ancillary_wind)
    mv = -np.imag(ancillary_wind)

    Jwind = ((lut_co_zon-mu)/du)**2 + \
        ((lut_co_mer-mv)/dv)**2

    lut_ncrs__ = gmf_ufunc_inc(incid, phi_1d, wspd_1d)
    """
    lut_ncrs__ = np.empty(shape=(len(phi_1d), len(wspd_1d)))
    for i_spd, one_wspd in enumerate(wspd_1d):
        lut_ncrs__[:, i_spd] = 10*np.log10(cmod5(
            one_wspd, phi_1d, incid, neutral=True))
    """
    Jsig = ((lut_ncrs__-sigco)/dsig)**2

    J = Jwind+Jsig

    __min = 99999999
    i_min = 0
    j_min = 0

    for i in range(0, J.shape[0]):
        j = (np.argmin(J[i, :]) % J.shape[-1])
        # np.where(J[i, :] == J[i, :].min())[0][0]
        min_t = J[i, j]
        if min_t < __min:
            __min = min_t
            i_min = i
            j_min = j

    return lut_co_spd[i_min, j_min]


# , nopython=True, target="parallel")
@vectorize([float64(float64, float64, float64, float64, complex64)], forceobj=True)
def perform_dualpol_inversion_1pt_guvect_analytique(sigco, sigcr, nesz_cr, incid, ancillary_wind):
    """

    Parameters
    ----------
    sigco: float64
    sigcr: float64
    nesz_cr: float64
    incid: float64
    ancillary_wind: complex64

    Returns
    -------
    float64
    """
    if np.isnan(sigco) or np.isneginf(sigco) or np.isnan(sigcr) or np.isneginf(sigcr) or np.isnan(nesz_cr) or np.isneginf(nesz_cr) or np.isnan(ancillary_wind):
        return np.nan

    # co pol solution
    mu = np.real(ancillary_wind)
    mv = -np.imag(ancillary_wind)
    arg = np.argmin(np.abs(lut_co_inc-incid))
    Jwind = ((lut_co_zon-mu)/du)**2 + \
        ((lut_co_mer-mv)/dv)**2
    # lut_ncrs__ = lut_co_nrcs[arg, :, :]

    lut_ncrs__ = gmf_ufunc_inc(incid, phi_1d, wspd_1d)

    Jsig = ((lut_ncrs__-sigco)/dsig)**2
    J = Jwind+Jsig
    __min = 99999999
    i_min = 0
    j_min = 0
    for i in range(0, J.shape[0]):
        j = (np.argmin(J[i, :]) % J.shape[-1])
        # j = np.where(J == J.min())[0][0]
        min_t = J[i, j]
        if min_t < __min:
            __min = min_t
            i_min = i
            j_min = j
    wsp_first_guess = lut_co_spd[i_min, j_min]

    index_cp_inc = np.argmin(np.abs(lut_cr_inc-incid))
    J_wind = ((lut_cr_spd-wsp_first_guess)/du10_fg)**2.

    nrcslin = 10.**(sigcr/10.)
    dsigcrpol = 1./(1.25/(nrcslin/nesz_cr))**4.
    J_sigcrpol2 = (
        (lut_cr_nrcs[index_cp_inc, :]-sigcr)*dsigcrpol)**2

    J_final2 = J_sigcrpol2 + J_wind

    # min__ = np.where(J_final2 == J_final2.min())[0][0]
    min__ = (np.argmin(J_final2) % J_final2.shape[-1])

    wsp_mouche = lut_cr_spd[min__]

    if (wsp_mouche < 5 or wsp_first_guess < 5 or np.isnan(wsp_mouche)):
        return wsp_first_guess
    return wsp_mouche
