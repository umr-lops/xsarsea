import numpy as np
from numba import vectorize, float64, complex64
from numba.experimental import jitclass
from gmfs import *
du = 2
dv = 2
dsig = 0.1
du10_fg = 2

my_luts_co = "/home/vincelhx/Documents/ifremer/data/gmfs/lut_co.nc"
my_luts_cr = "/home/vincelhx/Documents/ifremer/data/gmfs/lut_cr.nc"
my_luts_co_cmod5n_decim1___ = "/home/vincelhx/Documents/ifremer/data/gmfs/GMF_cmod5n/_cmod5n_decim1___.nc"


lut_co_zon, lut_co_mer, lut_co_spd, lut_co_nrcs, lut_co_inc = get_LUTs_co_arrays(
    my_luts_co)
lut_cr_spd, lut_cr_nrcs, lut_cr_inc = get_LUTs_cr_arrays(
    my_luts_cr)


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
        j = np.where(J[i, :] == J[i, :].min())[0][0]
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
        j = np.where(J == J.min())[0][0]
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

    min__ = np.where(J_final2 == J_final2.min())[0][0]

    wsp_mouche = lut_cr_spd[min__]

    if (wsp_mouche < 5 or wsp_first_guess < 5 or np.isnan(wsp_mouche)):
        return wsp_first_guess
    return wsp_mouche
