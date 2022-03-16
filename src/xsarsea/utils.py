import numpy as np
from numba import vectorize, float64, complex64
from numba.experimental import jitclass
from gmfs import *


du = 2
dv = 2
dsig = 0.1
du10_fg = 2

try:
    luts_co = "/home/vincelhx/Documents/ifremer/data/gmfs/lut_co.nc"
    luts_cr = "/home/vincelhx/Documents/ifremer/data/gmfs/lut_cr.nc"
    luts_cr_rs2_1 = "/home/vincelhx/Documents/ifremer/data/gmfs/gmfs_mouche_cr/new_lut_cr_1.nc"
    luts_cr_rs2_2 = "/home/vincelhx/Documents/ifremer/data/gmfs/gmfs_mouche_cr/new_lut_cr_2.nc"
    luts_cr_rs2_3 = "/home/vincelhx/Documents/ifremer/data/gmfs/gmfs_mouche_cr/new_lut_cr_3.nc"
    luts_cr_rs2_4 = "/home/vincelhx/Documents/ifremer/data/gmfs/gmfs_mouche_cr/new_lut_cr_4.nc"
    luts_cr_s1a_5 = "/home/vincelhx/Documents/ifremer/data/gmfs/gmfs_mouche_cr/new_lut_cr_5.nc"

except Exception as e:
    luts_co = "$HOME/.xsar/lut_co.nc"
    luts_cr = "$HOME/.xsar/lut_cr.nc"
    luts_cr_rs2_1 = "$HOME/.xsar/new_lut_cr_1.nc"
    luts_cr_rs2_2 = "$HOME/.xsar/new_lut_cr_2.nc"
    luts_cr_rs2_3 = "$HOME/.xsar/new_lut_cr_3.nc"
    luts_cr_rs2_4 = "$HOME/.xsar/new_lut_cr_4.nc"
    luts_cr_s1a_5 = "$HOME/.xsar/new_lut_cr_5.nc"


# CHANGE PATHs HERE TO CHANGE LUT
gmfs_loader = Gmfs_Loader()


lut_co_zon, lut_co_mer, lut_co_spd, lut_co_nrcs, lut_co_inc = gmfs_loader.get_LUTs_co_arrays(
    luts_co)
lut_cr_spd, lut_cr_nrcs, lut_cr_inc = gmfs_loader.get_LUTs_cr_arrays(
    luts_cr)


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
