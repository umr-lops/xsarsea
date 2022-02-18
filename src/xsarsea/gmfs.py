from numba import guvectorize, float64
from numba import njit
import pickle
import numpy as np
import os
import xarray as xr


my_luts_co = "/home/vincelhx/Documents/ifremer/data/gmfs/lut_co.nc"
my_luts_cr = "/home/vincelhx/Documents/ifremer/data/gmfs/lut_cr.nc"
my_luts_co_cmod5n_decim1___ = "/home/vincelhx/Documents/ifremer/data/gmfs/GMF_cmod5n/_cmod5n_decim1___.nc"


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


@guvectorize([(float64[:], float64[:], float64[:], float64[:, :, :])], '(n),(m),(p)->(n,m,p)')
def gmf_ufunc(inc_1d, phi_1d, wspd_1d, sigma0_out):
    for i_phi, one_phi in enumerate(phi_1d):
        for i_spd, one_wspd in enumerate(wspd_1d):
            sigma0_out[:, i_phi, i_spd] = cmod5(
                one_wspd, one_phi, inc_1d, neutral=True)


"""
wspd_min = 0.2
wspd_max = 50
wspd_step = 1
wspd_1d = np.arange(wspd_min,wspd_max+wspd_step,wspd_step)

phi_min = 0
phi_max = 360
phi_step = 1
phi_1d = np.arange(phi_min,phi_max+phi_step,phi_step)

inc_min = 17.5
inc_max = 50
inc_step = 0.1
inc_1d = np.arange(inc_min,inc_max+inc_step,inc_step)   
"""


def create_LUT_sigma0(inc_1d, phi_1d, wspd_1d, name, savepath):
    # TODO add POL attribute
    sigma0_gmf = gmf_ufunc(inc_1d, phi_1d, wspd_1d)
    SPD_LUT, PHI_LUT = np.meshgrid(wspd_1d, phi_1d)

    ZON_LUT = SPD_LUT*np.cos(np.radians(PHI_LUT))
    MER_LUT = SPD_LUT*np.sin(np.radians(PHI_LUT))

    lut = xr.DataArray(10*np.log10(sigma0_gmf), dims=['incidence', 'phi', 'wspd'],
                       coords={'incidence': inc_1d, 'phi': phi_1d, 'wspd': wspd_1d}).to_dataset(name='sigma0')

    lut['spd'] = xr.DataArray(SPD_LUT, dims=['phi', 'wspd'])
    lut['zon'] = xr.DataArray(ZON_LUT, dims=['phi', 'wspd'])
    lut['mer'] = xr.DataArray(MER_LUT, dims=['phi', 'wspd'])

    #lut_co.attrs["pol"] = pol
    lut.attrs["LUT used"] = name
    if os.path.exists(savepath):
        os.remove(savepath)

    lut.to_netcdf(savepath, format="NETCDF4")


def get_LUTs_co(path_nc):
    return xr.open_dataset(path_nc).transpose(
        'incidence', 'phi', 'wspd')


def get_LUTs_cr(path_nc):
    return xr.open_dataset(path_nc)


def get_LUTs_co_arrays(path_nc):
    lut_co = get_LUTs_co(path_nc)
    lut_co_zon = lut_co['zon'].values
    lut_co_mer = lut_co['mer'].values
    lut_co_spd = lut_co['spd'].values
    lut_co_nrcs = lut_co['sigma0'].values
    lut_co_inc = lut_co["incidence"].values
    return lut_co_zon, lut_co_mer, lut_co_spd, lut_co_nrcs, lut_co_inc


def get_LUTs_cr_arrays(path_nc):
    lut_cr = get_LUTs_cr(path_nc)
    lut_cr_spd = lut_cr['wspd'].values
    lut_cr_nrcs = lut_cr['sigma0'].values
    lut_cr_inc = lut_cr["incidence"].values
    return lut_cr_spd, lut_cr_nrcs, lut_cr_inc


def get_LUTs(pol):

    # Register LUT_co in xarray.Dataset
    # pol = self.ds_xsar.coords["pol"].values[0]

    LUT_co_folder = "/home/vincelhx/Documents/ifremer/data/gmfs/GMF_cmod5n/"
    LUT_cr_folder = "/home/vincelhx/Documents/ifremer/data/gmfs/GMF_cmodms1ahw/"

    if pol == 'VV':
        LUT_co_path = os.path.join(LUT_co_folder, 'sigma.npy')
        sigma0_LUT = np.load(LUT_co_path)

    else:
        LUT_co_path = os.path.join(
            LUT_co_folder, 'sigma_hh_ratio_zhang2.npy')
        sigma0_LUT = np.load(LUT_co_path)

    inc_LUT = pickle.load(open(os.path.join(
        LUT_co_folder, 'incidence_angle.pkl'), 'rb'), encoding='iso-8859-1')

    phi_LUT_1d, wspd_LUT_1d = pickle.load(open(os.path.join(
        LUT_co_folder, 'wind_speed_and_direction.pkl'), 'rb'), encoding='iso-8859-1')
    phi_LUT_1d = np.concatenate(
        (phi_LUT_1d, -phi_LUT_1d[::-1][1:-1])) % 360

    sigma0_LUT = np.concatenate(
        (sigma0_LUT, sigma0_LUT[:, ::-1, :][:, 1:-1, :]), axis=1)

    WSPD_LUT, PHI_LUT = np.meshgrid(wspd_LUT_1d, phi_LUT_1d)
    ZON_LUT = WSPD_LUT*np.cos(np.radians(PHI_LUT))
    MER_LUT = WSPD_LUT*np.sin(np.radians(PHI_LUT))

    lut_co = xr.DataArray(sigma0_LUT, dims=['incidence', 'phi', 'wspd'],
                          coords={'incidence': inc_LUT, 'phi': phi_LUT_1d, 'wspd': wspd_LUT_1d}).to_dataset(
        name='sigma0')

    lut_co['spd'] = xr.DataArray(WSPD_LUT, dims=['phi', 'wspd'])
    lut_co['zon'] = xr.DataArray(ZON_LUT, dims=['phi', 'wspd'])
    lut_co['mer'] = xr.DataArray(MER_LUT, dims=['phi', 'wspd'])

    lut_co.attrs["pol"] = pol
    lut_co.attrs["LUT used"] = LUT_co_path

    # Register LUT_cr in xarray.Dataset

    # LUT CR
    LUT_cr_path = os.path.join(
        LUT_cr_folder, 'sigma.npy')
    sigma0_LUT_cr = np.load(LUT_cr_path)

    inc_LUT_cr = pickle.load(open(os.path.join(
        LUT_cr_folder, 'incidence_angle.pkl'), 'rb'), encoding='iso-8859-1')
    wsp_LUT_cr = pickle.load(open(os.path.join(
        LUT_cr_folder, 'wind_speed.pkl'), 'rb'), encoding='iso-8859-1')

    lut_cr = xr.DataArray(sigma0_LUT_cr, dims=['incidence', 'wspd'],
                          coords={'incidence': inc_LUT_cr, 'wspd': wsp_LUT_cr}).to_dataset(
        name='sigma0')

    lut_cr.attrs["LUT used"] = LUT_cr_path

    lut_co.to_netcdf(
        "/home/vincelhx/Documents/ifremer/data/gmfs/lut_co.nc")
    lut_cr.to_netcdf(
        "/home/vincelhx/Documents/ifremer/data/gmfs/lut_cr.nc")

    return lut_co, lut_cr
