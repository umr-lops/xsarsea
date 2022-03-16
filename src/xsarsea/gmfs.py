from numba import guvectorize, float64
#import pickle
import logging
import numpy as np
import os
import xarray as xr

from xsar.utils import timing
from gmfs_methods import *


class Gmfs_Loader:
    """
    Gmfs_Loader class
    """

    def __init__(self):

        return

    def load_lut(self, pol, tabulated, path, dims={}):
        # use match / case in python>3.1

        if pol == "co" or pol == "copol":

            if tabulated:
                lut_co_zon, lut_co_mer, lut_co_spd, lut_co_nrcs, lut_co_inc = self.get_LUTs_co_arrays(
                    path)
                self.lut_co_dict = {}
                self.lut_co_dict["lut_co_zon"] = lut_co_zon
                self.lut_co_dict["lut_co_mer"] = lut_co_mer
                self.lut_co_dict["lut_co_spd"] = lut_co_spd
                self.lut_co_dict["lut_co_nrcs"] = lut_co_nrcs
                self.lut_co_dict["lut_co_inc"] = lut_co_inc
            elif not(tabulated):
                if not(dims):
                    logging.error("needs wspd_1d, phi_1d, inc_1d")
                    # return

                self.lut_co_dict = {}
                self.lut_co_dict["lut_co_nrcs"] = self.gmf_ufunc_co(
                    dims["inc_1d"], dims["phi_1d"], dims["wspd_1d"])

                SPD_LUT, PHI_LUT = np.meshgrid(dims["wspd_1d"], dims["phi_1d"])
                ZON_LUT = SPD_LUT*np.cos(np.radians(PHI_LUT))
                MER_LUT = SPD_LUT*np.sin(np.radians(PHI_LUT))

                self.lut_co_dict['lut_co_spd'] = SPD_LUT
                self.lut_co_dict['lut_co_zon'] = ZON_LUT
                self.lut_co_dict['lut_co_mer'] = MER_LUT
                self.lut_co_dict["lut_co_inc"] = dims["inc_1d"]
            else:
                logging.ERROR("`tabulated` arg has to be boolean")
                return None

        elif pol == "cr" or pol == "crpol":
            if tabulated:
                lut_cr_spd, lut_cr_nrcs, lut_cr_inc = self.get_LUTs_cr_arrays(
                    path)
                self.lut_cr_dict = {}
                self.lut_cr_dict["lut_cr_spd"] = lut_cr_spd
                self.lut_cr_dict["lut_cr_nrcs"] = lut_cr_nrcs
                self.lut_cr_dict["lut_cr_inc"] = lut_cr_inc

            elif not(tabulated):
                if not(dims):
                    logging.ERROR("needs wspd_1d, inc_1d, fct_number")
                    return None
                self.lut_cr_dict = {}
                self.lut_cr_dict["lut_cr_nrcs"] = self.gmf_ufunc_cr(
                    dims["inc_1d"], dims["wspd_1d"], np.array([dims["fct_number"]]))
                self.lut_cr_dict['lut_cr_spd'] = dims["wspd_1d"]
                self.lut_cr_dict["lut_cr_inc"] = dims["inc_1d"]
            else:
                logging.error("`tabulated` arg has to be boolean")
                return None

        return None

    """ TABULATED LUTS  """

    def get_LUTs_co_arrays(self, path_nc):
        lut_co = self.get_LUTs_co(path_nc)
        lut_co_zon = lut_co['zon'].values
        lut_co_mer = lut_co['mer'].values
        lut_co_spd = lut_co['spd'].values
        lut_co_nrcs = lut_co['sigma0'].values
        lut_co_inc = lut_co["incidence"].values
        return lut_co_zon, lut_co_mer, lut_co_spd, lut_co_nrcs, lut_co_inc

    def get_LUTs_cr_arrays(self, path_nc):
        lut_cr = self.get_LUTs_cr(path_nc)
        lut_cr_spd = lut_cr['wspd'].values
        lut_cr_nrcs = lut_cr['sigma0'].values
        lut_cr_inc = lut_cr["incidence"].values
        return lut_cr_spd, lut_cr_nrcs, lut_cr_inc

    def get_LUTs_co(self, path_nc):
        return xr.open_dataset(path_nc).transpose('incidence', 'phi', 'wspd')

    def get_LUTs_cr(self, path_nc):
        return xr.open_dataset(path_nc)

    """ NON-TABULATED LUTS"""

    @ timing
    def save_LUT_sigma0_co(self, inc_1d, phi_1d, wspd_1d, savepath):
        # TODO add POL attribute
        sigma0_gmf = self.gmf_ufunc_co(inc_1d, phi_1d, wspd_1d)
        SPD_LUT, PHI_LUT = np.meshgrid(wspd_1d, phi_1d)

        ZON_LUT = SPD_LUT*np.cos(np.radians(PHI_LUT))
        MER_LUT = SPD_LUT*np.sin(np.radians(PHI_LUT))

        lut = xr.DataArray(sigma0_gmf, dims=['incidence', 'phi', 'wspd'],
                           coords={'incidence': inc_1d, 'phi': phi_1d, 'wspd': wspd_1d}).to_dataset(name='sigma0')

        lut['spd'] = xr.DataArray(SPD_LUT, dims=['phi', 'wspd'])
        lut['zon'] = xr.DataArray(ZON_LUT, dims=['phi', 'wspd'])
        lut['mer'] = xr.DataArray(MER_LUT, dims=['phi', 'wspd'])

        # lut_co.attrs["pol"] = pol
        lut.attrs["LUT used"] = savepath.split("/")[-1]
        if os.path.exists(savepath):
            os.remove(savepath)
        lut.to_netcdf(savepath, format="NETCDF4")

    @ timing
    def save_LUT_sigma0_cr(self, inc_1d, wspd_1d, savepath, fct_number):
        # TODO add POL attribute
        import os
        sigma0_gmf = self.gmf_ufunc_cr(inc_1d, wspd_1d, fct_number)

        lut = xr.DataArray(sigma0_gmf, dims=['incidence', 'wspd'],
                           coords={'incidence': inc_1d, 'wspd': wspd_1d}).to_dataset(name='sigma0')

        lut.attrs["LUT used"] = savepath.split("/")[-1]
        if os.path.exists(savepath):
            os.remove(savepath)
        lut.to_netcdf(savepath, format="NETCDF4")

    def gmf_ufunc_cr(self, inc_1d, wspd_1d, fct_name):
        return 10*np.log10(gmf_ufunc_cr(inc_1d, wspd_1d, fct_name))

    def gmf_ufunc_co(self, inc_1d, phi_1d, wspd_1d):
        return 10*np.log10(gmf_ufunc_co(inc_1d, phi_1d, wspd_1d))

    def gmf_ufunc_co_inc(inc_1d, phi_1d, wspd_1d):
        # return sigma 0 values of cmod5n for a given incidence (°)
        return gmf_ufunc_co_inc(inc_1d, phi_1d, wspd_1d)


@guvectorize([(float64[:], float64[:], float64[:], float64[:, :, :])], '(n),(m),(p)->(n,m,p)')
def gmf_ufunc_co(inc_1d, phi_1d, wspd_1d, sigma0_out):
    for i_phi, one_phi in enumerate(phi_1d):
        for i_spd, one_wspd in enumerate(wspd_1d):
            sigma0_out[:, i_phi, i_spd] = cmod5(
                one_wspd, one_phi, inc_1d, neutral=True)


@guvectorize([(float64[:], float64[:], float64[:],  float64[:, :])], '(n),(m),(p)->(n,m)')
def gmf_ufunc_cr(inc_1d, wspd_1d, fct_number, sigma0_out):
    for i_spd, one_wspd in enumerate(wspd_1d):
        # print(i_spd,one_wspd)
        sigma0_out[:, i_spd] = corresponding_gmfs[fct_number[0]](
            inc_1d, one_wspd)


@guvectorize([(float64[:], float64[:], float64[:], float64[:, :])], '(n),(m),(p)->(m,p)')
def gmf_ufunc_co_inc(inc_1d, phi_1d, wspd_1d, sigma0_out):
    # return sigma 0 values of cmod5n for a given incidence (°)
    for i_spd, one_wspd in enumerate(wspd_1d):
        sigma0_out[:, i_spd] = cmod5(
            one_wspd, phi_1d, inc_1d, neutral=True)


"""
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
"""
