"""
Ressources TODO

Combined Co- and Cross-Polarized SAR Measurements Under Extreme Wind Conditions
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2006JC003743

"""
import numpy as np
import xarray as xr
import pickle
import os
from numba import vectorize, float64, complex64
from numba.experimental import jitclass


try:
    from xsar.utils import timing
except ImportError:
    # null decorator
    def timing(func):
        return func

# CONSTANTS
du = 2
dv = 2
dsig = 0.1
du10_fg = 2
my_luts_co = "/home/vincelhx/Documents/ifremer/data/gmfs/lut_co.nc"
my_luts_cr = "/home/vincelhx/Documents/ifremer/data/gmfs/lut_cr.nc"


class WindInversion:
    """
    WindInversion class
    """

    def __init__(self, ds_xsar):
        """

        Parameters
        ----------
        ds_xsar: xarray.Dataset
            Dataset with dims `('pol','atrack','xtrack')`.

        Returns
        -------
        """
        self.ds_xsar = ds_xsar

        self.neszcr_mean = self.ds_xsar.nesz.isel(
            pol=1).mean(axis=0, skipna=True)

        self._spatial_dims = ['atrack', 'xtrack']
        self.lut_co = xr.open_dataset(my_luts_co)
        self.lut_cr = xr.open_dataset(my_luts_cr)
        # self.lut_co, self.lut_cr = get_LUTs()

        # LOAD LUTS VARIABLES
        self.lut_co_zon = self.lut_co['zon'].values
        self.lut_co_mer = self.lut_co['mer'].values
        self.lut_co_spd = self.lut_co['spd'].values
        self.lut_co_nrcs = self.lut_co['sigma0'].values
        self.lut_co_inc = self.lut_co["incidence"].values

        self.lut_cr_spd = self.lut_cr['wspd'].values
        self.lut_cr_nrcs = self.lut_cr['sigma0'].values
        self.lut_cr_inc = self.lut_cr["incidence"].values

        # GET ANCILLARY WIND
        self.get_ancillary_wind()

    def get_ancillary_wind(self):
        """

        Parameters
        ----------

        Returns
        -------
        """
        self.ds_xsar['ancillary_wind_azi'] = np.sqrt(
            self.ds_xsar["ecmwf_0100_1h_U10"] ** 2 +
            self.ds_xsar["ecmwf_0100_1h_V10"] ** 2
        ) * np.exp(1j * (np.arctan2(self.ds_xsar["ecmwf_0100_1h_U10"], self.ds_xsar["ecmwf_0100_1h_V10"]) - np.deg2rad(
            self.ds_xsar['ground_heading'])))

        self.ds_xsar['ancillary_wind_azi'].attrs['comment'] = """
        Ancillary wind, as a complex number.
        complex angle is the wind direction relative to azimuth (atrack)
        module is windspeed
        real part is atrack wind component
        imag part is xtrack wind component
        """
        self.ds_xsar['ancillary_wind_antenna'] = np.imag(self.ds_xsar['ancillary_wind_azi']) + 1j * np.real(
            self.ds_xsar['ancillary_wind_azi'])

    def perform_noise_flattening_1row(self, nesz_row, incid_row, display=False):
        """

        Parameters
        ----------
        nesz_row: xarray.DataArray
            DataArray with dims `('xtrack')`.
        incid_row: xarray.DataArray
            DataArray with dims `('xtrack')`.
        display : boolean

        Returns
        -------
        xarray.DataArray:
            DataArray with dims `('xtrack')`.
        """
        nesz_flat = nesz_row.copy()
        # replacing nan values by nesz mean value for concerned incidence
        nesz_flat[np.isnan(nesz_flat)] = self.neszcr_mean[np.isnan(nesz_flat)]

        noise = 10*np.log10(nesz_flat)

        try:
            _coef = np.polyfit(incid_row[np.isfinite(noise)],
                               noise[np.isfinite(noise)], 1)
        except Exception as e:
            print(e)
            return np.full(nesz_row.shape, np.nan)

        nesz_flat = 10.**((incid_row*_coef[0] + _coef[1] - 1.0)/10.)

        if display:
            # TODO USEFUL ?
            None
        return nesz_flat

    def perform_noise_flattening(self, nesz_cr, incid):
        """

        Parameters
        ----------
        nesz_cr: xarray.DataArray
            DataArray with dims `('atrack','xtrack')`.
        incid: xarray.DataArray
            DataArray with dims `('atrack','xtrack')`.

        Returns
        -------
        xarray.DataArray
            DataArray with dims `('atrack','xtrack')`.
        """

        return xr.apply_ufunc(self.perform_noise_flattening_1row,
                              nesz_cr,
                              incid,
                              input_core_dims=[["xtrack"], ["xtrack"]],
                              output_core_dims=[["xtrack"]],
                              dask='parallelized',
                              vectorize=True)

    def get_wind_from_cost_function(self, J, lut):
        """

        # TODO We can delete this function on long term perspective // or make it usable for numba
        Parameters
        ----------
        J: numpy.ndarray
        lut : numpy.ndarray with a dimension that depends on lut.

        Returns
        -------
        numpy.ndarray (length 1)

        """
        ind = np.where(J == J.min())
        if len(ind[0]) == 1:
            return lut[ind]
        elif len(ind[0]) > 1:
            # sometimes there are 2 minimums (indices could be 0 & 360, so its could be due to LUTS) :
            # we take the first indice
            return lut[np.array([ind[0][0]]), np.array([ind[1][0]])]
        else:
            return np.array([np.nan])

    def perform_copol_inversion_1pt(self, sigco, theta, ancillary_wind):
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

        Jwind = ((self.lut_co_zon-mu)/du)**2 + \
            ((self.lut_co_mer-mv)/dv)**2

        Jsig = ((self.lut_co_nrcs[np.argmin(
            np.abs(self.lut_co_inc-theta)), :, :]-sigco)/dsig)**2

        Jfinal = Jwind+Jsig

        return self.get_wind_from_cost_function(Jfinal, self.lut_co_spd)

    @vectorize([float64(float64, float64, complex64)], forceobj=False, nopython=True, target="parallel")
    def perform_copol_inversion_1pt_guvect(self, sigco, incid, ancillary_wind):
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

        arg = np.argmin(np.abs(self.lut_co_inc-incid))

        Jwind = ((self.lut_co_zon-mu)/du)**2 + \
            ((self.lut_co_mer-mv)/dv)**2
        lut_ncrs__ = self.lut_co_nrcs[arg, :, :]
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

        return self.lut_co_spd[i_min, j_min]

    def perform_crpol_inversion_1pt(self, sigcr, incid, dsig=0.1):
        """

        Parameters
        ----------
        sigcr: float64
        incid: float64
        ancillary_wind: complex64

        Returns
        -------
        float64
        """
        index_cp_inc = np.argmin(abs(self.lut_cr_inc-incid))

        if np.isfinite(sigcr) == False:
            sigcr = np.nan

        Jsig_mouche = ((self.lut_cr_nrcs[index_cp_inc, :]-sigcr)/dsig)**2
        J = Jsig_mouche
        ind = np.where(J == np.nanmin(J))
        if len(ind[0]) == 1:
            return self.lut_cr_spd[ind]
        elif len(ind[0]) > 1:
            # sometimes there are 2 minimums (indices could be 0 & 360, so its could be due to LUTS) :
            # we take the first indice
            return self.lut_cr_spd[np.array([ind[0]])]
        else:
            return np.array([np.nan])

    def perform_dualpol_inversion_1pt(self, sigco, sigcr, nesz_cr,  incid, ancillary_wind):
        if np.isnan(sigco) or np.isneginf(sigco) or np.isnan(sigcr) or np.isneginf(sigcr) or np.isnan(nesz_cr) or np.isneginf(nesz_cr) or np.isnan(ancillary_wind):
            return np.nan
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
        index_cp_inc = np.argmin(abs(self.lut_cr_inc-incid))

        wsp_first_guess = self.perform_copol_inversion_1pt(
            sigco, incid, ancillary_wind, du, dv, dsig)
        J_wind = ((self.lut_cr_spd-wsp_first_guess)/du10_fg)**2.

        # code sarwing
        try:
            nrcslin = 10.**(sigcr/10.)
            dsigcrpol = 1./(1.25/(nrcslin/nesz_cr))**4.
            # code alx
            # dsigcrpol = (1./((10**(sigcr/10))/nesz_cr))**2.
            # J_sigcrpol2 = ((sigma0_cp_LUT2[index_cp_inc, :]-sigcr)/dsigcrpol)**2
            J_sigcrpol2 = (
                (self.lut_cr_nrcs[index_cp_inc, :]-sigcr)*dsigcrpol)**2
        except Exception as e:
            print(e)
            return np.nan

        J_final2 = J_sigcrpol2 + J_wind

        wsp_mouche = self.get_wind_from_cost_function(
            J_final2, self.lut_cr_spd)
        if (wsp_mouche < 5 or wsp_first_guess < 5 or np.isnan(wsp_mouche)):
            return wsp_first_guess

        return wsp_mouche

    @vectorize([float64(float64, float64, float64, float64, complex64)], forceobj=False, nopython=True, target="parallel")
    def perform_dualpol_inversion_1pt_guvect(self, sigco, sigcr, nesz_cr, incid, ancillary_wind):
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
        arg = np.argmin(np.abs(self.lut_co_inc-incid))
        Jwind = ((self.lut_co_zon-mu)/du)**2 + \
            ((self.lut_co_mer-mv)/dv)**2
        lut_ncrs__ = self.lut_co_nrcs[arg, :, :]
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
        wsp_first_guess = self.lut_co_spd[i_min, j_min]

        index_cp_inc = np.argmin(np.abs(self.lut_cr_inc-incid))
        J_wind = ((self.lut_cr_spd-wsp_first_guess)/du10_fg)**2.

        nrcslin = 10.**(sigcr/10.)
        dsigcrpol = 1./(1.25/(nrcslin/nesz_cr))**4.
        J_sigcrpol2 = (
            (self.lut_cr_nrcs[index_cp_inc, :]-sigcr)*dsigcrpol)**2

        J_final2 = J_sigcrpol2 + J_wind

        min__ = np.where(J_final2 == np.min(J_final2))[0][0]

        wsp_mouche = self.lut_cr_spd[min__]

        if (wsp_mouche < 5 or wsp_first_guess < 5 or np.isnan(wsp_mouche)):
            return wsp_first_guess
        return wsp_mouche

    @ timing
    def perform_copol_inversion(self):
        """

        Parameters
        ----------

        Returns
        -------
        copol_wspd: xarray.DataArray
            DataArray with dims `('atrack','xtrack')`.
        """

        return xr.apply_ufunc(self.perform_copol_inversion_1pt,
                              10*np.log10(self.ds_xsar.sigma0.isel(pol=0)
                                          ).compute(),
                              self.ds_xsar.incidence.compute(),
                              self.ds_xsar.ancillary_wind_antenna.compute(),
                              vectorize=True)

    @ timing
    def perform_crpol_inversion(self):
        """

        Parameters
        ----------

        Returns
        -------
        copol_wspd: xarray.DataArray
            DataArray with dims `('atrack','xtrack')`.
        """

        return xr.apply_ufunc(self.perform_crpol_inversion_1pt,
                              10*np.log10(self.ds_xsar.sigma0.isel(pol=1)
                                          ).compute(),
                              self.ds_xsar.incidence.compute(),
                              self.ds_xsar.ancillary_wind_antenna.compute(),
                              # dask='parallelized',
                              vectorize=True)

    @ timing
    def perform_dualpol_inversion(self):
        """
        Parameters

        ----------

        Returns
        -------
        dualpol_wspd: xarray.DataArray
            DataArray with dims `('atrack','xtrack')`.
        """

        # Perform noise_flatteing
        noise_flatened = self.perform_noise_flattening(self.ds_xsar.necz.isel(pol=1)
                                                                   .compute(),
                                                       self.ds_xsar.incidence.compute())

        # ecmwf_dir_img = ecmwf_dir-self.ds_xsar["ground_heading"]
        return xr.apply_ufunc(self.perform_dualpol_inversion_1pt,
                              10*np.log10(self.ds_xsar.sigma0.isel(pol=0)
                                          ).compute(),
                              10*np.log10(self.ds_xsar.sigma0.isel(pol=1)
                                          ).compute(),
                              noise_flatened.compute(),
                              self.ds_xsar.incidence.compute(),
                              self.ds_xsar.ancillary_wind_antenna.compute(),
                              # dask='parallelized',
                              vectorize=True)

    @ timing
    def perform_copol_inversion_guvect(self):
        """

        Parameters
        ----------

        Returns
        -------
        copol_wspd: xarray.DataArray
            DataArray with dims `('atrack','xtrack')`.
        """

        return xr.apply_ufunc(self.perform_copol_inversion_1pt_guvect,
                              10*np.log10(self.ds_xsar.sigma0.isel(pol=0)
                                          ).compute(),
                              self.ds_xsar.incidence.compute(),
                              self.ds_xsar.ancillary_wind_antenna.compute(),
                              vectorize=False)

    @ timing
    def perform_dualpol_inversion_guvect(self):
        """
        Parameters

        ----------

        Returns
        -------
        dualpol_wspd: xarray.DataArray
            DataArray with dims `('atrack','xtrack')`.
        """

        # Perform noise_flatteing
        noise_flatened = self.perform_noise_flattening(self.ds_xsar.necz.isel(pol=1)
                                                                   .compute(),
                                                       self.ds_xsar.incidence.compute())
        return xr.apply_ufunc(self.perform_dualpol_inversion_1pt_guvect,
                              10*np.log10(self.ds_xsar.sigma0.isel(pol=0)
                                          ).compute(),
                              10*np.log10(self.ds_xsar.sigma0.isel(pol=1)
                                          ).compute(),
                              noise_flatened.compute(),
                              self.ds_xsar.incidence.compute(),
                              self.ds_xsar.ancillary_wind_antenna.compute(),
                              vectorize=False)


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
