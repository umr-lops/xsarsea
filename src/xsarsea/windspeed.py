"""
Ressources TODO

Combined Co- and Cross-Polarized SAR Measurements Under Extreme Wind Conditions
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2006JC003743

"""
import numpy as np
import xarray as xr
from gmfs import *
from xsarsea.utils import perform_copol_inversion_1pt_guvect, perform_dualpol_inversion_1pt_guvect


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


class WindInversion:
    """
    WindInversion class
    """

    def __init__(self, ds_xsar, path_LUT_co, path_LUT_cr):
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
        # GET LUTS
        self.lut_co_zon, self.lut_co_mer, self.lut_co_spd, self.lut_co_nrcs, self.lut_co_inc = get_LUTs_co_arrays(
            path_LUT_co)
        self.lut_cr_spd, self.lut_cr_nrcs, self.lut_cr_inc = get_LUTs_cr_arrays(
            path_LUT_cr)

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
        ind = np.where(J == J.min())
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

        return xr.apply_ufunc(perform_copol_inversion_1pt_guvect,
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
        return xr.apply_ufunc(perform_dualpol_inversion_1pt_guvect,
                              10*np.log10(self.ds_xsar.sigma0.isel(pol=0)
                                          ).compute(),
                              10*np.log10(self.ds_xsar.sigma0.isel(pol=1)
                                          ).compute(),
                              noise_flatened.compute(),
                              self.ds_xsar.incidence.compute(),
                              self.ds_xsar.ancillary_wind_antenna.compute(),
                              vectorize=False)
