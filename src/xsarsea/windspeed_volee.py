"""
Ressources TODO

Combined Co- and Cross-Polarized SAR Measurements Under Extreme Wind Conditions
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2006JC003743

"""
import numpy as np
import xarray as xr
from gmfs import *


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


@guvectorize([(float64[:], float64[:], float64[:], float64[:, :])], '(n),(m),(p)->(m,p)')
def gmf_ufunc_inc(inc_1d, phi_1d, wspd_1d, sigma0_out):
    # return sigma 0 values of cmod5n for a given incidence (°)
    for i_spd, one_wspd in enumerate(wspd_1d):
        sigma0_out[:, i_spd] = 10*np.log10(cmod5(
            one_wspd, phi_1d, inc_1d, neutral=True))


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
dsig_crpol = 0.1
du10_fg = 2


class WindInversion:
    """
    WindInversion class
    """

    def __init__(self, ds_xsar, is_rs2=False):
        """
        # TODO add LUTs_co_resolutions
        Parameters
        ----------
        ds_xsar: xarray.Dataset
            Dataset with dims `('pol','atrack','xtrack')`.

        Returns
        -------
        """
        self.ds_xsar = ds_xsar
        self.is_rs2 = is_rs2
        self._spatial_dims = ['atrack', 'xtrack']

        # GET ANCILLARY WIND
        self.get_ancillary_wind()

    def load_wind_lut_co(self, wspd_1d, phi_1d):
        SPD_LUT, PHI_LUT = np.meshgrid(wspd_1d, phi_1d)
        ZON_LUT = SPD_LUT*np.cos(np.radians(PHI_LUT))
        MER_LUT = SPD_LUT*np.sin(np.radians(PHI_LUT))

        self.phi_1d = phi_1d
        self.wspd_1d = wspd_1d

        self.lut_co_spd = SPD_LUT
        self.lut_co_zon = ZON_LUT
        self.lut_co_mer = MER_LUT

    def load_lut_cr(self, PATH_luts_cr):
        self.lut_cr_spd, self.lut_cr_nrcs, self.lut_cr_inc = get_LUTs_cr_arrays(
            PATH_luts_cr)

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
            # print(e)
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
        # TODO numba-ize
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
        # WARNING argmin get the first min not all of them
        Parameters
        ----------
        J: numpy.ndarray
        lut : numpy.ndarray with a dimension that depends on lut.

        Returns
        -------
        numpy.ndarray (length 1)

        """

        # TODO careful argmin handles only first minimum
        if J.ndim == 2:
            ind = (np.argmin(J) // J.shape[-1], np.argmin(J) % J.shape[-1])
            # print(ind, np.where(J == np.min(J)))
        elif J.ndim == 1:
            ind = (np.argmin(J) % J.shape[-1])
        return lut[ind]

        """
        else:
            print(ind)
            return np.array([np.nan])
        """

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
        if np.isnan(sigco) or np.isnan(ancillary_wind):
            return np.nan

        mu = np.real(ancillary_wind)
        mv = -np.imag(ancillary_wind)

        Jwind = ((self.lut_co_zon-mu)/du)**2 + \
            ((self.lut_co_mer-mv)/dv)**2

        Jsig = (
            (gmf_ufunc_inc(np.array([theta]), self.phi_1d, self.wspd_1d)-sigco)/dsig)**2

        Jfinal = Jwind+Jsig
        return self.get_wind_from_cost_function(Jfinal, self.lut_co_spd)

    def perform_crpol_inversion_1pt(self, sigcr, incid):
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

        if np.isnan(sigcr) or np.isfinite(sigcr) == False:
            sigcr = np.nan

        Jsig_mouche = ((self.lut_cr_nrcs[index_cp_inc, :]-sigcr)/dsig_crpol)**2
        return self.get_wind_from_cost_function(Jsig_mouche, self.lut_cr_spd)

    def perform_dualpol_inversion_1pt(self, sigco, sigcr, nesz_cr,  incid, ancillary_wind):
        if np.isnan(sigco) or np.isnan(sigcr) or np.isnan(nesz_cr) or np.isnan(ancillary_wind):
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
            sigco, incid, ancillary_wind)
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
            # print(e)
            return np.nan

        J_final2 = J_sigcrpol2 + J_wind

        wsp_mouche = self.get_wind_from_cost_function(
            J_final2, self.lut_cr_spd)
        if (wsp_mouche < 5 or wsp_first_guess < 5 or np.isnan(wsp_mouche)):
            return wsp_first_guess

        return wsp_mouche

    def perform_copol_inversion_1pt_one_iter(self, sigco, theta, ancillary_wind, phi_1d, wspd_1d):
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
        if np.isnan(sigco) or np.isnan(ancillary_wind):
            # print("there")
            return np.nan, np.nan

        lut_co_spd, lut_co_phi = np.meshgrid(wspd_1d, phi_1d)
        lut_co_zon = lut_co_spd*np.cos(np.radians(lut_co_phi))
        lut_co_mer = lut_co_spd*np.sin(np.radians(lut_co_phi))

        mu = np.real(ancillary_wind)
        mv = -np.imag(ancillary_wind)

        Jwind = ((lut_co_zon-mu)/du)**2 + \
            ((lut_co_mer-mv)/dv)**2

        Jsig = (
            (gmf_ufunc_inc(np.array([theta]), phi_1d, wspd_1d)-sigco)/dsig)**2

        Jfinal = Jwind+Jsig
        if (Jfinal.shape[1] == 0):
            print(Jwind, Jsig)
            print(phi_1d, wspd_1d)

        ind = (np.argmin(Jfinal) //
               Jfinal.shape[-1], np.argmin(Jfinal) % Jfinal.shape[-1])

        # print(ind, "=> (", lut_co_spd[ind],lut_co_phi[ind] ,")")
        return lut_co_spd[ind]  # ,lut_co_phi[ind]

    def perform_copol_inversion_1pt_iterations(self, sigco, theta, ancillary_wind):
        wspd_min = 0.2
        wspd_max = 50
        phi_min = 0
        phi_max = 360

        step_LR_phi = 1
        steps = np.geomspace(12.5, 0.1, num=4)

        wspd_1d = np.arange(wspd_min, wspd_max+steps[0], steps[0])
        phi_1d = np.arange(phi_min, phi_max+step_LR_phi, step_LR_phi)
        spd = self.perform_copol_inversion_1pt_one_iter(
            sigco, theta, ancillary_wind, phi_1d, wspd_1d)

        for idx, val in enumerate(steps[1:]):
            if (np.isnan(spd)):
                return np.nan
            wspd_1d = np.arange(spd-steps[idx], spd+steps[idx]+val, val)
            spd = self.perform_copol_inversion_1pt_one_iter(
                sigco, theta, ancillary_wind, phi_1d, wspd_1d)

        return

    def perform_dualpol_inversion_1pt_iterations(self, sigco, sigcr, nesz_cr,  incid, ancillary_wind):

        if np.isnan(sigco) or np.isnan(sigcr) or np.isnan(nesz_cr) or np.isnan(ancillary_wind):
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

        wsp_first_guess = self.perform_copol_inversion_1pt_iterations(
            sigco, incid, ancillary_wind)
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
            # print(e)
            return np.nan

        J_final2 = J_sigcrpol2 + J_wind

        wsp_mouche = self.get_wind_from_cost_function(
            J_final2, self.lut_cr_spd)
        if (wsp_mouche < 5 or wsp_first_guess < 5 or np.isnan(wsp_mouche)):
            return wsp_first_guess
        return wsp_mouche

    @ timing
    def perform_copol_inversion_iterations(self):
        """

        Parameters
        iter : booleean
        ----------

        Returns
        -------
        copol_wspd: xarray.DataArray
            DataArray with dims `('atrack','xtrack')`.
        """

        if iter:
            return xr.apply_ufunc(self.perform_copol_inversion_1pt_iterations,
                                  10*np.log10(self.ds_xsar.sigma0.isel(pol=0)
                                              ).compute(),
                                  self.ds_xsar.incidence.compute(),
                                  self.ds_xsar.ancillary_wind_antenna.compute(),
                                  vectorize=True)

    @ timing
    def perform_copol_inversion(self, wspd_1d, phi_1d):
        """

        Parameters
        iter : booleean
        ----------

        Returns
        -------
        copol_wspd: xarray.DataArray
            DataArray with dims `('atrack','xtrack')`.
        """
        self.load_wind_lut_co(wspd_1d, phi_1d)

        return xr.apply_ufunc(self.perform_copol_inversion_1pt,
                              10*np.log10(self.ds_xsar.sigma0.isel(pol=0)
                                          ).compute(),
                              self.ds_xsar.incidence.compute(),
                              self.ds_xsar.ancillary_wind_antenna.compute(),
                              vectorize=True)

    @ timing
    def perform_crpol_inversion(self, PATH_luts_cr):
        """

        Parameters
        ----------

        Returns
        -------
        copol_wspd: xarray.DataArray
            DataArray with dims `('atrack','xtrack')`.
        """

        self.load_lut_cr(PATH_luts_cr)

        return xr.apply_ufunc(self.perform_crpol_inversion_1pt,
                              10*np.log10(self.ds_xsar.sigma0.isel(pol=1)
                                          ).compute(),
                              self.ds_xsar.incidence.compute(),
                              self.ds_xsar.ancillary_wind_antenna.compute(),
                              # dask='parallelized',
                              vectorize=True)

    @ timing
    def perform_dualpol_inversion(self, wspd_1d, phi_1d, iter, PATH_luts_cr):
        """
        Parameters

        ----------

        Returns
        -------
        dualpol_wspd: xarray.DataArray
            DataArray with dims `('atrack','xtrack')`.
        """

        self.load_lut_cr(PATH_luts_cr)

        # Perform noise_flatteing
        if self.is_rs2 == False:
            # Noise flatening for s1a, s1b
            self.neszcr_mean = self.ds_xsar.nesz.isel(
                pol=1).mean(axis=0, skipna=True)
            noise_flatened = self.perform_noise_flattening(self.ds_xsar.necz.isel(pol=1)
                                                           .compute(),
                                                           self.ds_xsar.incidence.compute())
        else:
            # No noise flatening for rs2
            noise_flatened = self.ds_xsar.necz.isel(pol=1)

        self.load_wind_lut_co(wspd_1d, phi_1d)

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
    def perform_dualpol_inversion_iterations(self, iter, PATH_luts_cr):
        """
        Parameters

        ----------

        Returns
        -------
        dualpol_wspd: xarray.DataArray
            DataArray with dims `('atrack','xtrack')`.
        """

        self.load_lut_cr(PATH_luts_cr)

        # Perform noise_flatteing
        if self.is_rs2 == False:
            # Noise flatening for s1a, s1b
            self.neszcr_mean = self.ds_xsar.nesz.isel(
                pol=1).mean(axis=0, skipna=True)
            noise_flatened = self.perform_noise_flattening(self.ds_xsar.necz.isel(pol=1)
                                                           .compute(),
                                                           self.ds_xsar.incidence.compute())
        else:
            # No noise flatening for rs2
            noise_flatened = self.ds_xsar.necz.isel(pol=1)

        return xr.apply_ufunc(self.perform_dualpol_inversion_1pt_iterations,
                              10*np.log10(self.ds_xsar.sigma0.isel(pol=0)
                                          ).compute(),
                              10*np.log10(self.ds_xsar.sigma0.isel(pol=1)
                                          ).compute(),
                              noise_flatened.compute(),
                              self.ds_xsar.incidence.compute(),
                              self.ds_xsar.ancillary_wind_antenna.compute(),
                              # dask='parallelized',
                              vectorize=True)
