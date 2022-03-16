import numpy as np
import xarray as xr
from numba import vectorize, float64, complex64, boolean, complex128
from xsar.utils import timing

from xsarsea.gmfs import gmf_ufunc_co_inc
from config import *

if inversion_parameters["inversion_method"] == 'point_by_point':
    lut_co_zon = inversion_parameters["lut_co_dict"]["lut_co_zon"]
    lut_co_mer = inversion_parameters["lut_co_dict"]["lut_co_mer"]
    lut_co_spd = inversion_parameters["lut_co_dict"]["lut_co_spd"]
    lut_co_nrcs = inversion_parameters["lut_co_dict"]["lut_co_nrcs"]
    lut_co_inc = inversion_parameters["lut_co_dict"]["lut_co_inc"]

    lut_cr_spd = inversion_parameters["lut_cr_dict"]["lut_cr_spd"]
    lut_cr_nrcs = inversion_parameters["lut_cr_dict"]["lut_cr_nrcs"]
    lut_cr_inc = inversion_parameters["lut_cr_dict"]["lut_cr_inc"]


class WindInversion:
    """
    WindInversion class
    """

    def __init__(self, params, ds_xsar=xr.Dataset()):
        """

        Parameters
        ----------
        ds_xsar: xarray.Dataset
            Dataset with dims `('pol','atrack','xtrack')`.
        params : dict of parameters
        Returns
        -------
        """
        self.params = params
        self.is_rs2 = self.params["is_rs2"]
        self.inversion_method = self.params["inversion_method"]
        if ds_xsar:
            self.ds_xsar = ds_xsar
            # Load ancillary wind
            print("...Loading ancillary wind")
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
        # Rotation de 90° pour inverser
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

        noise = 10.*np.log10(nesz_flat)

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

    @timing
    def perform_dual_inversion(self, args={}):

        if inversion_parameters["inversion_method"] == "point_by_point":
            fct = perform_dualpol_inversion_1pt_guvect

        elif self.inversion_method == "iterative":
            fct = self.perform_dualpol_iterative_inversion_1pt

        if self.ds_xsar:

            # Perform noise_flatteing
            if self.is_rs2 == False:
                # Noise flatening for s1a, s1b
                self.neszcr_mean = self.ds_xsar.nesz.isel(
                    pol=1).mean(axis=0, skipna=True)
                noise_flatened = self.perform_noise_flattening(self.ds_xsar.nesz.isel(pol=1)
                                                               .compute(),
                                                               self.ds_xsar.incidence.compute())
            else:
                # No noise flatening for rs2
                noise_flatened = self.ds_xsar.nesz.isel(pol=1)

            # mask_co = ((L2_ds.owiLandFlag.values == 1) | (np.isnan(L2_ds.owiNrcs)) | (L2_ds.owiNrcs == 0))
            # mask_cr = ((L2_ds.owiLandFlag.values == 1) | (np.isnan(L2_ds.owiNrcs_cross)) | (np.isnan(_nesz_cr_sarwing)))

            mask_co = ((self.ds_xsar.land_mask.values == 1) | (np.isnan(
                self.ds_xsar.sigma0.isel(pol=0))) | (self.ds_xsar.sigma0.isel(pol=0) == 0))
            mask_cr = ((self.ds_xsar.values == 1) | (np.isnan(
                self.ds_xsar.sigma0.isel(pol=1))) | (np.isnan(self.ds_xsar.nesz.isel(pol=1))))

            mask_dual = mask_co | mask_cr

            return xr.apply_ufunc(fct,
                                  10*np.log10(self.ds_xsar.sigma0.isel(pol=0)
                                              ).compute(),
                                  10*np.log10(self.ds_xsar.sigma0.isel(pol=1)
                                              ).compute(),
                                  noise_flatened.compute(),
                                  self.ds_xsar.incidence.compute(),
                                  self.ds_xsar.ancillary_wind_antenna.compute(),
                                  mask_dual.compute(),
                                  vectorize=False)

        else:
            return xr.apply_ufunc(fct,
                                  args["nrcs_co"],
                                  args["nrcs_cr"],
                                  args["noise_flattened"],
                                  args["inc"],
                                  args["noise_flattened"],
                                  args["ancillary_wind_antenna"],
                                  args["mask"],
                                  vectorize=False)

    def perform_dualpol_iterative_inversion_1pt(self, sigco, sigcr, nesz_cr,  incid, ancillary_wind):
        """
        Parameters
        ----------
        sigco: float64
        sigcr: float64
        nesz_cr: float64
        incid: float64
        ancillary_wind: complex128

        Returns
        -------
        float64
        """

        if np.isnan(sigco) or np.isnan(sigcr) or np.isnan(nesz_cr) or np.isnan(ancillary_wind):
            return np.nan

        index_cp_inc = np.argmin(abs(self.lut_cr_inc-incid))

        spd_co = self.perform_copol_inversion_1pt_iterative(
            sigco, incid, ancillary_wind)

        J_wind_cr = ((self.lut_cr_spd - spd_co)/du10_fg)**2.

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

        J_final2 = J_sigcrpol2 + J_wind_cr

        spd_dual = self.get_wind_from_cost_function(
            J_final2, self.lut_cr_spd)

        if (spd_dual < 5 or spd_co < 5 or np.isnan(spd_dual)):
            return spd_co
        return spd_dual

    def perform_copol_inversion_1pt_iterative(self, sigco, theta, ancillary_wind):
        return perform_copol_inversion_1pt_iterative(sigco, theta, ancillary_wind)


@vectorize([float64(float64, float64, complex128)], forceobj=True)
def perform_copol_inversion_1pt_iterative(sigco, theta, ancillary_wind):
    wspd_min = 0.2
    wspd_max = 50
    phi_min = 0
    phi_max = 360

    step_LR_phi = 1
    steps = np.geomspace(12.5, 0.1, num=4)

    wspd_1d = np.arange(wspd_min, wspd_max+steps[0], steps[0])
    phi_1d = np.arange(phi_min, phi_max+step_LR_phi, step_LR_phi)
    spd, phi = perform_copol_inversion_1pt_once(
        sigco, theta, ancillary_wind, phi_1d, wspd_1d)

    for idx, val in enumerate(steps[1:]):
        if (np.isnan(spd)):
            return np.nan
        wspd_1d = np.arange(spd-steps[idx], spd+steps[idx]+val, val)
        spd, phi = perform_copol_inversion_1pt_once(
            sigco, theta, ancillary_wind, phi_1d, wspd_1d)

    return spd


def perform_copol_inversion_1pt_once(sigco, theta, ancillary_wind, phi_1d, wspd_1d):
    """

    Parameters
    ----------
    sigco: float64
    incid: float64
    ancillary_wind: complex128

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
        (gmf_ufunc_co_inc(np.array([theta]), phi_1d, wspd_1d)-sigco)/dsig)**2

    Jfinal = Jwind+Jsig
    if (Jfinal.shape[1] == 0):
        print(Jwind, Jsig)
        print(phi_1d, wspd_1d)

    ind = (np.argmin(Jfinal) //
           Jfinal.shape[-1], np.argmin(Jfinal) % Jfinal.shape[-1])

    # print(ind, "=> (", lut_co_spd[ind],lut_co_phi[ind] ,")")
    return lut_co_spd[ind], lut_co_phi[ind]


### point-by-point 1pt ###

@vectorize([float64(float64, float64, complex128, boolean)], forceobj=False, nopython=True, target="parallel")
def perform_copol_inversion_1pt_guvect(sigco, incid, ancillary_wind, mask):
    """

    Parameters
    ----------
    sigco: float64
    incid: float64
    ancillary_wind: complex128
    mask: boolean

    Returns
    -------
    float64
    """

    if mask:
        return np.nan

    mu = np.real(ancillary_wind)
    mv = -np.imag(ancillary_wind)

    arg_inc = np.argmin(np.abs(lut_co_inc-incid))

    Jwind = ((lut_co_zon-mu)/du)**2 + \
        ((lut_co_mer-mv)/dv)**2

    lut_ncrs__ = lut_co_nrcs[arg_inc, :, :]
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


@vectorize([float64(float64, float64, float64, float64, complex128)], forceobj=False, nopython=True, target="parallel")
def perform_dualpol_inversion_1pt_guvect(sigco, sigcr, nesz_cr, incid, ancillary_wind):
    """

    Parameters
    ----------
    sigco: float64
    sigcr: float64
    nesz_cr: float64
    incid: float64
    ancillary_wind: complex128
    mask: boolean

    Returns
    -------
    float64
    """
    if np.isnan(sigco) or np.isnan(sigcr) or np.isneginf(sigco) or np.isnan(nesz_cr) or np.isnan(ancillary_wind):
        return np.nan

    # co pol solution
    mu = np.real(ancillary_wind)
    mv = -np.imag(ancillary_wind)
    arg_inc = np.argmin(np.abs(lut_co_inc-incid))
    Jwind = ((lut_co_zon-mu)/du)**2 + \
        ((lut_co_mer-mv)/dv)**2
    lut_ncrs__ = lut_co_nrcs[arg_inc, :, :]
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
