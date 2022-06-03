import numpy as np
import xarray as xr
from numba import vectorize, guvectorize, float64, boolean, complex128
from xsar.utils import timing
from xsarsea import gmfs, gmfs_methods
from xsarsea.gmfs import gmf_ufunc_co_inc
from config import *
from gmfs_methods import cmod5

lut_cr_spd = inversion_parameters["lut_cr_dict"]["lut_cr_spd"]
lut_cr_nrcs = inversion_parameters["lut_cr_dict"]["lut_cr_nrcs"]
lut_cr_inc = inversion_parameters["lut_cr_dict"]["lut_cr_inc"]

if inversion_parameters["inversion_method"] == 'point_by_point':
    lut_co_zon = inversion_parameters["lut_co_dict"]["lut_co_zon"]
    lut_co_mer = inversion_parameters["lut_co_dict"]["lut_co_mer"]
    lut_co_spd = inversion_parameters["lut_co_dict"]["lut_co_spd"]
    lut_co_nrcs = inversion_parameters["lut_co_dict"]["lut_co_nrcs"]
    lut_co_inc = inversion_parameters["lut_co_dict"]["lut_co_inc"]

if inversion_parameters["inversion_method"] == 'third':
    wspd_1d = dims["wspd_1d"]
    inc_1d = dims["inc_1d"]
    phi_1d = dims["phi_1d"]
    lut_co_spd, lut_co_phi = np.meshgrid(wspd_1d, phi_1d)
    lut_co_zon = lut_co_spd*np.cos(np.radians(lut_co_phi))
    lut_co_mer = lut_co_spd*np.sin(np.radians(lut_co_phi))


class WindInversion:
    """
    WindInversion class
    """

    def __init__(self, params, ds_xsar):
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
        self.product_type = self.params["product_type"]
        self.ds_xsar = ds_xsar
        if self.ds_xsar:
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
        complex angle is the wind direction relative to azimuth(atrack)
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
        display: boolean

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

        return nesz_flat

    def perform_noise_flattening(self, nesz_cr, incid):
        """

        Parameters
        ----------
        nesz_cr: xarray.DataArray
            DataArray with dims `('atrack', 'xtrack')`.
        incid: xarray.DataArray
            DataArray with dims `('atrack', 'xtrack')`.

        Returns
        -------
        xarray.DataArray
            DataArray with dims `('atrack', 'xtrack')`.
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
    def perform_copol_inversion(self, args={}):

        if self.inversion_method == "point_by_point":
            fct = perform_copol_inversion_1pt_guvect

        elif self.inversion_method == "iterative":
            fct = self.perform_copol_inversion_1pt_iterative

        elif self.inversion_method == "third":
            fct = self.perform_copol_inversion_met3

        if self.ds_xsar:
            mask_co = ((self.ds_xsar.land_mask.values == 1) | (np.isnan(
                self.ds_xsar.sigma0.isel(pol=0))) | (self.ds_xsar.sigma0.isel(pol=0) == 0))
            mask_cr = ((self.ds_xsar.values == 1) | (np.isnan(
                self.ds_xsar.sigma0.isel(pol=1))) | (np.isnan(self.ds_xsar.nesz.isel(pol=1))))

            mask_dual = mask_co | mask_cr

            return xr.apply_ufunc(fct,
                                  10*np.log10(self.ds_xsar.sigma0.isel(pol=0)
                                              ).compute(),
                                  self.ds_xsar.incidence.compute(),
                                  self.ds_xsar.ancillary_wind_antenna.compute(),
                                  mask_dual.compute(),
                                  vectorize=False)
        else:
            return xr.apply_ufunc(fct,
                                  args["nrcs_co"],
                                  args["inc"],
                                  args["ancillary_wind_antenna"],
                                  args["mask"],
                                  vectorize=False)

    @timing
    def perform_dual_inversion(self, args={}):

        if inversion_parameters["inversion_method"] == "point_by_point":
            fct = perform_dualpol_inversion_1pt_guvect

        elif self.inversion_method == "iterative":
            fct = self.perform_dualpol_iterative_inversion_1pt

        elif self.inversion_method == "third":
            fct = self.perform_dualpol_inversion_met3

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

            mask_cr = ((self.ds_xsar.values == 1) | (np.isnan(self.ds_xsar.sigma0.isel(pol=1))) | (
                np.isnan(self.ds_xsar.nesz.isel(pol=1)) | (self.ds_xsar.sigma0.isel(pol=1) == 0)))

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
                                  args["ancillary_wind_antenna"],
                                  args["mask"],
                                  vectorize=False)

    def perform_dualpol_iterative_inversion_1pt(self, sigco, sigcr, nesz_cr,  incid, ancillary_wind, mask):
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

        if mask:
            return np.nan

        index_cp_inc = np.argmin(abs(self.lut_cr_inc-incid))

        spd_co = self.perform_copol_inversion_1pt_iterative(
            sigco, incid, ancillary_wind, False)

        J_wind_cr = ((self.lut_cr_spd - spd_co)/du10_fg)**2.

        nrcslin = 10.**(sigcr/10.)

        dsig_cr_local = 1./(1.25/(nrcslin/nesz_cr))**4.
        lut_nrcs_inc_cr = lut_cr_nrcs[index_cp_inc, :]
        Jsig_cr = ((lut_nrcs_inc_cr-sigcr)*dsig_cr_local)**2

        J_cr = Jsig_cr + J_wind_cr

        min__ = (np.argmin(J_cr) % J_cr.shape[-1])
        spd_dual = lut_cr_spd[min__]

        if (spd_dual < 5 or spd_co < 5 or np.isnan(spd_dual)):
            return spd_co
        return spd_dual

    def perform_copol_inversion_1pt_iterative(self, sigco, theta, ancillary_wind, mask):
        return perform_copol_inversion_1pt_iterative(sigco, theta, ancillary_wind, mask)

    @guvectorize([(float64[:, :], float64[:, :], complex128[:, :], float64[:, :], boolean[:, :])], '(n,m),(n,m),(n,m),(n,m)->(n,m)', forceobj=False, nopython=True, fastmath=False)
    def perform_copol_inversion_met3(nrcs_co_2d, inc_2d, ancillary_wind_2d, mask_2d, wspd_co):
        # return sigma 0 values of cmod5n for a given incidence (°)
        for j in range(nrcs_co_2d.shape[1]):
            # constant inc
            gmf_cmod5n_2d = np.empty(shape=lut_co_spd.shape)

            mean_incidence = np.nanmean(inc_2d[:, j])
            for i_spd, one_wspd in enumerate(wspd_1d):
                gmf_cmod5n_2d[:, i_spd] = 10*np.log10(cmod5(
                    one_wspd, phi_1d, np.array([mean_incidence]), neutral=True))

            for i in range(nrcs_co_2d.shape[0]):
                if mask_2d[i, j]:
                    wspd_co[i, j] = np.nan
                else:

                    mu = np.real(ancillary_wind_2d[i, j])
                    mv = -np.imag(ancillary_wind_2d[i, j])

                    Jwind_co = ((lut_co_zon-mu)/du)**2 + \
                        ((lut_co_mer-mv)/dv)**2

                    Jsig_co = ((gmf_cmod5n_2d-nrcs_co_2d[i, j])/dsig_co)**2

                    # print(lut_co_spd.shape)

                    J_co = Jwind_co+Jsig_co
                    wspd_co[i, j] = lut_co_spd[(
                        np.argmin(J_co) // J_co.shape[-1], np.argmin(J_co) % J_co.shape[-1])]

    @guvectorize([(float64[:, :], float64[:, :], float64[:, :], float64[:, :], complex128[:, :], float64[:, :], boolean[:, :])], '(n,m),(n,m),(n,m),(n,m),(n,m),(n,m)->(n,m)', forceobj=True)
    def perform_dualpol_inversion_met3(nrcs_co_2d, nrcs_cr_2d, nesz_cr_2d, inc_2d, ancillary_wind_2d, mask_2d, wspd_dual):
        for j in range(nrcs_co_2d.shape[1]):
            # constant inc
            gmf_cmod5n_2d = np.empty(shape=lut_co_spd.shape)

            mean_incidence = np.nanmean(inc_2d[:, j])
            for i_spd, one_wspd in enumerate(wspd_1d):
                gmf_cmod5n_2d[:, i_spd] = 10*np.log10(cmod5(
                    one_wspd, phi_1d, np.array([mean_incidence]), neutral=True))

            idx_inc_cr = np.argmin(np.abs(lut_cr_inc-mean_incidence))
            lut_nrcs_inc_cr = lut_cr_nrcs[idx_inc_cr, :]

            for i in range(nrcs_co_2d.shape[0]):
                # print(mean_incidence,inc_2d[i,j])

                if mask_2d[i, j]:
                    wspd_dual[i, j] = np.nan

                else:
                    mu = np.real(ancillary_wind_2d[i, j])
                    mv = -np.imag(ancillary_wind_2d[i, j])

                    Jwind = ((lut_co_zon-mu)/du)**2 + \
                            ((lut_co_mer-mv)/dv)**2

                    Jsig = ((gmf_cmod5n_2d-nrcs_co_2d[i, j])/dsig_co)**2

                    J = Jwind+Jsig
                    spd_co = lut_co_spd[(
                        np.argmin(J) // J.shape[-1], np.argmin(J) % J.shape[-1])]

                    Jwind_cr = ((lut_cr_spd-spd_co)/du10_fg)**2.

                    nrcslin = 10.**(nrcs_cr_2d[i, j]/10.)
                    dsig_cr = 1./(1.25/(nrcslin/nesz_cr_2d[i, j]))**4.

                    Jsig_cr = ((lut_nrcs_inc_cr-nrcs_cr_2d[i, j])*dsig_cr)**2

                    J_cr = Jsig_cr + Jwind_cr

                    spd_dual = lut_cr_spd[(np.argmin(J_cr) % J_cr.shape[-1])]

                    if (spd_dual < 5 or spd_co < 5):
                        wspd_dual[i, j] = spd_co
                    wspd_dual[i, j] = spd_dual

### iterative on copol 1pt ###


@vectorize([float64(float64, float64, complex128, boolean)], forceobj=True)
def perform_copol_inversion_1pt_iterative(sigco, theta, ancillary_wind, mask):
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
            sigco, theta, ancillary_wind, phi_1d, wspd_1d, mask)

    return spd


def perform_copol_inversion_1pt_once(sigco, theta, ancillary_wind,  phi_1d, wspd_1d, mask=False):
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
        return np.nan, np.nan

    lut_co_spd, lut_co_phi = np.meshgrid(wspd_1d, phi_1d)
    lut_co_zon = lut_co_spd*np.cos(np.radians(lut_co_phi))
    lut_co_mer = lut_co_spd*np.sin(np.radians(lut_co_phi))

    mu = np.real(ancillary_wind)
    mv = -np.imag(ancillary_wind)

    Jwind_co = ((lut_co_zon-mu)/du)**2 + ((lut_co_mer-mv)/dv)**2

    Jsig_co = (
        (gmf_ufunc_co_inc(np.array([theta]), phi_1d, wspd_1d)-sigco)/dsig_co)**2

    J_co = Jwind_co+Jsig_co

    ind = (np.argmin(J_co) // J_co.shape[-1], np.argmin(J_co) % J_co.shape[-1])

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

    idx_inc_cr = np.argmin(np.abs(lut_co_inc-incid))

    Jwind_co = ((lut_co_zon-mu)/du)**2 + \
        ((lut_co_mer-mv)/dv)**2

    lut_ncrs_inc = lut_co_nrcs[idx_inc_cr, :, :]
    Jsig_co = ((lut_ncrs_inc-sigco)/dsig_co)**2

    J_co = Jwind_co + Jsig_co

    __min = 99999999
    i_min = 0
    j_min = 0

    for i in range(0, J_co.shape[0]):
        j = (np.argmin(J_co[i, :]) % J_co.shape[-1])
        # np.where(J[i, :] == J[i, :].min())[0][0]
        min_t = J_co[i, j]
        if min_t < __min:
            __min = min_t
            i_min = i
            j_min = j

    return lut_co_spd[i_min, j_min]


@vectorize([float64(float64, float64, float64, float64, complex128, boolean)], forceobj=False, nopython=True, target="parallel")
def perform_dualpol_inversion_1pt_guvect(sigco, sigcr, nesz_cr, incid, ancillary_wind, mask):
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
    if mask:
        return np.nan

    # co pol solution
    mu = np.real(ancillary_wind)
    mv = -np.imag(ancillary_wind)

    idx_inc_cr = np.argmin(np.abs(lut_co_inc-incid))

    Jwind_co = ((lut_co_zon-mu)/du)**2 + \
        ((lut_co_mer-mv)/dv)**2

    lut_nrcs_inc_co = lut_co_nrcs[idx_inc_cr, :, :]
    Jsig_co = ((lut_nrcs_inc_co-sigco)/dsig_co)**2

    J_co = Jwind_co + Jsig_co

    __min = 99999999
    i_min = 0
    j_min = 0

    for i in range(0, J_co.shape[0]):
        j = (np.argmin(J_co[i, :]) % J_co.shape[-1])
        # np.where(J[i, :] == J[i, :].min())[0][0]
        min_t = J_co[i, j]
        if min_t < __min:
            __min = min_t
            i_min = i
            j_min = j

    spd_co = lut_co_spd[i_min, j_min]

    idx_inc_cr = np.argmin(np.abs(lut_cr_inc-incid))
    Jwind_cr = ((lut_cr_spd-spd_co)/du10_fg)**2.

    nrcslin = 10.**(sigcr/10.)
    dsig_cr = 1./(1.25/(nrcslin/nesz_cr))**4.
    lut_nrcs_inc_cr = lut_cr_nrcs[idx_inc_cr, :]
    Jsig_cr = ((lut_nrcs_inc_cr-sigcr)*dsig_cr)**2

    J_cr = Jsig_cr + Jwind_cr

    min__ = (np.argmin(J_cr) % J_cr.shape[-1])
    spd_dual = lut_cr_spd[min__]

    if (spd_dual < 5 or spd_co < 5 or np.isnan(spd_dual)):
        return spd_co
    return spd_dual
