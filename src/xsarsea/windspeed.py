"""
Ressources TODO

Combined Co- and Cross-Polarized SAR Measurements Under Extreme Wind Conditions
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2006JC003743

"""
import numpy as np
import xarray as xr
import pickle
try:
    from xsar.utils import timing
except ImportError:
    # null decorator
    def timing(func):
        return func


# LUT_path = '/home1/datahome/vlheureu/inversion_Alx/'

LUT_path = "/home/vincelhx/Documents/ifremer/CDD/"
# Version a 50 m/s
LUT_file_name_vv = 'sig_vv_LUT_cmod5n_upto50ms.pkl'
# LUT_file_name_cp = 'LUT_CP_GMF_H14E_2D.pkl'
LUT_file_name_cp_hwang = 'LUT_CP_GMF_H14E_2D_80ms.pkl'  # Hwang
# LUT_file_name_cp_mouche = 'LUT_CP_GMF_MS1A_2D_80ms.pkl' # Exeter
# LUT_file_name_cp_mouche = 'LUT_CP_GMF_MS1An_2D_80ms.pkl' # IEEE 2017
LUT_file_name_cp_mouche = 'LUT_CP_GMF_MS1AHW_2D_80ms.pkl'  # High Wind

# LUT_path2 = "/home/datawork-cersat-public/cache/project/sarwing/GMFS/v1.6/"


def format_angle(angle, cycle=360, compass_format=False):
    r"""
    This routine performs modulo operation on input angle given the value
    of cycle. The output value is then either formatted between 0 and
    cycle (when compass_format keyword is set) or between -cycle/2 and
    cycle/2.

    :param angle: The angle to be formatted (expressed in degrees)
    :param cycle: The modulo value (set to 360° if not defined)
    :param compass_format: If set to True, then the output value will be
                           formatted between 0 and cycle, else it will be between -cycle/2
                           and cycle/2

    Procedure:
        1. The angle value is transformed as follows:
            .. math:: angle = Mod(angle,cycle)
            .. math:: angle = \lbrace^{angle+cycle, \qquad if \qquad angle<=cycle/2}_{angle-cycle, \qquad if \qquad angle>cycle/2}
        2. Finally, the returned value of angle is given by:
            .. math:: angle = \lbrace^{angle+cycle, \qquad if \qquad compass_{format}=True}_{angle, \qquad else}

    """
    # stretch angles between [-180, 180] or [0, 360]
    theta = np.mod(angle, cycle)
    try:
        theta[theta <= cycle/2] += cycle
        theta[theta > cycle/2] -= cycle

        if (compass_format):
            theta[theta < 0] += cycle
    except TypeError:
        if theta <= cycle/2:
            theta += cycle
        if theta > cycle/2:
            theta -= cycle

        if (compass_format):
            if theta < 0:
                theta += cycle

    return theta


class WindInversion:
    """
    WindInversion class
    """

    def __init__(self, ds_xsar):
        """
        # TODO we start with the full dataset?

        Parameters
        ----------
        sigma0: xarray.DataArray
            DataArray with dims `('pol','atrack','xtrack')`.
        nesz: xarray.DataArray
            DataArray with dims `('pol','atrack','xtrack')`.
        """
        self.ds_xsar = ds_xsar
        self.neszcr_mean = self.ds_xsar.nesz.isel(
            pol=1).mean(axis=0, skipna=True)
        self._spatial_dims = ['atrack', 'xtrack']

        self.LUT_cr, self.LUT_co = self.get_LUTs()

    def UV_to_spd_dir(self, _U, _V):
        return np.sqrt(_U**2+_V**2), np.arctan2(_V, _U) / np.pi*180

    def get_LUTs(self):
        print('... LUT loading ...')

        # Cross-Pol Hwang GMFnp.sin(phi_LUT/180.*np.pi)
        LUT_co = {},
        inc_cp_LUT, wspd_cp_LUT, sigma0_cp_LUT = pickle.load(
            open(LUT_path+LUT_file_name_cp_hwang, 'rb'), encoding='iso-8859-1')
        # wspd_cp_LUT,phi_cp_LUT = np.meshgrid(wspd_cp_LUT,phi_cp_LUT)
        ZON_cp_LUT = wspd_cp_LUT * \
            0.  # wspd_cp_LUT*np.cos(phi_cp_LUT/180.*np.pi)
        MER_cp_LUT = wspd_cp_LUT * \
            0.  # wspd_cp_LUT*np.sin(phi_cp_LUT/180.*np.pi)
        phi_cp_LUT = wspd_cp_LUT*0.
        LUT_cr = {}
        LUT_cr['H14E'] = {}
        LUT_cr['H14E']['INC'] = inc_cp_LUT
        LUT_cr['H14E']['ZON'] = ZON_cp_LUT
        LUT_cr['H14E']['MER'] = MER_cp_LUT
        LUT_cr['H14E']['WSPD'] = wspd_cp_LUT
        LUT_cr['H14E']['PHI'] = phi_cp_LUT
        LUT_cr['H14E']['NRCS'] = sigma0_cp_LUT*0.9

        # Cross-Pol Mouche GMF
        inc_cp_LUT, wspd_cp_LUT, sigma0_cp_LUT = pickle.load(
            open(LUT_path+LUT_file_name_cp_mouche, 'rb'), encoding='iso-8859-1')
        # wspd_cp_LUT,phi_cp_LUT = np.meshgrid(wspd_cp_LUT,phi_cp_LUT)
        ZON_cp_LUT = wspd_cp_LUT * \
            0.  # wspd_cp_LUT*np.cos(phi_cp_LUT/180.*np.pi)
        MER_cp_LUT = wspd_cp_LUT * \
            0.  # wspd_cp_LUT*np.sin(phi_cp_LUT/180.*np.pi)
        phi_cp_LUT = wspd_cp_LUT*0.
        LUT_cr['MS1A'] = {}
        LUT_cr['MS1A']['INC'] = inc_cp_LUT
        LUT_cr['MS1A']['ZON'] = ZON_cp_LUT
        LUT_cr['MS1A']['MER'] = MER_cp_LUT
        LUT_cr['MS1A']['WSPD'] = wspd_cp_LUT
        LUT_cr['MS1A']['PHI'] = phi_cp_LUT
        LUT_cr['MS1A']['NRCS'] = sigma0_cp_LUT

        # Co-Pol
        inc_LUT, phi_LUT, wspd_LUT, sigma0_LUT = pickle.load(
            open(LUT_path+LUT_file_name_vv, 'rb'), encoding='iso-8859-1')
        wspd_LUT, phi_LUT = np.meshgrid(wspd_LUT, phi_LUT)
        ZON_LUT = wspd_LUT*np.cos(phi_LUT/180.*np.pi)
        MER_LUT = wspd_LUT*np.sin(phi_LUT/180.*np.pi)
        LUT_co = {}
        LUT_co['CMOD5n'] = {}
        LUT_co['CMOD5n']['INC'] = inc_LUT
        LUT_co['CMOD5n']['ZON'] = ZON_LUT
        LUT_co['CMOD5n']['MER'] = MER_LUT
        LUT_co['CMOD5n']['WSPD'] = wspd_LUT
        LUT_co['CMOD5n']['PHI'] = phi_LUT
        LUT_co['CMOD5n']['NRCS'] = sigma0_LUT

        print('loaded !')
        return LUT_cr, LUT_co

    def noise_flattening_1row(self, nesz_row, incid_row, display=False):
        """
        TODO : Alexis is selecting peaks with find_peaks for polyfit
        Parameters
        ----------
        sigma0: xarray.DataArray
            DataArray with dims `('xtrack')`.
        incid: xarray.DataArray
            DataArray with dims `('xtrack')`.
        nesz_mean: xarray.DataArray
            DataArray with dims `('xtrack')`.
        display : boolean

        Returns
        -------
        xarray.Dataset
        """
        nesz_flat = nesz_row.copy()
        # replacing nan values by nesz mean value for concerned incidence
        nesz_flat[np.isnan(nesz_flat)] = self.neszcr_mean[np.isnan(nesz_flat)]

        noise = 10*np.log10(nesz_flat)
        _coef = np.polyfit(incid_row[np.isfinite(noise)],
                           noise[np.isfinite(noise)], 1)

        nesz_flat = 10.**((incid_row*_coef[0] + _coef[1] - 1.0)/10.)

        if display:
            # TODO USEFUL ?
            None
        return nesz_flat

    def noise_flattening(self, nesz_cr, incid):
        """

        Parameters
        ----------
        sigma0: xarray.DataArray
            DataArray with dims `('atrack','xtrack')`.
        incid: xarray.DataArray
            DataArray with dims `('atrack','xtrack')`.

        Returns
        -------
        xarray.Dataset
            with 'noise_flattened' variable.
        """

        return xr.apply_ufunc(self.noise_flattening_1row, nesz_cr,
                              incid,
                              input_core_dims=[["xtrack"], ["xtrack"]],
                              output_core_dims=[["xtrack"]],
                              dask='parallelized',
                              vectorize=True)

    def get_wind_from_cost_function(self, J, wspd_LUT):
        ind = np.where(J == J.min())
        if len(ind[0]) == 1:
            return wspd_LUT[ind]
        elif len(ind[0]) > 1:
            # sometimes there are 2 minimums (indices could be 0 & 360, so its could be due to LUTS) :
            # we take the first indice
            return wspd_LUT[np.array([ind[0][0]]), np.array([ind[1][0]])]
        else:
            return np.array([np.nan])

    def perform_copol_inversion_1pt(self, sigco, theta, ground_heading, ecmwf_wsp, ori_u, ori_v, du=2, dv=2, dsig=0.1):

        inc_LUT = self.LUT_co['CMOD5n']['INC']
        wspd_LUT = self.LUT_co['CMOD5n']['WSPD']
        sigma0_LUT = self.LUT_co['CMOD5n']['NRCS']

        ZON_LUT = self.LUT_co['CMOD5n']['ZON']
        MER_LUT = self.LUT_co['CMOD5n']['MER']

        index_inc = np.argmin(abs(inc_LUT-theta))

        ext_ancillary_wind_direction = 90. - \
            np.rad2deg(np.arctan2(ori_v, ori_u))

        # save the direction in meteorological conventionfor for export
        ext_ancillary_wind_direction = format_angle(
            ext_ancillary_wind_direction + 180, compass_format=True)

        mu = ecmwf_wsp * \
            np.cos(np.radians(ext_ancillary_wind_direction - ground_heading + 90.))
        mv = ecmwf_wsp * \
            np.sin(np.radians(ext_ancillary_wind_direction - ground_heading + 90.))

        Jwind = ((ZON_LUT-mu)/du)**2+((MER_LUT-mv)/dv)**2
        Jsig = ((sigma0_LUT[index_inc, :, :]-sigco)/dsig)**2
        J = Jwind+Jsig

        return self.get_wind_from_cost_function(J, wspd_LUT)

    def perform_copol_inversion_1pt(self, sigco, theta, ground_heading, ecmwf_wsp, ori_u, ori_v, du=2, dv=2, dsig=0.1):

        inc_LUT = self.LUT_co['CMOD5n']['INC']
        wspd_LUT = self.LUT_co['CMOD5n']['WSPD']
        sigma0_LUT = self.LUT_co['CMOD5n']['NRCS']

        ZON_LUT = self.LUT_co['CMOD5n']['ZON']
        MER_LUT = self.LUT_co['CMOD5n']['MER']

        index_inc = np.argmin(abs(inc_LUT-theta))

        ext_ancillary_wind_direction = 90. - \
            np.rad2deg(np.arctan2(ori_v, ori_u))

        # save the direction in meteorological conventionfor for export
        ext_ancillary_wind_direction = format_angle(
            ext_ancillary_wind_direction + 180, compass_format=True)

        mu = ecmwf_wsp * \
            np.cos(np.radians(ext_ancillary_wind_direction - ground_heading + 90.))
        mv = ecmwf_wsp * \
            np.sin(np.radians(ext_ancillary_wind_direction - ground_heading + 90.))

        Jwind = ((ZON_LUT-mu)/du)**2+((MER_LUT-mv)/dv)**2
        Jsig = ((sigma0_LUT[index_inc, :, :]-sigco)/dsig)**2
        J = Jwind+Jsig

        return self.get_wind_from_cost_function(J, wspd_LUT)

    def perform_crpol_inversion_1pt(self, sigcr, theta, ground_heading, ecmwf_wsp, ori_u, ori_v, du=2, dv=2, dsig=0.1):

        inc_cp_LUT = self.LUT_cr['H14E']['INC']
        ZON_cp_LUT = self.LUT_cr['H14E']['ZON']
        MER_cp_LUT = self.LUT_cr['H14E']['MER']
        wspd_cp_LUT = self.LUT_cr['H14E']['WSPD']
        #phi_cp_LUT = self.LUT_cr['H14E']['PHI']
        #sigma0_cp_LUT = self.LUT_cr['H14E']['NRCS']
        sigma0_cp_LUT2 = self.LUT_cr['MS1A']['NRCS']

        index_cp_inc = np.argmin(abs(inc_cp_LUT-theta))

        ext_ancillary_wind_direction = 90. - \
            np.rad2deg(np.arctan2(ori_v, ori_u))

        # save the direction in meteorological conventionfor for export
        ext_ancillary_wind_direction = format_angle(
            ext_ancillary_wind_direction + 180, compass_format=True)
        mu = ecmwf_wsp * \
            np.cos(np.radians(ext_ancillary_wind_direction - ground_heading + 90.))
        mv = ecmwf_wsp * \
            np.sin(np.radians(ext_ancillary_wind_direction - ground_heading + 90.))

        Jwind = ((ZON_cp_LUT-mu)/du)**2+((MER_cp_LUT-mv)/dv)**2
        Jsig_mouche = ((sigma0_cp_LUT2[index_cp_inc, :]-sigcr)/dsig)**2
        J = Jwind+Jsig_mouche

        return self.get_wind_from_cost_function(J, wspd_cp_LUT)

    def perform_dualpol_inversion_1pt(self, sigco, sigcr, nesz_cr,  theta, ground_heading, ecmwf_wsp, ori_u, ori_v, du=2, dv=2, dsig=0.1):

        inc_cp_LUT = self.LUT_cr['H14E']['INC']
        wspd_cp_LUT = self.LUT_cr['H14E']['WSPD']
        sigma0_cp_LUT2 = self.LUT_cr['MS1A']['NRCS']
        index_cp_inc = np.argmin(abs(inc_cp_LUT-theta))

        wsp_first_guess = self.perform_co_inversion_1pt(
            sigco, theta, ground_heading, ecmwf_wsp, ori_u, ori_v, du, dv, dsig)

        du10_fg = 2

        dsigcrpol = (1./((10**(sigcr/10))/nesz_cr))**2.
        # returning solution
        J_wind = ((wspd_cp_LUT-wsp_first_guess)/du10_fg)**2.
        J_sigcrpol2 = ((sigma0_cp_LUT2[index_cp_inc, :]-sigcr)/dsigcrpol)**2
        J_final2 = J_sigcrpol2 + J_wind

        wsp_mouche = self.get_wind_from_cost_function(J_final2, wspd_cp_LUT)

        if (wsp_mouche < 5 or wsp_first_guess < 5 or np.isnan(wsp_mouche)):
            return wsp_first_guess

        return wsp_mouche

    @timing
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
                                          ),
                              self.ds_xsar.incidence,
                              self.ds_xsar.ground_heading,

                              self.ds_xsar.ecmwf_0100_1h_WSPD,
                              self.ds_xsar.ecmwf_0100_1h_U10,
                              self.ds_xsar.ecmwf_0100_1h_V10,
                              dask='parallelized',
                              vectorize=True)

    @timing
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
                                          ),
                              self.ds_xsar.incidence,
                              self.ds_xsar.ground_heading,
                              self.ds_xsar.ecmwf_0100_1h_WSPD,
                              self.ds_xsar.ecmwf_0100_1h_U10,
                              self.ds_xsar.ecmwf_0100_1h_V10,
                              vectorize=True)

    @timing
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
        noise_flatened = self.noise_flattening(self.ds_xsar.sigma0.isel(pol=1),
                                               self.ds_xsar.incidence)

        # ecmwf_dir_img = ecmwf_dir-self.ds_xsar["ground_heading"]
        return xr.apply_ufunc(self.perform_mouche_inversion_1pt,
                              10*np.log10(self.ds_xsar.sigma0.isel(pol=0)
                                          ),
                              10*np.log10(self.ds_xsar.sigma0.isel(pol=1)
                                          ),
                              noise_flatened,
                              self.ds_xsar.incidence,
                              self.ds_xsar.ground_heading,
                              self.ds_xsar.ecmwf_0100_1h_WSPD,
                              self.ds_xsar.ecmwf_0100_1h_U10,
                              self.ds_xsar.ecmwf_0100_1h_V10,
                              vectorize=True)
