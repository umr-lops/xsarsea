"""
Ressources TODO

Combined Co- and Cross-Polarized SAR Measurements Under Extreme Wind Conditions
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2006JC003743

"""
import numpy as np
import xarray as xr
import pickle

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
            pol=1).mean(axis=0, skipna=True).compute()
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

        return xr.apply_ufunc(self.noise_flattening_1row, nesz_cr.compute(),
                              incid.compute(),
                              input_core_dims=[["xtrack"], ["xtrack"]],
                              output_core_dims=[["xtrack"]],
                              vectorize=True)

    def get_wind_from_cost_function(self, J, wspd_LUT):
        J[np.isnan(J)] = +9999.
        ind = np.where(J == J.min())
        if len(ind[0]) > 1:
            wspd_du2 = np.array([0])
        else:
            wspd_du2 = wspd_LUT[ind]
        return wspd_du2

    def spd_dir_to_UV(self, _spd, _dir, ground_heading=None):

        if ground_heading is not None:
            return _spd*np.cos((_dir-ground_heading+90)/180.*np.pi), _spd*np.sin((_dir-ground_heading+90)/180.*np.pi)
        else:
            return _spd*np.cos(_dir/180.*np.pi), _spd*np.sin(_dir/180.*np.pi)

    def perform_co_inversion_1pt(self, sigco, theta, ground_heading, ecmwf_u, ecmwf_v):

        du = 2
        dv = 2
        dsig = 0.1

        inc_LUT = self.LUT_co['CMOD5n']['INC']
        wspd_LUT = self.LUT_co['CMOD5n']['WSPD']
        sigma0_LUT = self.LUT_co['CMOD5n']['NRCS']
        ZON_LUT = self.LUT_co['CMOD5n']['ZON']
        MER_LUT = self.LUT_co['CMOD5n']['MER']

        index_inc = np.argmin(abs(inc_LUT-theta))

        # Cost functions ======================================

        ecmwf_wsp, ecmwf_dir = self.UV_to_spd_dir(ecmwf_u, ecmwf_v)

        mu, mv = self.spd_dir_to_UV(
            ecmwf_wsp, ecmwf_dir, ground_heading=ground_heading)

        Jwind = ((ZON_LUT-mu)/du)**2+((MER_LUT-mv)/dv)**2
        Jsig = ((sigma0_LUT[index_inc, :, :]-sigco)/dsig)**2
        J = Jwind+Jsig

        return self.get_wind_from_cost_function(J, wspd_LUT)

    def perform_mouche_inversion_1pt(self, sigco, sigcr, nesz_cr,  theta, track, ecmwf_wsp, ecmwf_dir, du=2, dv=2, dsig=0.1,  dsig_crpol=0.1):

        # ===============================================
        # Dual-Polarization Channel
        # ===============================================

        LUT_co = self.get_LUT_co()  # TODO
        LUT_cr = self.get_LUT_cr()  # TODO

        inc_cp_LUT = LUT_cr['H14E']['INC']
        wspd_cp_LUT = LUT_cr['H14E']['WSPD']
        sigma0_cp_LUT2 = LUT_cr['MS1A']['NRCS']

        index_cp_inc = np.argmin(abs(inc_cp_LUT-theta))

        wsp_mouche = 0
        wsp_first_guess = self.perform_co_inversion_1pt(
            sigco, theta, track, ecmwf_wsp, ecmwf_dir, LUT_co, du, dv, dsig)
        du10_fg = 2

        # full xarray here
        sigcr_flatten = self.noise_flattening(self, nesz_cr, theta)

        # select le bon point
        dsigcrpol = (1./(sigcr/sigcr_flatten))**2.

        # returning solution
        J_wind = ((wspd_cp_LUT-wsp_first_guess)/du10_fg)**2.
        J_sigcrpol2 = ((sigma0_cp_LUT2[index_cp_inc, :]-sigcr)/dsigcrpol)**2
        J_final2 = J_sigcrpol2 + J_wind
        wsp_mouche = self.getWindFromCostFunction(J_final2, wspd_cp_LUT)

        if (wsp_mouche < 5 or wsp_first_guess < 5):
            wsp_mouche = wsp_first_guess

    def perform_co_inversion(self):
        """

        Parameters
        ----------
        Returns
        -------
        xarray.Dataset
        """

        return xr.apply_ufunc(self.perform_co_inversion_1pt,
                              self.ds_xsar.sigma0.isel(pol=0).compute(),
                              self.ds_xsar.incidence.compute(),
                              self.ds_xsar.ground_heading.compute(),
                              self.ds_xsar.ecmwf_0100_1h_U10.compute(),
                              self.ds_xsar.ecmwf_0100_1h_V10.compute(),
                              vectorize=True)
