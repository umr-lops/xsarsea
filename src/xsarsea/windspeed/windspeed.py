

import sys
from numba import vectorize, float64, complex128
import numpy as np
import warnings
import xarray as xr
from ..utils import timing
from .utils import logger
from .models import get_model

@timing(logger.info)
def invert_from_model(inc, sigma0, sigma0_dual=None, /, ancillary_wind=None, dsig_co=0.1, dsig_cr=0.1, model=None, **kwargs):
    """
    Invert sigma0 to retrieve windspeed from model (lut or gmf).

    Parameters
    ----------
    inc: xarray.DataArray
        incidence angle
    sigma0: xarray.DataArray
        sigma0 to be inverted.
    sigma0_dual: xarray.DataArray (optional)
        sigma0 in cross pol for dualpol invertion
    ancillary_wind=: xarray.DataArray (numpy.complex28)
        ancillary wind

            | (for example ecmwf winds), in **image convention**
    model=: str or tuple
        model to use.

            | If mono pol invertion, it could be a single str
            | If dualpol, it should be ordered (model_co, model_cr)

    Other Parameters
    ----------------
    dsig_co=: float
        parameter used for

            | `Jsig_co=((sigma0_gmf - sigma0) / dsig_co) ** 2`
    dsig_cr=: float or xarray.DataArray
        parameters used for

            | `Jsig_cr=((sigma0_gmf - sigma0) / dsig_cr) ** 2`

    Returns
    -------
    xarray.DataArray or tuple
        inverted windspeed in m/s

    See Also
    --------
    xsarsea.windspeed.available_models
    """
    # default array values if no dualpol
    nan = sigma0 * np.nan

    if not isinstance(model, tuple):
        models = (model, None)
    else:
        models = model

    models = tuple(get_model(m) if m is not None else None for m in models)

    if ancillary_wind is None:
        ancillary_wind = nan

    if sigma0_dual is None:
        # mono-pol inversion
        try:
            pol = sigma0.pol
        except AttributeError:
            pol = None
        model_pol = models[0].pol
        if pol is None:
            warnings.warn('Unable to check sigma0 pol. Assuming  %s' % model_pol)
        else:
            if pol not in model_pol:
                raise ValueError("sigma0 pol is %s, and model %s can only handle %s" % (pol, models[0].name, model_pol))

        if models[0].iscopol:
            sigma0_co = sigma0
            sigma0_cr = nan
            # copol needs valid ancillary wind
            assert np.any(~np.isnan(ancillary_wind))
        elif models[0].iscrosspol:
            sigma0_co = nan
            sigma0_cr = sigma0
            # cross pol only is better with no ancillary wind
            if not np.all(np.isnan(ancillary_wind)):
                warnings.warn('crosspol inversion is best without ancillary wind, but using it as requested.')
            models = (None, models[0])
    else:
        # dualpol inversion
        sigma0_co = sigma0
        sigma0_cr = sigma0_dual

    if np.isscalar(dsig_cr):
        dsig_cr = sigma0_cr * 0 + dsig_cr


    # to dB
    sigma0_co_db = 10 * np.log10(sigma0_co + 1e-15)
    if sigma0_cr is not nan:
        sigma0_cr_db = 10 * np.log10(sigma0_cr + 1e-15)
    else:
        sigma0_cr_db = nan

    def _invert_from_model_numpy(np_inc, np_sigma0_co_db, np_sigma0_cr_db, np_dsig_cr, np_ancillary_wind):
        # this wrapper function is only useful if using dask.array.map_blocks:
        # local variables defined here will be available on the worker, and they will be used
        # in _invert_copol_numba


        d_antenna = 2
        d_azi = 2
        dwspd_fg = 2

        try:
            sigma0_co_lut_db = models[0].to_lut(units='dB', **kwargs)  # shape (inc, wspd, phi)
            np_sigma0_co_lut_db = np.ascontiguousarray(np.asarray(sigma0_co_lut_db.transpose('wspd', 'phi', 'incidence')))
            np_wspd_dim = np.asarray(sigma0_co_lut_db.wspd)
            np_phi_dim = np.asarray(sigma0_co_lut_db.phi)
            np_inc_dim = np.asarray(sigma0_co_lut_db.incidence)


            if (180 - (np_phi_dim[-1] - np_phi_dim[0])) < 2:
                # phi is in range [ 0, 180 ] (symetrical lut)
                phi_180 = True
            else:
                phi_180 = False
        except AttributeError:
            # no copol, crosspol only
            # declare dummy numpy arrays for numba
            np_sigma0_co_lut_db = np.array([[[]]], dtype=np.float64)
            np_wspd_dim = np.array([], dtype=np.float64)
            np_phi_dim = np.array([], dtype=np.float64)
            np_inc_dim = np.array([], dtype=np.float64)
            phi_180 = False

        np_phi_lut, np_wspd_lut = np.meshgrid(np_phi_dim, np_wspd_dim)  # shape (wspd,phi)
        np_wspd_lut_co_antenna = np_wspd_lut * np.cos(np.radians(np_phi_lut))  # antenna (xtrack)
        np_wspd_lut_co_azi = np_wspd_lut * np.sin(np.radians(np_phi_lut))  # azi (atrack)


        if not np.all(np.isnan(np_sigma0_cr_db)):
            sigma0_cr_lut_db = models[1].to_lut(units='dB', **kwargs)
            np_sigma0_cr_lut_db = np.ascontiguousarray(np.asarray(sigma0_cr_lut_db.transpose('wspd', 'incidence')))
            np_wspd_lut_cr = np.asarray(sigma0_cr_lut_db.wspd)
            np_inc_cr_dim = np.asarray(sigma0_cr_lut_db.incidence)
        else:
            # dummy empty for numba typing
            np_inc_cr_dim = np.array([], dtype=np.float64)
            np_wspd_lut_cr = np.array([], dtype=np.float64)
            np_sigma0_cr_lut_db = np.array([[]], dtype=np.float64)


        def __invert_from_model_scalar(one_inc, one_sigma0_co_db, one_sigma0_cr_db, one_dsig_cr, one_ancillary_wind):
            # invert from gmf for scalar (float) input.
            # this function will be vectorized with 'numba.vectorize' or 'numpy.frompyfunc'
            # set debug=True below to force 'numpy.frompyfunc', so you can debug this code

            if np.isnan(one_inc):
                return np.nan + 1j * np.nan

            if not np.isnan(one_sigma0_co_db):
                # copol inversion available
                i_inc = np.argmin(np.abs(np_inc_dim-one_inc))
                np_sigma0_co_lut_db_inc = np_sigma0_co_lut_db[:, :, i_inc]

                # get wind dir components, relative to antenna and azi
                # '-'np.real, because for luts and gmf, wind speed is positive when it's xtrack component is negative
                m_antenna = -np.real(one_ancillary_wind)  # antenna (xtrack)
                m_azi = np.imag(one_ancillary_wind)  # azi (atrack)
                if phi_180:
                    m_azi = np.abs(m_azi)  # symetrical lut
                Jwind_co = ((np_wspd_lut_co_antenna - m_antenna) / d_antenna) ** 2 + \
                           ((np_wspd_lut_co_azi - m_azi) / d_azi) ** 2  # shape (phi, wspd)
                Jsig_co = ((np_sigma0_co_lut_db_inc - one_sigma0_co_db) / dsig_co) ** 2  # shape (wspd, phi)
                J_co = Jwind_co + Jsig_co
                ## cost function
                iJ_co = np.argmin(J_co)
                lut_idx = (iJ_co // J_co.shape[-1], iJ_co % J_co.shape[-1])
                wspd_co = np_wspd_lut[lut_idx]
            else:
                # no copol. use ancillary wind as wspd_co
                wspd_co = np.abs(one_ancillary_wind)

            wspd_dual = wspd_co
            if not np.isnan(one_sigma0_cr_db):
                # crosspol available, do dualpol inversion
                i_inc = np.argmin(np.abs(np_inc_cr_dim - one_inc))
                np_sigma0_cr_lut_db_inc = np_sigma0_cr_lut_db[:, i_inc]

                Jwind_cr = ((np_wspd_lut_cr - wspd_co) / dwspd_fg) ** 2.
                Jsig_cr = ((np_sigma0_cr_lut_db_inc - one_sigma0_cr_db) / one_dsig_cr) ** 2.
                if not np.all(np.isnan(Jwind_cr)):
                    J_cr = Jsig_cr + Jwind_cr
                else:
                    J_cr = Jsig_cr
                # numba doesn't support nanargmin
                # having nan in J_cr is an edge case, but if some nan where provided to analytical
                # function, we have to handle it
                # J_cr[np.isnan(J_cr)] = np.nanmax(J_cr)
                wspd_dual = np_wspd_lut_cr[np.argmin(J_cr)]
                #if (wspd_dual < 5) or (wspd_co < 5):
                #    wspd_dual = wspd_co

            # numba.vectorize doesn't allow multiple outputs, so we pack wspd_co and wspd_dual into a complex
            return wspd_co + 1j * wspd_dual

        # build a vectorized function from __invert_from_gmf_scalar
        debug = sys.gettrace()
        # debug = True  # force np.frompyfunc
        if debug:
            __invert_from_model_vect = timing(logger=logger.debug)(np.frompyfunc(__invert_from_model_scalar, 5, 1))
        else:
            # fastmath can be used, but we will need nan handling
            __invert_from_model_vect = timing(logger=logger.debug)(
                vectorize([complex128(float64, float64, float64, float64, complex128)], fastmath={'nnan': False}, target='parallel')
                (__invert_from_model_scalar)
            )
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*invalid value encountered.*', category=RuntimeWarning)
            return __invert_from_model_vect(np_inc, np_sigma0_co_db,
                                            np_sigma0_cr_db, dsig_cr, np_ancillary_wind)


    def _invert_from_model_any(inc, sigma0_co_db, sigma0_cr_db, dsig_cr, ancillary_wind):
        # wrapper to allow computation on any type (xarray, numpy)

        try:
            # if input is xarray, will return xarray
            da_ws = xr.zeros_like(sigma0_co_db, dtype=np.complex128)
            da_ws.name = 'windspeed_gmf'
            da_ws.attrs.clear()
            try:
                # if dask array, use map_blocks
                # raise ImportError
                import dask.array as da
                if any(
                        [
                            isinstance(v.data, da.Array)
                            for v in [inc, sigma0_co_db, sigma0_cr_db, dsig_cr, ancillary_wind]
                        ]
                ):
                    da_ws.data = da.map_blocks(
                        _invert_from_model_numpy,
                        inc.data, sigma0_co_db.data, sigma0_cr_db.data, dsig_cr.data, ancillary_wind.data,
                        meta=sigma0_co_db.data
                    )
                    logger.debug('invert with map_blocks')
                else:
                    raise TypeError

            except (ImportError, TypeError):
                # use numpy array, but store in xarray
                da_ws.data = _invert_from_model_numpy(
                    np.asarray(inc),
                    np.asarray(sigma0_co_db),
                    np.asarray(sigma0_cr_db),
                    np.asarray(dsig_cr),
                    np.asarray(ancillary_wind),
                )
                logger.debug('invert with xarray.values. no chunks')
        except TypeError:
            # full numpy
            logger.debug('invert with numpy')
            da_ws = _invert_from_model_numpy(
                inc,
                sigma0_co_db,
                sigma0_cr_db,
                dsig_cr,
                ancillary_wind
            )

        return da_ws.astype(np.complex128)

    # main
    ws = _invert_from_model_any(inc, sigma0_co_db, sigma0_cr_db, dsig_cr, ancillary_wind)

    # ws is complex128, extract ws_co and  ws_dual
    ws_co, ws_cr_or_dual = (np.real(ws), np.imag(ws))

    if models[0] and models[0].iscopol:
        try:
            ws_co.attrs['comment'] = "windspeed inverted from model %s (%s)" % ( models[0].name, models[0].pol)
            ws_co.attrs['model'] = models[0].name
            ws_co.attrs['units'] = 'm/s'
        except AttributeError:
            # numpy only
            pass


    if models[1]:
        if sigma0_dual is None and models[1].iscrosspol:
            # crosspol only
            try:
                ws_cr_or_dual.attrs['comment'] = "windspeed inverted from model %s (%s)" % (models[1].name, models[1].pol)
                ws_cr_or_dual.attrs['model'] = models[1].name
                ws_cr_or_dual.attrs['units'] = 'm/s'
            except AttributeError:
                # numpy only
                pass

    if sigma0_dual is None:
        # monopol inversion
        if models[0] is not None:
            # mono copol
            return ws_co
        else:
            # mono crosspol
            return ws_cr_or_dual
    else:
        # dualpol inversion
        wspd_dual = xr.where((ws_co < 5) | (ws_cr_or_dual < 5), ws_co, ws_cr_or_dual)
        try:
            wspd_dual.attrs['comment'] = "windspeed inverted from model %s (%s) and %s (%s)" % (models[0].name, models[0].pol, models[1].name, models[1].pol)
            wspd_dual.attrs['model'] = "%s %s" % (models[0].name, models[1].name)
            wspd_dual.attrs['units'] = 'm/s'
        except AttributeError:
            # numpy only
            pass
        return ws_co, wspd_dual
