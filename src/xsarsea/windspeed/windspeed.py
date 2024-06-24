import sys
from numba import vectorize, guvectorize, float64, complex128, void
import numpy as np
import warnings
import xarray as xr
from ..utils import timing
from .utils import logger
from .models import get_model


@timing(logger.debug)
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

            | (for example ecmwf winds), in **model convention**
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
        inverted windspeed in m/s.
        If available (copol or dualpol), the returned array is `np.complex64`, with the angle of the returned array is
        inverted direction in **gmf convention** (use `-np.conj(result))` to get it in standard convention)

    See Also
    --------
    xsarsea.windspeed.available_models
    """
    # default array values if no dualpol
    nan = sigma0 * np.nan

    #  put nan values where sigma0 is nan or 0
    #  sigma0 = np.where(sigma0 == 0, np.nan, sigma0)
    #  if sigma0_dual is not None:
    #    sigma0_dual = np.where(sigma0_dual == 0, np.nan, sigma0_dual)

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
            pol = sigma0.pol.values.item()
        except AttributeError:
            pol = None
        model_pol = models[0].pol
        if pol is None:
            warnings.warn(
                'Unable to check sigma0 pol. Assuming  %s' % model_pol)
        else:
            if pol not in model_pol:
                raise ValueError("sigma0 pol is %s, and model %s can only handle %s" % (
                    pol, models[0].name, model_pol))

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
                warnings.warn(
                    'crosspol inversion is best without ancillary wind, but using it as requested.')
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
            sigma0_co_lut_db = models[0].to_lut(
                units='dB', **kwargs)  # shape (inc, wspd, phi)
            np_sigma0_co_lut_db = np.ascontiguousarray(np.asarray(
                sigma0_co_lut_db.transpose('wspd', 'phi', 'incidence')))
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

        np_phi_lut, np_wspd_lut = np.meshgrid(
            np_phi_dim, np_wspd_dim)  # shape (wspd,phi)
        np_wspd_lut_co_antenna = np_wspd_lut * \
            np.cos(np.radians(np_phi_lut))  # antenna (sample)
        np_wspd_lut_co_azi = np_wspd_lut * \
            np.sin(np.radians(np_phi_lut))  # azi (line)

        if not np.all(np.isnan(np_sigma0_cr_db)):
            sigma0_cr_lut_db = models[1].to_lut(units='dB', **kwargs)
            np_sigma0_cr_lut_db = np.ascontiguousarray(
                np.asarray(sigma0_cr_lut_db.transpose('wspd', 'incidence')))
            np_wspd_lut_cr = np.asarray(sigma0_cr_lut_db.wspd)
            np_inc_cr_dim = np.asarray(sigma0_cr_lut_db.incidence)
        else:
            # dummy empty for numba typing
            np_inc_cr_dim = np.array([], dtype=np.float64)
            np_wspd_lut_cr = np.array([], dtype=np.float64)
            np_sigma0_cr_lut_db = np.array([[]], dtype=np.float64)

        def __invert_from_model_1d(inc_1d, sigma0_co_db_1d, sigma0_cr_db_1d, dsig_cr_1d, ancillary_wind_1d, out_co, out_cr):
            # invert from gmf for 1d vector (float) input.
            # this function will be vectorized with 'numba.guvectorize' or 'numpy.frompyfunc'
            # set debug=True below to force 'numpy.frompyfunc', so you can debug this code

            # gmf and lut doesn't have the same direction convention than xsarsea in the sample direction
            # for xsarsea, positive sample means in the sample increasing direction
            # for gmf and lut, positive means in the sample decreasing direction
            # we switch ancillary wind to the gmf convention
            # ancillary_wind_1d = -np.conj(ancillary_wind_1d)

            for i in range(len(inc_1d)):
                one_inc = inc_1d[i]
                one_sigma0_co_db = sigma0_co_db_1d[i]
                one_sigma0_cr_db = sigma0_cr_db_1d[i]
                one_dsig_cr = dsig_cr_1d[i]
                one_ancillary_wind = ancillary_wind_1d[i]

                #  if incidence is NaN ; all output is NaN
                if np.isnan(one_inc):
                    out_co[i] = np.nan
                    out_cr[i] = np.nan
                    continue

                #  if ancillary is NaN, copol & dualpol are NaN
                if (not np.isnan(np.abs(one_sigma0_co_db)) and np.isnan(np.abs(one_ancillary_wind))):
                    out_co[i] = np.nan
                    out_cr[i] = np.nan
                    continue

                if not np.isnan(one_sigma0_co_db):
                    # copol inversion available

                    i_inc = np.argmin(np.abs(np_inc_dim-one_inc))
                    np_sigma0_co_lut_db_inc = np_sigma0_co_lut_db[:, :, i_inc]

                    # get wind dir components, relative to antenna and azi
                    m_antenna = np.real(one_ancillary_wind)  # antenna (sample)
                    m_azi = np.imag(one_ancillary_wind)  # azi (line)
                    if phi_180:
                        m_azi = np.abs(m_azi)  # symetrical lut
                    Jwind_co = ((np_wspd_lut_co_antenna - m_antenna) / d_antenna) ** 2 + \
                               ((np_wspd_lut_co_azi - m_azi) /
                                d_azi) ** 2  # shape (phi, wspd)
                    # shape (wspd, phi)
                    Jsig_co = ((np_sigma0_co_lut_db_inc -
                               one_sigma0_co_db) / dsig_co) ** 2
                    J_co = Jwind_co + Jsig_co
                    # cost function
                    iJ_co = np.argmin(J_co)
                    lut_idx = (iJ_co // J_co.shape[-1], iJ_co % J_co.shape[-1])
                    wspd_co = np_wspd_lut[lut_idx]
                    wphi_co = np_phi_lut[lut_idx]
                    if phi_180:
                        # two phi solution (phi & -phi). choose closest from ancillary wind

                        sol = wspd_co * np.exp(1j * np.deg2rad(wphi_co))
                        sol_2 = wspd_co * \
                            np.exp(1j * (np.deg2rad(-wphi_co)))

                        diff_angle = np.angle(one_ancillary_wind / sol)
                        diff_angle_2 = np.angle(one_ancillary_wind / sol_2)

                        wind_co = sol if np.abs(
                            diff_angle) <= np.abs(diff_angle_2) else sol_2

                        # xr.where(np.abs(diff_angle) > np.pi/2, -sol, sol)

                    else:
                        wind_co = wspd_co * np.exp(1j * np.deg2rad(wphi_co))

                else:
                    wind_co = np.nan * 1j

                if not np.isnan(one_sigma0_cr_db) and not np.isnan(one_dsig_cr):
                    # crosspol available, do dualpol inversion
                    i_inc = np.argmin(np.abs(np_inc_cr_dim - one_inc))
                    np_sigma0_cr_lut_db_inc = np_sigma0_cr_lut_db[:, i_inc]

                    Jwind_cr = (
                        (np_wspd_lut_cr - np.abs(wind_co)) / dwspd_fg) ** 2.
                    Jsig_cr = ((np_sigma0_cr_lut_db_inc -
                                one_sigma0_cr_db) / one_dsig_cr) ** 2.
                    if not np.isnan(np.abs(wind_co)):
                        # dualpol inversion, or crosspol with ancillary wind
                        J_cr = Jsig_cr + Jwind_cr
                    else:
                        # crosspol only inversion
                        J_cr = Jsig_cr
                    # numba doesn't support nanargmin
                    # having nan in J_cr is an edge case, but if some nan where provided to analytical
                    # function, we have to handle it
                    # J_cr[np.isnan(J_cr)] = np.nanmax(J_cr)
                    wspd_dual = np_wspd_lut_cr[np.argmin(J_cr)]
                    if not np.isnan(np.abs(wind_co)):
                        # dualpol inversion, or crosspol with ancillary wind
                        phi_dual = np.angle(wind_co)
                    else:
                        # crosspol only, no direction
                        phi_dual = 0
                    wind_dual = wspd_dual * np.exp(1j*phi_dual)
                else:
                    wind_dual = np.nan * 1j

                out_co[i] = wind_co
                out_cr[i] = wind_dual
            return None

        # build a vectorized function from __invert_from_gmf_scalar
        debug = sys.gettrace()
        #  debug = True  # force pure python
        # debug = False # force numba.guvectorize
        if debug:
            logger.debug(
                'using __invert_from_model_1d in pure python mode (debug)')

            @ timing(logger=logger.debug)
            def __invert_from_model_vect(*args):
                ori_shape = args[0].shape
                args_flat = tuple((arg.flatten() for arg in args))

                out_co = np.empty_like(args_flat[0])
                out_cr = np.empty_like(args_flat[1])
                __invert_from_model_1d(*args_flat, out_co, out_cr)
                return out_co.reshape(ori_shape), out_cr.reshape(ori_shape)
        else:
            # fastmath can be used, but we will need nan handling
            __invert_from_model_vect = timing(logger=logger.debug)(
                guvectorize(
                    [void(float64[:], float64[:], float64[:], float64[:],
                          complex128[:], complex128[:], complex128[:])],
                    '(n),(n),(n),(n),(n)->(n),(n)',
                    fastmath={'nnan': False}, target='parallel')
                (__invert_from_model_1d)
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message='.*invalid value encountered.*', category=RuntimeWarning)
            return __invert_from_model_vect(np_inc, np_sigma0_co_db,
                                            np_sigma0_cr_db, np_dsig_cr, np_ancillary_wind)

    def _invert_from_model_any(inc, sigma0_co_db, sigma0_cr_db, dsig_cr, ancillary_wind):
        # wrapper to allow computation on any type (xarray, numpy)

        try:
            # if input is xarray, will return xarray
            da_ws_co = xr.zeros_like(sigma0_co_db, dtype=np.complex128)
            da_ws_co.name = 'windspeed_gmf'
            da_ws_co.attrs.clear()
            da_ws_cr = xr.zeros_like(sigma0_co_db, dtype=np.float64)
            da_ws_cr.name = 'windspeed_gmf'
            da_ws_cr.attrs.clear()

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
                    da_ws_co.data, da_ws_cr.data = da.apply_gufunc(
                        _invert_from_model_numpy,
                        '(n),(n),(n),(n),(n)->(n),(n)',
                        inc.data, sigma0_co_db.data, sigma0_cr_db.data, dsig_cr.data, ancillary_wind.data
                    )
                    logger.debug('invert with map_blocks')
                else:
                    raise TypeError

            except (ImportError, TypeError):
                # use numpy array, but store in xarray
                da_ws_co.data, da_ws_cr.data = _invert_from_model_numpy(
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
            da_ws_co, da_ws_cr = _invert_from_model_numpy(
                inc,
                sigma0_co_db,
                sigma0_cr_db,
                dsig_cr,
                ancillary_wind
            )

        return da_ws_co, da_ws_cr

    # main
    ws_co, ws_cr_or_dual = _invert_from_model_any(
        inc, sigma0_co_db, sigma0_cr_db, dsig_cr, ancillary_wind)

    if models[0] and models[0].iscopol:
        try:
            ws_co.attrs['comment'] = "wind speed and direction inverted from model %s (%s)" % (
                models[0].name, models[0].pol)
            ws_co.attrs['model'] = models[0].name
            #  ws_co.attrs['units'] = 'm/s'
        except AttributeError:
            # numpy only
            pass

    if models[1]:
        if sigma0_dual is None and models[1].iscrosspol:
            # crosspol only
            try:
                ws_cr_or_dual.attrs['comment'] = "wind speed inverted from model %s (%s)" % (
                    models[1].name, models[1].pol)
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
            # mono crosspol (no wind direction)
            ws_cr = np.abs(ws_cr_or_dual)
            return ws_cr
    else:
        # dualpol inversion
        wspd_dual = xr.where((np.abs(ws_co) < 5) | (
            np.abs(ws_cr_or_dual) < 5), ws_co, ws_cr_or_dual)
        # wspd_dual = ws_cr_or_dual
        try:
            wspd_dual.attrs['comment'] = "wind speed and direction inverted from model %s (%s) and %s (%s)" % (
                models[0].name, models[0].pol, models[1].name, models[1].pol)
            wspd_dual.attrs['model'] = "%s %s" % (
                models[0].name, models[1].name)
            # wspd_dual.attrs['units'] = 'm/s'
        except AttributeError:
            # numpy only
            pass
        return ws_co, wspd_dual
