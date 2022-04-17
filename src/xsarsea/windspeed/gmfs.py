import numpy as np
import warnings
from ..xsarsea import logger, timing
from functools import lru_cache
from numba import njit, vectorize, guvectorize, float64, float32
import xarray as xr


registered_gmfs = {}

def register_gmf(name=None, inc_range=[17., 50.], wspd_range=[0.2, 50.], phi_range=None):
    """TODO: docstring"""
    def inner(func):
        gmf_name = name or func.__name__

        if not gmf_name.startswith('gmf_'):
            raise ValueError("gmf function must start with 'gmf_'. Got %s" % gmf_name)

        update = {
            'gmf': func,
            'inc_range': inc_range,
            'wspd_range': wspd_range,
            'phi_range': phi_range,
        }
        update = {k: update[k] for k in update.keys() if update[k] is not None}

        if gmf_name not in registered_gmfs:
            registered_gmfs[gmf_name] = {}

        registered_gmfs[gmf_name].update(update)

        return func

    return inner

@timing
def _gmf_to_lut(name, inc_range=None, phi_range=None, wspd_range=None, allow_interp=True):

    try:
        cmod_attrs = registered_gmfs[name].copy()
    except KeyError:
        raise KeyError('gmf %s not available. (available: %s)' % (name, registered_gmfs.keys()))

    inc_range = inc_range or cmod_attrs.pop('inc_range')
    phi_range = phi_range or cmod_attrs.pop('phi_range', None)
    wspd_range = wspd_range or cmod_attrs.pop('wspd_range')

    inc_step_hr = 0.1
    wspd_step_hr = 0.1
    phi_step_hr = 1

    inc_step_lr = 0.2
    wspd_step_lr = 0.5
    phi_step_lr = 1

    if allow_interp:
        inc_step = inc_step_lr
        wspd_step = wspd_step_lr
        phi_step = phi_step_lr
    else:
        inc_step = inc_step_hr
        wspd_step = wspd_step_hr
        phi_step = phi_step_hr

    # 2*step, because we want to be sure to not have bounds conditions in interp
    inc = np.arange(inc_range[0] - inc_step, inc_range[1] + 2*inc_step, inc_step)
    wspd = np.arange(np.max([0, wspd_range[0] - wspd_step]), wspd_range[1] + 2*wspd_step, wspd_step)

    try:
        phi = np.arange(phi_range[0], phi_range[1] + 2*phi_step, phi_step)
        dims = ['incidence', 'wspd', 'phi']
        coords = {'incidence': inc, 'wspd': wspd, 'phi': phi}
    except TypeError:
        phi = None
        dims = ['incidence', 'wspd']
        coords = {'incidence': inc, 'wspd': wspd}


    lut = xr.DataArray(
        gmf(inc, wspd, phi, name),
        dims=dims,
        coords=coords
    )

    if allow_interp:
        # interp to get high res
        interp_kwargs = {}
        interp_kwargs['incidence'] = np.arange(inc_range[0], inc_range[1] + inc_step_hr, inc_step_hr)
        interp_kwargs['wspd'] = np.arange(wspd_range[0], wspd_range[1] + wspd_step_hr, wspd_step_hr)
        if phi is not None:
            interp_kwargs['phi'] = np.arange(phi_range[0], phi_range[1] + phi_step_hr, phi_step_hr)

        lut = lut.interp(**interp_kwargs, kwargs=dict(bounds_error=True))

    # crop lut to exact range
    crop_cond = (lut.incidence >= inc_range[0]) & (lut.incidence <= inc_range[1])
    crop_cond = crop_cond & (lut.wspd >= wspd_range[0]) & (lut.wspd <= wspd_range[1])
    if phi is not None:
        crop_cond = crop_cond & (lut.phi >= phi_range[0]) & (lut.phi <= phi_range[1])

    lut = lut.where(crop_cond, drop=True)

    # TODO add lut.attrs

    return lut


@timing
@lru_cache
def _get_gmf_func(name, ftype='numba_vectorize'):
    """
     get function for gmf `name`

    Parameters
    ----------
    name: str
        gmf name

    ftype: str
        return function type. Allowed values are:

            - 'numba_vectorize': vectorized with `numba.vectorize` (default)
            - 'numba_guvectorize': vectorized with `numba.guvectorize`
            - 'vectorize': vectorized with `numpy.vectorize`, without signature
            - 'guvectorize': vectorized with `numpy.vectorize`, with signature
            - 'numba_njit': compiled with `numba.njit` (only scalar input will be allowed)
            - None: pure python function (only scalar input will be allowed)


    Returns
    -------
    function
        `sigma0_linear = function(inc, wspd, [phi])`
    """

    try:
        gmf_attrs = registered_gmfs[name].copy()
    except KeyError:
        raise KeyError('gmf %s not available. (available: %s)' % (name, registered_gmfs.keys()))

    pyfunc = gmf_attrs['gmf']

    if pyfunc is None:
        raise NotImplementedError('no gmf implementation for %s' % name)

    if ftype is None:
        return pyfunc

    if ftype == 'numba_njit':
        return njit([float64(float64, float64, float64)], nogil=True, inline='never')(pyfunc)

    if ftype == 'numba_vectorize':
        return vectorize(
            [
                float64(float64, float64, float64),
                float32(float32, float32, float64)
            ], target='parallel', nopython=True)(pyfunc)

    if ftype == 'numba_guvectorize':
        func_njit = _get_gmf_func(name, ftype='numba_njit')

        @guvectorize(
            [
                (float64[:], float64[:], float64[:], float64[:, :, :]),
                (float32[:], float32[:], float32[:], float32[:, :, :])
            ], '(n),(m),(p)->(n,m,p)', target='cpu')
        def func(inc, wspd, phi, sigma0_out):
            for i_phi, one_phi in enumerate(phi):
                for i_wspd, one_wspd in enumerate(wspd):
                    for i_inc, one_inc in enumerate(inc):
                        sigma0_out[i_inc, i_wspd, i_phi] = func_njit(one_inc, one_wspd, one_phi)

        return func

    raise TypeError('ftype "%s" not known')


@timing
def gmf(inc, wspd, phi=None, name=None, numba=True):
    try:
        assert registered_gmfs[name]['gmf'] is not None
    except (KeyError, AssertionError):
        available_cmod = [k for k in registered_gmfs.keys() if registered_gmfs[k]['gmf'] is not None]
        raise ValueError("gmf name must be one of %s, and have an analytical function" % available_cmod)


    # input ndim give the function ftype
    try:
        ndim = wspd.ndim
    except AttributeError:
        # scalar input
        ndim = 0

    if numba:
        if ndim == 0:
            ftype = 'numba_njit'
        elif ndim == 1:
            ftype = 'numba_guvectorize'
        else:
            ftype = 'numba_vectorize'
    else:
        if ndim == 0:
            ftype = None
        elif ndim == 1:
            ftype = 'guvectorize'
        else:
            ftype = 'vectorize'

    gmf_func = _get_gmf_func(name, ftype=ftype)

    # every gmf needs a phi, even for crosspol, but we will squeeze it after compute (for guvectorize function)
    squeeze_phi_dim = (phi is None) and (ndim == 1)
    if squeeze_phi_dim:
        phi = np.array([np.nan])
    if phi is None:
        # non guvectorized function with no phi
        phi = wspd * np.nan

    sigma0_lin = gmf_func(inc, wspd, phi)
    if squeeze_phi_dim:
        sigma0_lin = np.squeeze(sigma0_lin, axis=2)

    # add name and comment to variable, if xarray
    try:
        sigma0_lin.name = 'sigma0_gmf'
        sigma0_lin.attrs['comment'] = "sigma0_gmf from '%s'" % name
        sigma0_lin.attrs['units'] = 'linear'
    except AttributeError:
        pass

    return sigma0_lin




