import numpy as np
import warnings
from ..utils import logger, timing
from functools import lru_cache
from numba import njit, vectorize, guvectorize, float64, float32
import xarray as xr
import dask.array as da
from .models import Model, available_models
import time


class GmfModel(Model):

    @classmethod
    def register(cls, name=None, inc_range=[17., 50.], wspd_range=[0.2, 50.], phi_range=None, pol=None, units='linear'):
        """TODO: docstring"""

        def inner(func):
            gmf_name = name or func.__name__

            if not gmf_name.startswith('gmf_'):
                raise ValueError("gmf function must start with 'gmf_'. Got %s" % gmf_name)

            gmf_model = cls(gmf_name, func, inc_range=inc_range, wspd_range=wspd_range, phi_range=phi_range,
                            pol=pol, units=units)

            return gmf_model

        return inner

    def __init__(self, name, gmf_pyfunc_scalar, **kwargs):
        super().__init__(name, **kwargs)
        self._gmf_pyfunc_scalar = gmf_pyfunc_scalar

    @timing
    @lru_cache
    def _gmf_function(self, ftype='numba_vectorize'):
        """
         get vectorized function for gmf

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

        t0 = time.time()
        gmf_function = None

        if ftype is None:
            gmf_function = self._gmf_pyfunc_scalar
        elif ftype == 'numba_njit':
            gmf_function = njit([float64(float64, float64, float64)], nogil=True, inline='never')(
                self._gmf_pyfunc_scalar)
        elif ftype == 'numba_vectorize':
            gmf_function = vectorize(
                [
                    float64(float64, float64, float64),
                    float32(float32, float32, float64)
                ], target='parallel', nopython=True)(self._gmf_pyfunc_scalar)
        elif ftype == 'numba_guvectorize':
            func_njit = self._gmf_function(ftype='numba_njit')

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

            gmf_function = func
        else:
            raise TypeError('ftype "%s" not known')

        logger.debug('_gmf_function from %s for ftype %s in %f.1s' % (
        self._gmf_pyfunc_scalar.__name__, str(ftype), time.time() - t0))

        return gmf_function

    @timing
    def _get_function_for_args(self, inc, wspd, phi=None, numba=True):
        # get vectorized function from argument type
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

        gmf_func = self._gmf_function(ftype=ftype)
        return gmf_func

    @timing
    def __call__(self, inc, wspd, phi=None, broadcast=False, numba=True):
        # if all scalar, will return scalar
        all_scalar = all(np.isscalar(v) for v in [inc, wspd, phi] if v is not None)

        # if all 1d, will return 2d or 3d shape('incidence', 'wspd', 'phi'), unless broadcast is True
        all_1d = all(hasattr(v, 'ndim') and v.ndim == 1 for v in [inc, wspd, phi] if v is not None)

        # if dask, will use da.map_blocks
        dask_used = any(hasattr(v, 'data') and isinstance(v.data, da.Array) for v in [inc, wspd, phi])

        # template, if available
        sigma0_gmf = None

        # if dims >1, will assume broadcastable
        if any(hasattr(v, 'ndim') and v.ndim > 1 for v in [inc, wspd, phi] if v is not None):
            broadcast = True

        has_phi = phi is not None
        if not has_phi:
            # dummy dim that we will remove later
            phi = np.array([np.nan])

        # if broadcast is True, try to broadcast arrays to the same shape (the result will have the same shape).
        if broadcast:
            if dask_used:
                broadcast_arrays = da.broadcast_arrays
            else:
                broadcast_arrays = np.broadcast_arrays

            inc_b, wspd_b, phi_b = broadcast_arrays(inc, wspd, phi)

            gmf_func = self._gmf_function(ftype='numba_vectorize' if numba else 'vectorize')

            # find datarray in inputs that looks like th result
            for v in (inc, wspd, phi):
                if isinstance(v, xr.DataArray):
                    # will use this dataarray as an output template
                    sigma0_gmf = v.copy().astype(np.float64)
                    sigma0_gmf.attrs.clear()
                    break
            sigma0_gmf_data = gmf_func(inc_b, wspd_b, phi_b)  # numpy or dask.array if some input are da
            if sigma0_gmf is not None:
                sigma0_gmf.data = sigma0_gmf_data
            else:
                # fallback to pure numpy for the result
                sigma0_gmf = sigma0_gmf_data
        elif all_1d or all_scalar:
            gmf_func = self._get_function_for_args(inc, wspd, phi=phi, numba=numba)
            if all_scalar:
                sigma0_gmf = gmf_func(inc, wspd, phi)
            elif all_1d:
                default_dims = {
                    'incidence': inc,
                    'wspd': wspd,
                    'phi': phi
                }
                dims = [v.dims[0] if hasattr(v, 'dims') else default for default, v in default_dims.items()]
                coords = {dim: default_dims[v] for dim,v in zip(dims, default_dims.keys())}
                sigma0_gmf = xr.DataArray(np.empty(tuple(len(v) for v in coords.values())), dims=dims, coords=coords)
                sigma0_gmf.data = gmf_func(inc, wspd, phi)

        else:
            raise ValueError('Non 1d shape must all have the same shape')

        if not has_phi:
            sigma0_gmf = np.squeeze(sigma0_gmf, -1)
            try:
                sigma0_gmf = sigma0_gmf.drop('phi')
            except AttributeError:
                pass

        try:
            sigma0_gmf.attrs['units'] = self.units
        except AttributeError:
            pass


        return sigma0_gmf

    def _raw_lut(self, inc_range=None, phi_range=None, wspd_range=None, allow_interp=True):
        inc_range = inc_range or self.inc_range
        phi_range = phi_range or self.phi_range
        wspd_range = wspd_range or self.wspd_range

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
        inc = np.arange(inc_range[0] - inc_step, inc_range[1] + 2 * inc_step, inc_step)
        wspd = np.arange(np.max([0, wspd_range[0] - wspd_step]), wspd_range[1] + 2 * wspd_step, wspd_step)

        try:
            phi = np.arange(phi_range[0], phi_range[1] + 2 * phi_step, phi_step)
        except TypeError:
            phi = None

        lut = self.__call__(inc, wspd, phi)

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

        return lut
