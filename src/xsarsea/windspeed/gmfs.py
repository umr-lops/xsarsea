import numpy as np
import warnings
from ..utils import logger, timing
from functools import lru_cache
from numba import njit, vectorize, guvectorize, float64, float32
import xarray as xr

from .models import Model, available_models



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

        if ftype is None:
            return self._gmf_pyfunc_scalar

        if ftype == 'numba_njit':
            return njit([float64(float64, float64, float64)], nogil=True, inline='never')(self._gmf_pyfunc_scalar)

        if ftype == 'numba_vectorize':
            return vectorize(
                [
                    float64(float64, float64, float64),
                    float32(float32, float32, float64)
                ], target='parallel', nopython=True)(self._gmf_pyfunc_scalar)

        if ftype == 'numba_guvectorize':
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

            return func

        raise TypeError('ftype "%s" not known')

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

    def __call__(self, inc, wspd, phi=None, numba=True):

        all_scalar = all(np.isscalar(v) for v in [inc, wspd, phi] if v is not None)

        all_1d = False
        try:
            all_1d = all(v.ndim == 1 for v in [inc, wspd, phi] if v is not None)
        except AttributeError:
            all_1d = False

        if all_scalar:
            gmf_func = self._get_function_for_args(inc, wspd, phi=phi, numba=numba)
            sigma0 = gmf_func(inc, wspd, phi)
        elif all_1d:
            # guvectorize like
            gmf_func = self._get_function_for_args(inc, wspd, phi=phi, numba=numba)
            has_phi = phi is not None
            if phi is None:
                phi = np.array([np.nan])
                try:
                    dims = [inc.dims[0], wspd.dims[0]]
                except AttributeError:
                    dims = ['incidence', 'wspd']
                coords = {dims[0]: inc, dims[1]: wspd}
            else:
                try:
                    dims = [inc.dims[0], wspd.dims[0], phi.dims[0]]
                except AttributeError:
                    dims = ['incidence', 'wspd', 'phi']
                coords = {dims[0]: inc, dims[1]: wspd, dims[2]: phi}
            sigma0 = gmf_func(inc, wspd, phi)

            if not has_phi:
                # squeeze phi dim
                sigma0 = np.squeeze(sigma0, axis=2)

            # sigma0 is numpy array of shape (inc, wspd, [phi]).
            # convert it to xarray
            sigma0 = xr.DataArray(sigma0, dims=dims, coords=coords)
        else:
            # 2d array or more, with possible constants
            # make all broadcastable
            if phi is None:
                inc_b, wspd_b = np.broadcast_arrays(inc, wspd)
                phi_b = None
            else:
                inc_b, wspd_b, phi_b = np.broadcast_arrays(inc, wspd, phi)

            gmf_func = self._get_function_for_args(inc_b, wspd_b, phi=phi_b, numba=numba)
            sigma0 = gmf_func(inc_b, wspd_b, phi_b)

            # try to match xarray if available
            for v in (inc, wspd, phi):
                if v is None:
                    continue
                if isinstance(v, xr.DataArray):
                    da_sigma0 = v.copy()
                    da_sigma0.data = sigma0
                    da_sigma0.attrs.clear()
                    break

        try:
            sigma0.name = 'sigma0_gmf'
            sigma0.attrs['model'] = self.name
            sigma0.attrs['units'] = self.units
        except AttributeError:
            pass

        return sigma0


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

