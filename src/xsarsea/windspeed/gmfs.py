import numpy as np
import warnings
from ..utils import timing
from .utils import logger
from functools import lru_cache
from numba import njit, vectorize, guvectorize, float64, float32
import xarray as xr
import dask.array as da
from .models import Model
import time

class GmfModel(Model):
    """
    GmfModel class for handling model from analitycal functions. See :func:`~Model`
    """

    _name_prefix = 'gmf_'
    _priority = 3

    @classmethod
    def register(cls, name=None, pol=None, units='linear', **kwargs):
        """
        | provide a decorator for registering a gmf function.
        | The decorated function should be able to handle float as input.

        Parameters
        ----------
        name: str
            name of the registered gmf. Should start with `gmf_`. default to function name.
        wspd_range: list
            windspeed interval validity. Default to [0.2, 50.] for copol, or [3.0, 80.] for crosspol
        pol: str
            gmf polarisation. for ex 'VV' or 'VH'
        units: str
            sigma0 units returned by this gmf. Should be 'linear' or 'dB'

        Examples
        --------
        Register a new gmf

        >>> @xsarsea.windspeed.gmfs.GmfModel.register(pol='VH', units='linear')
        >>> def gmf_dummy(inc, wspd, phi=None):
        >>>     a0 = 0.00013106836021008122
        >>>     a1 = -4.530598283705591e-06
        >>>     a2 = 4.429277425062766e-08
        >>>     b0 = 1.3925444179360706
        >>>     b1 = 0.004157838450541205
        >>>     b2 = 3.4735809771069953e-05
        >>>
        >>>     a = a0 + a1 * inc + a2 * inc ** 2
        >>>     b = b0 + b1 * inc + b2 * inc ** 2
        >>>     sig = a * wspd ** b
        >>>    return sig
        >>>
        >>> gmf_dummy
        <GmfModel('gmf_dummy') pol=VH>
        >>> gmf_dummy(np.arange(20,22), np.arange(10,12))
        <xarray.DataArray (incidence: 2, wspd: 2)>
        array([[0.00179606, 0.00207004],
        [0.0017344 , 0.00200004]])
        Coordinates:
        * incidence  (incidence) int64 20 21
        * wspd       (wspd) int64 10 11
        Attributes:
        units:    linear


        Returns
        -------
        GmfModel
            (if used as a decorator)


        """

        def inner(func):
            gmf_name = name or func.__name__

            if not gmf_name.startswith(cls._name_prefix):
                raise ValueError("gmf function must start with '%s'. Got %s" % ( cls._name_prefix, gmf_name ))

            wspd_range = kwargs.pop('wspd_range', None)
            if wspd_range is None:
                if len(set(pol)) == 1:
                    # copol
                    wspd_range = [0.2, 50.]
                else:
                    # crosspol
                    wspd_range = [3.0, 80.]

            gmf_model = cls(gmf_name, func,
                            wspd_range=wspd_range, pol=pol, units=units, **kwargs)

            return gmf_model

        return inner

    def __init__(self, name, gmf_pyfunc_scalar, wspd_range=[0.2, 50.], pol=None, units=None, **kwargs):
        # register gmf_pyfunc_scalar as model name

        # check the gmf with scalar inputs
        sigma0_gmf = gmf_pyfunc_scalar(35, 0.2, 90.)  # No try/expect. let TypeError raise if gmf use numpy arrays
        sigma0_gmf = [sigma0_gmf]

        # check if the gmf accepts phi
        try:
            gmf_pyfunc_scalar(35, 0.2, None)
            phi_range = None
            logger.debug("%s doesn't needs phi" % name)
        except TypeError:
            # gmf needs phi
            # guess the range [0., 180.] or [0., 360.]
            # if phi is [0, 180], opposite dir will give the same sigma0
            phi_list = [0, 90, 180, 270]
            sigma0_gmf = [np.abs(gmf_pyfunc_scalar(35, 0.2, phi) - gmf_pyfunc_scalar(35, 0.2, -phi)) for phi in phi_list]

            if min(sigma0_gmf) < 1e-15 :
                # modulo 180
                logger.debug("%s needs phi %% 180" % name)
                phi_range = [0., 180.]
            else:
                logger.debug("%s needs phi %% 360" % name)
                phi_range = [0., 360.]

        # we provide a very small windspeed. if units is dB, sigma0 should be negative.
        if (units == 'dB' and min(sigma0_gmf) > 0) or (units == 'linear' and min(sigma0_gmf) < 0):
            logger.info("Possible bad units '%s'  for gmf %s" % (units, name))

        super().__init__(name, units=units, pol=pol, wspd_range=wspd_range, phi_range=phi_range, **kwargs)
        self._gmf_pyfunc_scalar = gmf_pyfunc_scalar

    @timing(logger.debug)
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
            raise TypeError('ftype "%s" not known' % ftype)

        return gmf_function

    @timing(logger.debug)
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

    @timing()
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
                sigma0_gmf = sigma0_gmf.drop_vars('phi')
            except AttributeError:
                pass

        try:
            sigma0_gmf.attrs['units'] = self.units
        except AttributeError:
            pass


        return sigma0_gmf

    @timing(logger=logger.debug)
    def _raw_lut(self, **kwargs):


        resolution = kwargs.pop('resolution', 'low')  # low res by default
        if resolution is None:
            if self.iscopol:
                # low resolution by default if phi (copol)
                resolution = 'low'
            else:
                resolution = 'high'

        logger.debug('_raw_lut gmf at res %s' % resolution)

        if resolution == 'low':
            # the lut is generated at low res, for improved performance
            # self.to_lut() will interp it to high res

            inc_step = kwargs.pop('inc_step_lr', self.inc_step_lr)
            wspd_step = kwargs.pop('wspd_step_lr', self.wspd_step_lr)
            phi_step = kwargs.pop('phi_step_lr', self.phi_step_lr)
        elif resolution == 'high':
            inc_step = kwargs.pop('inc_step', self.inc_step)
            wspd_step = kwargs.pop('wspd_step', self.wspd_step)
            phi_step = kwargs.pop('phi_step', self.phi_step)

        inc, wspd, phi = [
            r and np.linspace(r[0], r[1], num=int(np.round((r[1] - r[0]) / step) + 1))
            for r, step in zip(
                [self.inc_range, self.wspd_range, self.phi_range],
                [inc_step, wspd_step, phi_step]
            )
        ]

        lut = self.__call__(inc, wspd, phi)
        lut.attrs['resolution'] = resolution

        return lut
