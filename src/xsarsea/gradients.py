"""
Implemented from:

'W. Koch, "Directional analysis of SAR images aiming at wind direction," IEEE Trans. Geosci. Remote Sens., vol. 42, no. 4, pp. 702-710, 2004.'

https://ieeexplore.ieee.org/document/1288365

https://www.climate-service-center.de/imperia/md/content/gkss/institut_fuer_kuestenforschung/ksd/paper/kochw_ieee_2004.pdf

"""

import numpy as np
from scipy import signal
import xarray as xr
import warnings
import cv2
from functools import lru_cache, reduce
from operator import mul

try:
    from xsar.utils import timing
except ImportError:
    # null decorator
    def timing(func):
        return func

import logging

logging.basicConfig()
logger = logging.getLogger('xsarsea.streaks')
logging.captureWarnings(True)


class Gradients:
    """
    Gradient class
    """
    def __init__(self, sigma0, heading=None, ortho=True, window_size=160, window_step=0.5, align_with=None):
        """

        Parameters
        ----------
        sigma0: xarray.DataArray
            DataArray with dims `('pol','atrack','xtrack')`. Koch recommend 100m pixel spacing for streaks detection.
        heading: xarray.DataArray or None
            DataArray with same dims `('atrack', 'xtrack')` as `sigma0`.
            If provided, gradients will be rotated by heading, so 0 is North. (not yet fully implemented)
            If not providef, 0 is increasing atrack, and 90 is increasing xtrack
        ortho: bool
            If True (the default), gradients are rotated by 90, to return orthogonal gradients.
        window_size: int, default 160
            Size of the rolling window. 160 by default. If pixel size is 100m, 160 is 16km.
        window_step: float
            Step of the rolling window. 1 means no overlaping, 0.5 means half overlaping.
        align_with: Gradient
            Align gradients and windows with another Gradient instance, so atracks and xtracks will be as close as possible.
            If provided, windows_step is ignored.
            This keyword is mainly used by MultiGradients.

        """
        self._sigma0 = sigma0
        self.heading = heading
        self.ortho = ortho
        self._ref_gradient = align_with

        self._spatial_dims = ['atrack', 'xtrack']

        # image will be resampled by a factor 4 before rolling window
        # so, to get a windows of 16 km for a 100m pixel, window size will be 40*40
        self.window_size = window_size
        self.window = {k: self.window_size // 4 for k in self._spatial_dims}
        self._window_dims = {k: "k_%s" % k for k in self._spatial_dims}

        self.n_angles = 72
        self.window_step = window_step

        # pixel count per window
        self._window_pixels = reduce(mul, self.window.values())

    @property
    @lru_cache
    def sigma0(self):
        """sigma0 provided by the caller"""
        __sigma0 = self._sigma0.compute()
        if 'pol' not in __sigma0.dims:
            # pol dim if not exists
            __sigma0 = __sigma0.expand_dims(dim='pol', axis=-1)
        return __sigma0

    @property
    @lru_cache
    def i2(self):
        """resampled sigma0 by factor 2, without moiré effect"""
        return xr.concat([R2(self.sigma0.sel(pol=p)) for p in self.sigma0.pol], dim='pol')

    @property
    @lru_cache
    def ampl(self):
        """amplitude (sqrt) of i2"""
        return np.sqrt(self.i2)

    @property
    @lru_cache
    def local_gradients(self):
        """local gradients from amplitude as an xarray.Dataset,
        with 'G2' variable that is a complex number whose argument is the direction and module the gradient weight,
        and 'c' variable that is a quality index"""
        return xr.concat([local_gradients(self.ampl.sel(pol=p), ortho=True) for p in self.ampl.pol], dim='pol')

    @property
    @lru_cache
    def rolling_gradients(self):
        """rolling window over `self.local_gradient`, according to `self.window_size` and `self.window_step`"""
        # construct sliding windows, (all, with step 1)
        rolling = self.local_gradients.rolling(self.window, center=True).construct(self._window_dims)

        if self._ref_gradient is None:
            # select index in rolling according to window_step
            # last index is appended, to prevent missing edge
            idx = {
                k: np.unique(
                    np.append(
                        np.arange(0, self.local_gradients[k].size, self.window[k] // int(1 / self.window_step)),
                        self.local_gradients[k].size - 1
                    )
                ) for k in self.window.keys()
            }
            rolling_step = rolling.isel(idx)
        else:
            # select window at same location as self._ref_gradient.rolling_gradients
            rolling_step = rolling.sel(
                atrack=self._ref_gradient.rolling_gradients.atrack,
                xtrack=self._ref_gradient.rolling_gradients.xtrack,
                method='nearest')
            # suppress duplicates indexes
            rolling_step = rolling_step.isel(
                atrack=np.unique(rolling_step.atrack, return_index=True)[1],
                xtrack=np.unique(rolling_step.xtrack, return_index=True)[1]
            )
        return rolling_step

    @property
    @lru_cache
    def direction_histogram(self):
        """
        direction histogramm as an xarray.Dataset, for all windows from `self.rolling_windows`
        """
        angles_bins = np.linspace(-180, 180, self.n_angles + 1)  # one extra bin
        angles_bins = (angles_bins[1:] + angles_bins[:-1]) / 2  # suppress extra bin (middle)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            grad_hist, ratio = xr.apply_ufunc(
                gradient_histogram,
                self.rolling_gradients['G2'], self.rolling_gradients['c'], angles_bins,
                input_core_dims=[self._window_dims.values(), self._window_dims.values(), ["angles"]],
                exclude_dims=set(self._window_dims.values()),
                output_core_dims=[['angles'], []],
                vectorize=True,
                output_dtypes=[np.complex128, np.float]
            )
            grad_hist = grad_hist.rename('angles_hist').assign_coords(angles=angles_bins)
            ratio = ratio.rename('used_ratio').fillna(0)
        __direction_histogram = xr.merge((grad_hist, ratio))

        # apply heading
        if self.heading is not None:
            __direction_histogram = __direction_histogram * np.exp(
                1j * xr.ufuncs.deg2rad(
                    self.heading.interp(
                        atrack=__direction_histogram.atrack,
                        xtrack=__direction_histogram.xtrack,
                        method='nearest')
                )
            )

        if self._ref_gradient is not None:
            # normalize histogram with self._ref_gradient
            __direction_histogram = __direction_histogram * self._ref_gradient._window_pixels / self._window_pixels
        return __direction_histogram

    @property
    @lru_cache
    def smoothed_direction_histogram(self):
        """smoothed version of `self.direction_histogram`"""
        return grad_hist_smooth(self.direction_histogram['angles_hist'])

    @lru_cache
    @timing
    def main_gradients(self, method='max', interpolate=True, complex=False):
        """

        Parameters
        ----------
        method: str
            'max' (histogram maximum) or 'peaks' (up to 2 locals maximums)
        interpolate: bool
            if True (the default), gradient will be interpolated onto original sigma0 grid.
            if False, only gradients at rolling windows center will be returned.
        complex: bool
            if True, gradient are returned as complex number.

        Returns
        -------
        xarray.Dataset
            with 'grad_dir' and 'grad_weight' variable if complex is False, or 'grad' variable if complex.
        """
        grad_dir = find_gradient(self.smoothed_direction_histogram, method=method)
        if interpolate:
            # interpolate so the shape is the same as sigma0
            # as complex argument is [-pi,+pi], interpolation is done squared
            grad_dir = np.sqrt((grad_dir ** 2).interp(atrack=self.sigma0.atrack, xtrack=self.sigma0.xtrack))
        if not complex:
            # convert complex to deg and module
            grad_ds = xr.merge(
                [
                    -xr.ufuncs.rad2deg(xr.ufuncs.angle(grad_dir)).rename('grad_dir'),
                    np.abs(grad_dir).rename('grad_weight')
                ])
        else:
            # keep as complex
            grad_ds = grad_dir.to_dataset(name='grad')
        return grad_ds


class MultiGradients:
    """
    Handle Gradients for multiples scales and resolutions
    """

    def __init__(self, sigma0, heading=None, ortho=True, downscale_factors=[1, 2, 4], windows_sizes=[160, 320], window_step=1):
        """

        Parameters
        ----------

        sigma0: xarray.DataArray
            DataArray with dims `('pol','atrack','xtrack')`. Koch recommend 100m pixel spacing for streaks detection.
        heading: xarray.DataArray or None
            DataArray with same dims `('atrack', 'xtrack')` as `sigma0`.
            If provided, gradients will be rotated by heading, so 0 is North. (not yet fully implemented)
            If not providef, 0 is increasing atrack, and 90 is increasing xtrack
        ortho: bool
            If True (the default), gradients are rotated by 90, to return orthogonal gradients.
        downscale_factors: list of numbers.
            for example, `[1,2,3]` will resample original 100m sigma0 to 100m, 200m, 300m
        windows_sizes: list of int
            Size of the rollings windows given to `Gradients`. For example `[ 160, 240 ]` for 16km and 24km (if original sigma0 is 100m)
            Note that those size refer to original (non downscaled) sigma0.
        window_step: float
            Step of the rolling window, for the smallest window. 1 means no overlaping, 0.5 means half overlaping.
            Biggest windows will be aligned onto this smallest window step (so they will have more overlapping).
        """
        self.sigma0 = sigma0
        self.heading = heading
        self.ortho = ortho
        self.downscale_factors = downscale_factors
        self.windows_sizes = windows_sizes  # 160 is 16km for pixel size of 100m

        # greatest downscale factor is used as reference
        self._ref_downscale_factors = self.downscale_factors[-1]
        # smallest window used as a reference
        self._ref_windows_sizes = self.windows_sizes[0]

        # window step will be computed from reference windows
        self.window_step = window_step

        # store pol as an attribute for convenience
        self.pol = self.sigma0.pol

    def _heading_resampled(self, factor):
        if self.heading is None:
            return None
        return self.heading.isel(atrack=slice(0, None, factor), xtrack=slice(0, None, factor))

    def _sigma0_resampled(self, factor):
        __sigma0 = self.sigma0.isel(atrack=slice(0, None, factor), xtrack=slice(0, None, factor))
        __sigma0.values = np.stack(
            [
                cv2.resize(__sigma0.sel(pol=p).values, __sigma0.sel(pol=p).shape[::-1], cv2.INTER_AREA) for p in
                __sigma0.pol
            ]
        )
        return __sigma0

    @property
    @lru_cache
    def _ref_gradient(self):
        return Gradients(
            self._sigma0_resampled(self._ref_downscale_factors),
            heading=self._heading_resampled(self._ref_downscale_factors),
            ortho=self.ortho,
            window_size=self._ref_windows_sizes // self._ref_downscale_factors,
            window_step=self.window_step
        )

    @lru_cache
    def gradients(self, downscale_factor, window_size):
        """
        Get `Gradient` instance for given downscale_factor and window size
        """
        if downscale_factor == self._ref_downscale_factors and window_size == self._ref_windows_sizes:
            return self._ref_gradient
        else:
            return Gradients(
                self._sigma0_resampled(downscale_factor),
                heading=self._heading_resampled(downscale_factor),
                ortho=self.ortho,
                window_size=window_size // downscale_factor,
                align_with=self._ref_gradient
            )

    @lru_cache
    def main_gradients(self, method='max', complex=False, interpolate=True):
        """

        Parameters
        ----------
        method: str
            'max' (histogram maximum) or 'peaks' (up to 2 locals maximums)
        interpolate: bool
            if True (the default), gradient will be interpolated onto original sigma0 grid.
            if False, only gradients at rolling windows center will be returned.
        complex: bool
            if True, gradient are returned as complex number.

        Returns
        -------
        xarray.Dataset
            with 'grad_dir' and 'grad_weight' variable if complex is False, or 'grad' variable if complex.
        """

        if interpolate:
            # interpolate gradient on original sigma0 grid
            atracks = self.sigma0.atrack
            xtracks = self.sigma0.xtrack
        else:
            # windows are not exactly aligned because some float are rounded to int
            # so an interpolation is needed to align axis
            # this will only induce a tiny values change, as interpolated coordinates
            # are very close to data coordinates
            atracks = self._ref_gradient.rolling_gradients.atrack
            xtracks = self._ref_gradient.rolling_gradients.xtrack

        # main gradients is a concatenation of gradients for all downscale_factors and windows sizes
        grad_dir = xr.concat(
            [
                xr.concat(
                    [
                        # as complex argument is [-pi,+pi], interpolation is done squared
                        np.sqrt(
                            (
                                    self.gradients(f, s).main_gradients(
                                        method=method,
                                        interpolate=False,
                                        complex=True
                                    )['grad'] ** 2
                            ).interp(
                                atrack=atracks,
                                xtrack=xtracks
                            )
                        )
                        for f in self.downscale_factors
                    ], dim='downscale_factor'
                ).assign_coords(downscale_factor=self.downscale_factors) for s in self.windows_sizes
            ], dim='window_size'
        ).assign_coords(window_size=self.windows_sizes)

        if not complex:
            # convert complex to deg and module
            grad_ds = xr.merge(
                [
                    -xr.ufuncs.rad2deg(xr.ufuncs.angle(grad_dir)).rename('grad_dir'),
                    np.abs(grad_dir).rename('grad_weight')
                ])
        else:
            # keep as complex
            grad_ds = grad_dir.to_dataset(name='grad')
        return grad_ds



def local_gradients(I, ortho=True):
    """
    compute local multi_gradients

    Parameters
    ----------
    I: xarray.DataArray with dims['atrack', 'xtrack']
        ( from ref article, it's should be 200m resolution )
    ortho: bool
        If True, return the orthogonal gradients.

    Returns
    -------
    xarray.Dataset (G2, c)
            - G2: complex multi_gradients, half size of I (ie 400m resolution)
            - c : G2 quality

    Notes
    -----
        G2 complex multi_gradients are squared,  so any gradient and its negative yield the same value.
        real angle can be retrieved with `np.angle(np.sqrt(G2))`

    """

    grad_r = cv2.Scharr(I.values, cv2.CV_64F, 0, 1)
    grad_i = cv2.Scharr(I.values, cv2.CV_64F, 1, 0)

    # to complex
    grad = xr.zeros_like(I, dtype=np.complex128)
    # why not `+1j*grad_i` ? Probably due to opencv indexing ?
    grad.values = grad_r - 1j * grad_i

    if ortho:
        # orthogonal gradient
        grad = grad * 1j * (np.pi / 2)

    # squared,  so any gradient and its negative yield the same value
    grad12 = grad ** 2

    # 2 factor resize
    grad2 = R2(grad12)
    grad2.name = 'G2'

    # grad quality
    grad3 = R2(abs(grad12))
    c = abs(grad2) / (grad3 + 0.00001)
    c = c.where(c <= 1).fillna(0)
    c.name = 'c'

    return xr.merge([grad2, c])


def convolve2d(in1, in2, boundary='symm', fillvalue=0, dask=True):
    """
    wrapper around scipy.signal.convolve2d for in1 as xarray.DataArray
    mode is forced to 'same', so axes are not changed.
    """
    # FIXME: to be removed, as dask is not used
    try:
        _ = in1.data.map_overlap
        parallel = True
    except:
        parallel = False

    # dict mapping boundary convolve to map_overlap option
    boundary_map = {
        'symm': 'reflect',
        'wrap': 'periodic',
        'fill': fillvalue
    }

    res = in1.copy()
    if parallel and dask:
        # wrapper so every args except in1 are by default
        def _conv2d(in1, in2=in2, mode='same', boundary=boundary, fillvalue=fillvalue):
            return signal.convolve2d(in1, in2, mode=mode, boundary=boundary)

        # make sure the smallest in1 chunk size is >= in2.shape.
        min_in1_chunk = tuple([min(c) for c in in1.chunks])
        if np.min(np.array(min_in1_chunk) - np.array(in2.shape)) < 0:
            raise IndexError("""Some chunks are too small (%s).
            all chunks must be >= %s.
            """ % (str(in1.chunks), str(in2.shape)))
        res.data = in1.data.map_overlap(_conv2d, depth=in2.shape, boundary=boundary_map[boundary])
    else:
        res.data = signal.convolve2d(in1.data, in2, mode='same', boundary=boundary)

    return res


def R2(image):
    """
    reduce image by factor 2, with no moire effect

    Parameters
    ----------
    image: xarray.DataArray with dims ['atrack', 'xtrack']

    Returns
    -------
    xarray.DataArray
        resampled
    """

    B2 = np.mat('[1,2,1; 2,4,2; 1,2,1]', float) * 1 / 16
    B2 = np.array(B2)
    B4 = signal.convolve(B2, B2)

    # pre smooth
    _image = convolve2d(image, B4, boundary='symm')
    num = convolve2d(xr.ones_like(_image), B4, boundary='symm')
    image = _image / num

    # resample
    image = image.coarsen({'atrack': 2, 'xtrack': 2}, boundary='trim').mean()

    # post-smooth
    _image = convolve2d(image, B2, boundary='symm')
    num = convolve2d(xr.ones_like(_image), B2, boundary='symm')
    image = _image / num

    return image


def gradient_histogram(g2, c, angles_bins):
    """
        internal function that compute histogram from local_gradients for only on small box.

        Parameters
        ----------
        g2: numpy.ndarray
            2D array of g2 values from local_gradients
        c: numpy.ndarray
            2D array of g2 values from local_gradients
        angles_bins: numpy.ndarray
            1D array of regulary spaced angles from ]-180,  180[

        Returns
        -------
        numpy.ndarray
            1D array with same shape as angles_bins, with histogram values
        """
    # pixel count in the box
    count = g2.size

    # weighted multi_gradients classes
    degree = np.angle(g2, deg=True)

    # so given an angle deg, the corresponding index in angles_bin is np.round((deg-angles_start)/angles_step)
    angles_step = angles_bins[1] - angles_bins[0]
    angles_start = angles_bins[0]
    k_all = np.round((degree - angles_start) / angles_step)

    # output array
    grads = np.zeros_like(angles_bins, dtype=np.complex128)

    # filter nan
    abs_g2 = np.abs(g2)
    mask = ~np.isnan(abs_g2) & (abs_g2 > 0)
    abs_g2 = abs_g2[mask]
    c = c[mask]
    g2 = g2[mask]
    k_all = k_all[mask]

    r = abs_g2 / (abs_g2 + np.median(abs_g2))

    grads_all = r * c * g2 / abs_g2
    # filter nan
    mask = ~np.isnan(k_all) & ~np.isnan(grads_all)
    grads_all = grads_all[mask]
    k_all = k_all[mask].astype(np.int64)

    np.add.at(grads, k_all, grads_all)

    return grads, g2.size / count


def grad_hist_smooth(hist):
    """
    Smooth hist returned by _grad_hist with kernels Bx Bx2 Bx4 Bx8.
    Histogram coordinates are angles, so begin and end are circulary wrapped.

    Parameters
    ----------
    hist: xarray.DatArray, with 'angles' dim.

    Returns
    -------
    xarray.DataArray
      same as hist, but smoothed.

    """
    Bx = np.array([1, 2, 1], float) * 1 / 4
    Bx2 = np.array([1, 0, 2, 0, 1], float) * 1 / 4
    Bx4 = np.array([1, 0, 0, 0, 2, 0, 0, 0, 1], float) * 1 / 4
    Bx8 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1], float) * 1 / 4
    Bs = [Bx, Bx2, Bx4, Bx8]

    # circular wrap
    maxsize_B = max([len(B) for B in Bs])
    smooth_hist = hist.pad({'angles': maxsize_B}, mode='wrap')

    for B in Bs:
        smooth_hist = xr.apply_ufunc(
            signal.convolve, smooth_hist, B, kwargs={'mode': 'same'},
            input_core_dims=[["angles"], ["kernel_len"]],
            output_core_dims=[['angles']],
            vectorize=True,
            output_dtypes=[np.complex128],
        )

    # unwrap
    smooth_hist = smooth_hist.isel(angles=slice(maxsize_B, -maxsize_B))

    return smooth_hist


def find_gradient(smooth_hist, method='max'):
    """
     Get gradient(s) from smooth hist.

     Parameters
     ----------
     smooth_hist: xarray.DataArray with 'angles' dim.
     method: 'max' or 'peaks'

     Returns
     -------
     xarray.Dataset with 'angle' dim removed
        one variable 'deg' containing the selected gradient
        if method == 'peaks', dataset also contain 'weight'

     Notes
     _____
     method 'max' is from from `Koch(2004)`.
     method 'peaks' return all peaks in histogram

     """

    sqrt_hist = np.sqrt(smooth_hist)
    # should also apply sqrt on 'angles' coordinate
    sqrt_hist['angles'] = np.rad2deg(np.angle(np.sqrt(np.exp(1j * np.deg2rad(sqrt_hist.angles)))))
    if method == 'max':
        ideg = np.abs(sqrt_hist).fillna(0).argmax(dim='angles')
        peak = sqrt_hist.isel(angles=ideg)
    elif method == 'peaks':
        def find_peaks1D(hist, num=2):
            ipeaks = signal.argrelextrema(np.abs(hist), comparator=np.greater, order=2, mode='wrap')[0]
            peaks = hist[ipeaks][0:num]  # keep only first num
            peaks = np.pad(peaks, (0, num - peaks.size))  # pad if too few peaks
            weight = np.argsort(np.abs(peaks))
            # ordered by descending weight
            peaks = peaks[weight][::-1]
            # return as tuple
            return tuple(peaks)

        num = 2  # max number of peaks to find
        peak = xr.concat(
            xr.apply_ufunc(
                find_peaks1D, sqrt_hist, kwargs=dict(num=num),
                input_core_dims=[["angles"]],
                output_core_dims=[[]] * num, vectorize=True, output_dtypes=[np.complex] * num),
            dim="peak")
    else:
        raise KeyError("method %s doesn't exist" % method)
    return peak.drop('angles', errors='ignore')


####
# following code is to build an interactive plot to dig into multigradients
# it's overloaded if there is multiple resolutions and scale.
# we might add it as a __repr__ method to MultiGradients if we found a way to make it
# more clean

try:
    import holoviews as hv
    import panel as pn
except ImportError:
    hv = None
    pn = None


class MultiGradientsViewer():
    """
    MultiGradients interactive viewer for notebook.
    This is work in progress. figures are overloaded, with missings legends ...
    """
    def __init__(self, multi_gradients):
        """

        Parameters
        ----------
        multi_gradients: MultiGradients
        """
        if hv is None or pn is None:
            raise ModuleNotFoundError('holoviews and panel required')
        self.multi_gradients = multi_gradients
        self._pol_line_dash = ['solid', 'dashed']
        self._peak_alpha = [1, 0.5, 0.3, 0.1]
        self._downscale_line_width = self.multi_gradients.downscale_factors
        self._windows_colors = ['b', 'g', 'orange', 'r']
        # get amplitude from 1st resolution, 1st pol
        self.pols = self.multi_gradients.pol
        self.sigma0 = self.multi_gradients.sigma0.isel(pol=0)
        self.main_gradients = self.multi_gradients.main_gradients(method='peaks', interpolate=False)
        # add radian for VectorField
        self.main_gradients['grad_dir_rad'] = xr.ufuncs.deg2rad(self.main_gradients['grad_dir'])

        self.global_view, pipe = self._global_view()
        self.local_view = hv.DynamicMap(self._local_view, streams=[pipe])
        self.all_view = pn.Row(self.local_view, self.global_view)
        """holoview and panel object"""

    def _global_view(self):
        img = hv.Image(self.sigma0).opts(cmap='gray',
                                         clim=(np.nanpercentile(self.sigma0, 5), np.nanpercentile(self.sigma0, 99)))
        atrack_center = self.sigma0.atrack[self.sigma0.atrack.size // 2].item()
        xtrack_center = self.sigma0.xtrack[self.sigma0.xtrack.size // 2].item()
        mouse = hv.streams.Tap(x=atrack_center, y=xtrack_center, source=img)
        pipe = hv.streams.Pipe(data=[atrack_center, xtrack_center])

        def send_pointer(x=0, y=0):
            # send mouse to local_view
            pipe.send((x, y))
            # local_view has set self.__hv_windows, corresponding to all windows at different downscale
            return self.__hv_windows

        quivers = []
        for pol, line_dash in zip(self.pols, self._pol_line_dash):
            for f, line_width in zip(self.main_gradients.downscale_factor, self._downscale_line_width):
                for w, color in zip(self.main_gradients.window_size, self._windows_colors):
                    quivers.append(
                        hv.VectorField(
                            self.main_gradients.sel(peak=0, downscale_factor=f, window_size=w, pol=pol),
                            vdims=['grad_dir_rad', 'grad_weight']
                        ).opts(
                            arrow_heads=False,
                            pivot='mid',
                            magnitude='grad_weight',
                            line_dash=line_dash,
                            scale=0.2, color=color, line_width=line_width)
                    )

        global_view = (img * hv.Overlay(quivers) * hv.DynamicMap(send_pointer, streams=[mouse])) \
            .opts(frame_width=700, frame_height=700, axiswise=False)
        return global_view, pipe

    def _local_view(self, data):
        atrack = data[0]
        xtrack = data[1]

        hv_windows = []

        for f, line_width in zip(self.multi_gradients.downscale_factors, self._downscale_line_width):
            for s, color in zip(self.multi_gradients.windows_sizes, self._windows_colors):
                g = self.multi_gradients.gradients(f, s)
                # get the nearest window
                window = g.rolling_gradients.sel(atrack=atrack, xtrack=xtrack, method='nearest').isel(pol=0)

                # find axtrack index in original G2 array
                idx = {
                    ax: np.nonzero(g.local_gradients['G2'][ax].values == window[ax].item())[0].item()
                    for ax in ['atrack', 'xtrack']
                }

                # extract coordinates from original G2 array
                atracks, xtracks = [
                    g.local_gradients['G2'][ax].isel(
                        {
                            ax: slice(idx[ax] - g.window[ax] // 2, idx[ax] + g.window[ax] // 2)
                        }
                    ) for ax in ['atrack', 'xtrack']
                ]

                # get windows coordinates
                amin, amax, xmin, xmax = (atracks.min(), atracks.max(), xtracks.min(), xtracks.max())
                hv_windows.append(
                    hv.Path(
                        [[(amin, xmin), (amin, xmax), (amax, xmax), (amax, xmin), (amin, xmin)]]
                    ).opts(
                        color=color,
                        line_width=line_width
                    )
                )

        # windows coordinates to be retrieved by global_view (in send_pointer )
        self.__hv_windows = hv.Overlay(hv_windows)

        # set atrack and xtrack from window center, not from mouse
        atrack, xtrack = (atracks[atracks.size // 2].item(), xtracks[xtracks.size // 2].item())

        local_view_hists = []
        for pol, line_dash in zip(self.pols, self._pol_line_dash):
            hv_streaks_dir = []
            hv_hists = []
            for f, line_width in zip(self.multi_gradients.downscale_factors, self._downscale_line_width):
                for s, color in zip(self.multi_gradients.windows_sizes, self._windows_colors):
                    gradient = self.multi_gradients.gradients(f, s)
                    # get histogram
                    hist = gradient.smoothed_direction_histogram.sel(
                        atrack=atrack, xtrack=xtrack,
                        method='nearest'
                    ).sel(pol=pol)

                    sqrt_hist = np.sqrt(hist)

                    # sort by angles
                    sqrt_hist = sqrt_hist[np.argsort(np.angle(sqrt_hist))]

                    # add symmetric points to get 360° histogram
                    hist_pt = list(zip(
                        np.concatenate((np.real(sqrt_hist.values), -np.real(sqrt_hist.values))),
                        np.concatenate((-np.imag(sqrt_hist.values), np.imag(sqrt_hist.values)))
                    ))

                    # connect last point to first point
                    hist_pt.append(hist_pt[0])

                    # plot
                    hv_hist = hv.Path(hist_pt).opts(
                        aspect='equal',
                        show_grid=True,
                        frame_width=400,
                        frame_height=400,
                        axiswise=False,
                        framewise=False,
                        color=color, line_width=line_width, line_dash=line_dash)
                    hv_hists.append(hv_hist)

                    # get main streaks (peaks)
                    streaks_dir = gradient.main_gradients(
                        method='peaks',
                        interpolate=False,
                        complex=True
                    ).sel(atrack=atrack, xtrack=xtrack, method='nearest').sel(
                        pol=pol)  # * f * (self.multi_gradients.windows_sizes[0]/s)

                    streaks_dir = xr.merge(
                        [
                            -xr.ufuncs.angle(streaks_dir['grad']).rename('grad_dir_rad'),
                            np.abs(streaks_dir['grad']).rename('grad_weight')
                        ])

                    for peak, alpha in zip(streaks_dir.peak.values, self._peak_alpha):
                        peak_ds = streaks_dir.sel(peak=peak)
                        hv_streaks_dir.append(
                            hv.VectorField(
                                (0, 0, peak_ds['grad_dir_rad'].item(), peak_ds['grad_weight'].item())
                            ).opts(
                                arrow_heads=False,
                                rescale_lengths=False,
                                color=color,
                                line_dash=line_dash,
                                line_width=line_width,
                                alpha=alpha,
                                magnitude='Magnitude',
                                scale=0.1
                            )
                        )
            local_view_hists.append(hv_hist * hv.Overlay(hv_streaks_dir) * hv.Overlay(hv_hists))
        return hv.Overlay(local_view_hists).opts(xlim=(-4, 4), ylim=(-4, 4))
