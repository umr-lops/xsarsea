"""
Implemented from:

'W. Koch, "Directional analysis of SAR images aiming at wind direction," IEEE Trans. Geosci. Remote Sens., vol. 42, no. 4, pp. 702-710, 2004.'

https://ieeexplore.ieee.org/document/1288365

https://www.climate-service-center.de/imperia/md/content/gkss/institut_fuer_kuestenforschung/ksd/paper/kochw_ieee_2004.pdf

"""

__all__ = ['Gradients', 'Gradients2D', 'circ_smooth', 'PlotGradients', 'histogram_plot']

import numpy as np
from scipy import signal
import xarray as xr
import warnings
import cv2
from functools import reduce
from operator import mul
import pandas as pd

try:
    from xsar.utils import timing
except ImportError:
    # null decorator
    def timing(func):
        return func

# holoviews and panel are not mandatory
try:
    import holoviews as hv
    import panel as pn
except ImportError:
    hv = None
    pn = None

import logging

logging.basicConfig()
logger = logging.getLogger('xsarsea.streaks')
logging.captureWarnings(True)

from abc import ABC, abstractmethod


class Gradients2D:
    """Low level gradients analysis class, for mono pol (2D) sigma0, and single windows size"""

    def __init__(self, sigma0, window_size=1600, window_step=None, windows_at=None):
        """

        Parameters
        ----------
        sigma0: xarray.DataArray

            mono-pol, at medium resolution (like 100m)

        window_size: int

            window size, axtrack coordinate (so it's independent of sigma0 resolution).

            if sensor pixel size is 10m, 1600 will set a 16km window size (i.e 160 pixels if sigma0 is 100m, or 80 pixels if sigma0 res is 200m).

        window_step: float

            stepping of windows sliding. (0.5 is half of the windows, 1 is for non overlapping windows)

        windows_at: dict
        """
        if window_step is not None and windows_at is not None:
            raise ValueError('window_step and window_at are mutually exclusive')
        if window_step is None and windows_at is None:
            window_step = 1
        self.sigma0 = sigma0

        self._spatial_dims = ['atrack', 'xtrack']

        # image will be resampled by a factor 4 before rolling window
        # so, to get a windows of 16 km for a 100m pixel, window size will be 40*40 ( not 160*160 )
        self.window_size = window_size
        # with coords to pixels: int(np.mean(tuple( window_size / np.unique(np.diff(ax))[0] for ax in [self.sigma0.atrack, self.sigma0.xtrack])))

        self._window_dims = {k: "k_%s" % k for k in self._spatial_dims}

        self.n_angles = 72
        """Bin angles count, in the range [-pi/2, pi/2] (can be changed)"""
        self.window_step = window_step
        self._windows_at = windows_at

    @property
    def histogram(self):
        """
        direction histogram as a xarray.Dataset, for all windows from `self.stepping_gradients`.
        This is the main attribute needed by user.
        """
        angles_bins = np.linspace(-np.pi / 2, np.pi / 2, self.n_angles + 1)  # one extra bin
        angles_bins = (angles_bins[1:] + angles_bins[:-1]) / 2  # suppress extra bin (middle)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stepping_gradients = self.stepping_gradients
            grad_hist, ratio = xr.apply_ufunc(
                gradient_histogram,
                stepping_gradients['G2'], stepping_gradients['c'], angles_bins,
                input_core_dims=[self._window_dims.values(), self._window_dims.values(), ["angles"]],
                exclude_dims=set(self._window_dims.values()),
                output_core_dims=[['angles'], []],
                vectorize=True,
                output_dtypes=[np.float, np.float]
            )
            grad_hist = grad_hist.rename('weight').assign_coords(angles=angles_bins)
            ratio = ratio.rename('used_ratio').fillna(0)
        _histogram = xr.merge((grad_hist, ratio))
        # normalize histogram so values are independents from window size
        window_pixels = mul(*(stepping_gradients[k].size for k in self._window_dims.values()))
        _histogram['weight'] = _histogram['weight'] / window_pixels
        return _histogram

    @property
    def window_pixels(self):
        # pixel count per window
        return reduce(mul, self.window.values())

    @property
    def i2(self):
        """reduced sigma0 by factor 2, without moiré effect"""
        return R2(self.sigma0)

    @property
    def ampl(self):
        """amplitude (sqrt) of i2"""
        return np.sqrt(self.i2)

    @property
    def local_gradients(self):
        """local gradients from amplitude as an xarray.Dataset
        ( reduced by a factor 2 from self.ampl, so a factor 4 from self.sigma0),
        with 'G2' variable that is a complex number whose argument is the direction and module the gradient weight,
        and 'c' variable that is a quality index"""
        return local_gradients(self.ampl)

    @property
    def rolling_gradients(self):
        """rolling window over `self.local_gradient` (all, with step 1)"""
        lg = self.local_gradients
        # self.window_size is in axtrack coordinate, and we want it in pixels of lg
        window_size = np.mean(
            tuple(self.window_size / np.unique(np.diff(ax))[0] for ax in [lg.atrack, lg.xtrack])
        )
        window = {k: int(window_size)  for k in self._spatial_dims}
        return lg.rolling(window, center=True).construct(self._window_dims)

    @property
    def windows_at(self):
        """
        dict

            Dict `{'atrack': dataarray, 'xtrack': dataarray}`
            of windows center coordinates where locals histograms are computed.

            By default, those coordinates are computed by sliding windows with a step from `windows_step`

            This property can be set by user.
        """
        if self._windows_at is None and self.window_step is not None:
            # windows_at computed from window_step
            # self.window_size is in axtrack coordinate, and we want it in pixels of self.sigma0
            window_size = np.mean(
                tuple(self.window_size / np.unique(np.diff(ax))[0] for ax in [self.sigma0.atrack, self.sigma0.xtrack])
            )
            self._windows_at = {
                k: self.sigma0[k].isel(
                    {
                        k: np.linspace(
                            1, self.sigma0[k].size,
                            num=int(self.sigma0[k].size / (window_size / (1 / self.window_step))),
                            dtype=int
                        ) - 1
                    }
                )
                for k in self._spatial_dims
            }
        return self._windows_at

    @windows_at.setter
    def windows_at(self, windows_at):
        self._windows_at = windows_at

    @property
    def stepping_gradients(self):
        return self.rolling_gradients.sel(self.windows_at, method='nearest')


class StackedGradients:
    """
    Intermediate class between Gradients2D and Gradients.
    (several Gradient2D are stacked along a `stacked` dimension.
    """
    def __init__(self, gradients):
        """
        Parameters
        ----------
        gradients: list of `Gradients2D` objects.
            All windows will be aligned onto the first `Gradients2D` instance.
            (i.e. `windows_at` attr from others will be set from the 1st one)
        """
        # reference gradient is first gradient
        self._ref_gradient = gradients[0]
        self._others_gradients = gradients[1:]
        # align others gradients on reference gradient
        for g in self._others_gradients:
            g.windows_at = self._ref_gradient.windows_at

    @property
    def histogram(self):
        """Like `Gradients2D.histogram`, but with an added `stacked` dimension"""
        ref_hist = self._ref_gradient.histogram
        aligned_hists = [
            g.histogram.interp(
                atrack=ref_hist.atrack,
                xtrack=ref_hist.xtrack
            ) for g in self._others_gradients
        ]
        return xr.concat([ref_hist] + aligned_hists, dim='stacked')


class Gradients:
    """Gradients class to compute weighted direction histogram at multiscale and multi resolution """

    def __init__(self, sigma0, windows_sizes=[1600], downscales_factors=[1],  window_step=1):
        """

        Parameters
        ----------
        sigma0 : xarray.DataArray

            sigma0 at medium resolution (i.e 100m), with optional 'pol' dimension.

        windows_sizes: list of int

            list of windows size, like `[160, 320]`.

            to have 16km and 32km windows size (if input sigma0 resolution is 100m)

        downscales_factors: list of int

            list of downscale factors, like `[1,2]`

            (for 100m and 200m multi resolution  if input sigma0 resolution is 100m)

        window_step: float

            The overlapping factor for windows slidind. 0.5 is for half overlaping, 1 is no overlaping.

            Note that this overlapping is only for  `windows_sizes[0]` : others windows sizes are aligned with the 1st one.

        """
        self._drop_pol = False
        if 'pol' not in sigma0.dims:
            sigma0 = sigma0.expand_dims('pol')
            self._drop_pol = True
        self.sigma0 = sigma0

        # added dims to histogram
        self._add_dims = ['pol', 'downscale_factor', 'window_size']

        self._dims_values = {
            'pol': sigma0.pol.values,
            'window_size': windows_sizes,
            'downscale_factor': downscales_factors
        }

        self.gradients_list = []

        # append all gradients
        for p in sigma0.pol.values:
            for df in downscales_factors:
                sigma0_resampled = Gradients._sigma0_resample(sigma0.sel(pol=p), df) \
                    .assign_coords(downscale_factor=df)
                for ws in windows_sizes:
                    self.gradients_list.append(
                        Gradients2D(
                            sigma0_resampled.assign_coords(window_size=ws),
                            window_size=ws // df,  # adjust by df, so window_size refer to full scale sigma0
                        )
                    )

        # 1st gradient define windows_at from window_step for all others gradients
        self.gradients_list[0].window_step = window_step

        self.stacked_gradients = StackedGradients(self.gradients_list)

    @property
    @timing
    def histogram(self):
        """
        xarray.Dataset

            With dims:

            - atrack, xtrack : windows centers

            - pol, if sigma0 was provided with pol dim

            - window_size : as `windows_sizes` parameter

            - downscale_factor: as `downscales_factors` parameters

            - angles: histogram dimension (binned angles)
        """
        stacked_hist = self.stacked_gradients.histogram
        hist = xr.merge(
            [
                stacked_hist.isel(stacked=s).expand_dims(self._add_dims)
                for s in range(stacked_hist.stacked.size)
            ]
        )
        if self._drop_pol:
            hist = hist.squeeze('pol', drop=True)

        return hist

    @staticmethod
    def _sigma0_resample(sigma0, factor):
        if factor == 1:
            return sigma0
        __sigma0 = sigma0.isel(atrack=slice(0, None, factor), xtrack=slice(0, None, factor)).copy(True)
        __sigma0.values[::] = cv2.resize(sigma0.values, __sigma0.shape[::-1], cv2.INTER_AREA)
        return __sigma0


class PlotGradients:
    """Plotting class"""
    def __init__(self, gradients_hist):
        """

        Parameters
        ----------
        gradients_hist : xarray.Dataset

            from `Gradients2D.histogram` or mean from `Gradients.histogram`.

            TODO: allow non mean histogram, with multiples dims.
        """
        self.gradients_hist = gradients_hist
        self._spatial_dims = ['atrack', 'xtrack']
        self._non_spatial_dims = list(set(gradients_hist.dims) - set(self._spatial_dims))

        # get maximum histogram
        iangle = np.abs(self.gradients_hist['weight']).fillna(0).argmax(dim='angles')
        self.peak = self.gradients_hist.angles.isel(angles=iangle).to_dataset(name='angle')
        self.peak['used_ratio'] = self.gradients_hist['used_ratio']
        self.peak['weight'] = self.gradients_hist['weight'].isel(angles=iangle)

    def vectorfield(self):
        """Show gradients as a `hv.VectorField` object"""
        vf = hv.VectorField(
            self.peak,
            vdims=['angle', 'weight'],
            kdims=['atrack', 'xtrack'],
            label='mean'
        ).opts(pivot='mid', arrow_heads=False, magnitude='weight')
        return vf

    def linked_plots(self):
        """

        Returns
        -------
            (vector_field, histogram_plot)
        2 holoviews objects connected by mouse.
        """
        vectorfield = self.vectorfield()

        # get mouse pointer from vectorfield
        self._mouse_stream = hv.streams.Tap(x=0, y=0, source=vectorfield)
        # self._mouse_stream = hv.streams.PointerXY(x=0, y=0, source=vectorfield)

        atrack = self.peak.atrack.values[self.peak.atrack.size // 2]
        xtrack = self.peak.xtrack.values[self.peak.xtrack.size // 2]
        # connect mouse to self._pipe_stream ( to draw histogram at mouse position)
        self._pipe_stream = hv.streams.Pipe(data=[atrack, xtrack])
        self._hv_window = hv.Points(self._pipe_stream.data, kdims=['atrack', 'xtrack'])

        def send_mouse(x=0, y=0):
            self._pipe_stream.send((x, y))
            # the callback self.histogram_plot has been called,
            # and has set self._hv_window, corresponding to used window to compute gradient.
            # this is returned to DynamicMap
            return self._hv_window

        # vectorfield will send mouse position
        vf = (
                hv.DynamicMap(send_mouse, streams=[self._mouse_stream]) \
                * vectorfield
        ).opts(axiswise=True)

        # hist get position from mouse
        hist = hv.DynamicMap(
            self.histogram_plot,
            streams=[self._pipe_stream]
        ).opts(frame_width=200, frame_height=200, aspect='equal')

        return vf, hist


    def _histogram_plot_at(self, atrack=None, xtrack=None, data=None):
        # called by histogram_plot to normalize coords
        if data is not None:
            # called by hv streams (like a mouse tap)
            atrack = data[0]
            xtrack = data[1]
        nearest_center = self.peak.sel(atrack=atrack, xtrack=xtrack, method='nearest', tolerance=2000)
        atrack = nearest_center.atrack.values.item()
        xtrack = nearest_center.xtrack.values.item()
        # will have to contain windows as hv.Path
        # self._hv_window = None
        # self._hv_window = hv.Points((atrack, xtrack), kdims=['atrack', 'xtrack'])
        return atrack, xtrack

    def histogram_plot(self, atrack=None, xtrack=None, data=None):
        """plot histogram at atrack, xtrack"""

        # atrack and xtrack are from mouse or user. get the nearest where histogram is defined
        atrack, xtrack = self._histogram_plot_at(atrack=atrack, xtrack=xtrack, data=data)

        # get histogram
        hist_at = self.gradients_hist['weight'].sel(atrack=atrack, xtrack=xtrack, method='nearest', tolerance=2000)

        plot_list = [histogram_plot(hist_at)]

        # vectors plots
        # grad_dir = xr.merge(
        #    [
        #        -xr.ufuncs.angle(grad['grad']).rename('grad_dir_rad'),
        #        np.abs(grad['grad']).rename('grad_weight')
        #    ])
        # try:
        #    peaks = grad_dir.peak.values
        # except AttributeError:
        #    grad_dir = grad_dir.expand_dims('peak')
        #
        # for peak, alpha in zip(grad_dir.peak.values, self._peak_alpha):
        #    peak_ds = grad_dir.sel(peak=peak)
        #    plot_list.append(
        #        hv.VectorField(
        #            (0, 0, peak_ds['grad_dir_rad'].item(), peak_ds['grad_weight'].item())
        #        ).opts(
        #            arrow_heads=False,
        #            rescale_lengths=False,
        #            alpha=alpha,
        #            magnitude='Magnitude',
        #            **self.attrs['plot_style']
        #        )
        #    )
        #
        ## windows shapes will be read by self.iplot() to draw shapes on global map
        # self._hv_window = hv.Path(self.get_nearest_window(atrack, xtrack))
        self._hv_window = hv.Points((atrack, xtrack), kdims=['atrack', 'xtrack'])

        return hv.Overlay(plot_list).opts(xlabel='atrack %d' % atrack, ylabel='xtrack %d' % xtrack)


def local_gradients(I):
    """
    compute local multi_gradients

    Parameters
    ----------
    I: xarray.DataArray with dims['atrack', 'xtrack']
        ( from ref article, it's should be 200m resolution )
    ortho: bool
        If True, return the orthogonal gradients_list.

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
    grad.values = grad_r + 1j * grad_i

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

    # return np.sqrt(grad2) ( so angles are in range [-pi/2, pi/2]
    return xr.merge([np.sqrt(grad2), c])


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
            1D array of regulary spaced angles from ]-pi/2,  pi/2[

        Returns
        -------
        tuple ( numpy.ndarray, float)
            * 1D numpy.ndarray with same shape as angles_bins, with histogram weight values
            * used ratio
        """
    # pixel count in the box
    count = g2.size

    # weighted multi_gradients classes
    angle = np.angle(g2)

    # so given an angle deg, the corresponding index in angles_bin is np.round((deg-angles_start)/angles_step)
    angles_step = angles_bins[1] - angles_bins[0]
    angles_start = angles_bins[0]
    k_all = np.round((angle - angles_start) / angles_step)

    # output array
    grads = np.zeros_like(angles_bins, dtype=np.float)

    # filter nan
    abs_g2 = np.abs(g2)
    mask = ~np.isnan(abs_g2) & (abs_g2 > 0)
    abs_g2 = abs_g2[mask]
    c = c[mask]
    g2 = g2[mask]
    k_all = k_all[mask]

    r = abs_g2 / (abs_g2 + np.median(abs_g2))

    grads_all = r * c  # * g2 / abs_g2
    # filter nan
    mask = ~np.isnan(k_all) & ~np.isnan(grads_all)
    grads_all = grads_all[mask]
    k_all = k_all[mask].astype(np.int64)

    np.add.at(grads, k_all, grads_all)

    return grads, g2.size / count


def circ_smooth(hist):
    """
    Smooth histogram with kernels Bx Bx2 Bx4 Bx8.
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
            output_dtypes=[np.float],
        )

    # unwrap
    smooth_hist = smooth_hist.isel(angles=slice(maxsize_B, -maxsize_B))

    return smooth_hist


def histogram_plot(hist_at):
    """
    Circular histogram plot, as a `hv.Path` object.

    Parameters
    ----------
    hist_at: xarray.Dataset
        Only one histogram (i.e. at one (atrack,xtrack) position.

    Returns
    -------
    hv.Path
    """
    # convert histogram to circular histogram
    # convert to complex
    hist_at = hist_at * np.exp(1j * hist_at.angles)

    # construct a dataframe, with central symmetry, to get 360°
    # columns are  ['atrack_g', 'xtrack_g'], and rows are path points
    hist360_pts = pd.DataFrame(
        np.array(
            [
                np.concatenate([np.real(hist_at.values), -np.real(hist_at.values)]),
                np.concatenate([np.imag(hist_at.values), -np.imag(hist_at.values)])
            ]).T,
        columns=['atrack_g', 'xtrack_g']
    )
    # close path
    hist360_pts = hist360_pts.append(hist360_pts.iloc[0])

    # histogram plot
    return hv.Path(hist360_pts, kdims=['atrack_g', 'xtrack_g']).opts(
        axiswise=False,
        framewise=False,
        aspect='equal',
        # **self.attrs['plot_style']
    )
