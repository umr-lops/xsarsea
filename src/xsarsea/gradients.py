"""
Implemented from:

'W. Koch, "Directional analysis of SAR images aiming at wind direction," IEEE Trans. Geosci. Remote Sens., vol. 42, no. 4, pp. 702-710, 2004.'

https://ieeexplore.ieee.org/document/1288365

https://www.climate-service-center.de/imperia/md/content/gkss/institut_fuer_kuestenforschung/ksd/paper/kochw_ieee_2004.pdf

"""

__all__ = ['Gradients', 'Gradients2D',
           'circ_smooth', 'PlotGradients', 'circ_hist']

import numpy as np
from scipy import signal, ndimage
import xarray as xr
import warnings
import cv2
from functools import reduce, partial
from operator import mul
from itertools import product
import pandas as pd
from xsarsea.utils import timing

import logging
logger = logging.getLogger('xsarsea.gradients')
logger.addHandler(logging.NullHandler())

# holoviews and panel are not mandatory
try:
    import holoviews as hv
    import panel as pn
except ImportError:
    hv = None
    pn = None


class Gradients2D:
    """Low level gradients analysis class, for mono pol (2D) sigma0, and single windows size"""

    def __init__(self, sigma0, window_size=1600, window_step=None, windows_at=None):
        """

        Parameters
        ----------
        sigma0: xarray.DataArray

            mono-pol, at medium resolution (like 100m)

        window_size: int

            window size, asample coordinate (so it's independent of sigma0 resolution).

            if sensor pixel size is 10m, 1600 will set a 16km window size (i.e 160 pixels if sigma0 is 100m, or 80 pixels if sigma0 res is 200m).

        window_step: float

            stepping of windows sliding. (0.5 is half of the windows, 1 is for non overlapping windows)

        windows_at: dict
        """
        if window_step is not None and windows_at is not None:
            raise ValueError(
                'window_step and window_at are mutually exclusive')
        if window_step is None and windows_at is None:
            window_step = 1
        self.sigma0 = sigma0

        self._spatial_dims = ['line', 'sample']

        # window size, in asample coordinate
        self.window_size = window_size

        self._window_dims = {k: "k_%s" % k for k in self._spatial_dims}

        self.n_angles = 72
        """Bin angles count, in the range [-pi/2, pi/2] (can be changed)"""
        self.window_step = window_step
        self._windows_at = windows_at

    @property
    @timing(logger=logger.debug)
    def histogram(self):
        """
        direction histogram as a xarray.Dataset, for all windows from `self.stepping_gradients`.
        This is the main attribute needed by user.
        """
        angles_bins = np.linspace(-np.pi / 2, np.pi / 2,
                                  self.n_angles + 1)  # one extra bin
        # suppress extra bin (middle)
        angles_bins = (angles_bins[1:] + angles_bins[:-1]) / 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stepping_gradients = self.stepping_gradients
            grad_hist, ratio = xr.apply_ufunc(
                gradient_histogram,
                stepping_gradients['G2'], stepping_gradients['c'], angles_bins,
                input_core_dims=[self._window_dims.values(
                ), self._window_dims.values(), ["angles"]],
                exclude_dims=set(self._window_dims.values()),
                output_core_dims=[['angles'], []],
                vectorize=True,
                output_dtypes=[np.float, np.float]
            )
            grad_hist = grad_hist.rename(
                'weight').assign_coords(angles=angles_bins)
            ratio = ratio.rename('used_ratio').fillna(0)
        _histogram = xr.merge((grad_hist, ratio))
        # normalize histogram so values are independents from window size
        window_pixels = mul(
            *(stepping_gradients[k].size for k in self._window_dims.values()))
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
        # self.window_size is in asample coordinate, and we want it in pixels of lg
        window_size = np.mean(
            tuple(self.window_size / np.unique(np.diff(ax))
                  [0] for ax in [lg.line, lg.sample])
        )
        window = {k: int(window_size) for k in self._spatial_dims}
        return lg.rolling(window, center=True).construct(self._window_dims)

    @property
    def windows_at(self):
        """
        dict

            Dict `{'line': dataarray, 'sample': dataarray}`
            of windows center coordinates where locals histograms are computed.

            By default, those coordinates are computed by sliding windows with a step from `windows_step`

            This property can be set by user.
        """
        if self._windows_at is None and self.window_step is not None:
            # windows_at computed from window_step
            # self.window_size is in asample coordinate, and we want it in pixels of self.sigma0
            window_size = int(
                np.mean(
                    tuple(self.window_size / np.unique(np.diff(ax))
                          [0] for ax in [self.sigma0.line, self.sigma0.sample])
                )
            )

            step_size = int(window_size * self.window_step)

            ds = self.sigma0.isel(
                line=slice(0, None, step_size),
                sample=slice(0, None, step_size)
            )

            self._windows_at = {
                'line': ds.line,
                'sample': ds.sample
            }
        return self._windows_at

    @windows_at.setter
    def windows_at(self, windows_at):
        self._windows_at = windows_at

    @property
    def stepping_gradients(self):
        # do not call .interp, it's exact, but slow and take much memory
        # return self.rolling_gradients.interp(self.windows_at, method='nearest')
        sg = self.rolling_gradients.sel(self.windows_at, method='nearest')
        sg['line'] = self.windows_at['line']
        sg['sample'] = self.windows_at['sample']
        return sg


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

    @staticmethod
    def _stackable(line, sample, g):
        # internal method for Pool().map
        return g.histogram.interp(line=line, sample=sample, method='nearest')

    @property
    def histogram(self):
        """Like `Gradients2D.histogram`, but with an added `stacked` dimension"""

        ref_hist = self._ref_gradient.histogram

        # list of gradients, with same asample (non parallelized)
        aligned_hists = [
            g.histogram.interp(
                line=ref_hist.line,
                sample=ref_hist.sample
            ) for g in self._others_gradients
        ]
        return xr.concat([ref_hist] + aligned_hists, dim='stacked')


class Gradients:
    """Gradients class to compute weighted direction histogram at multiscale and multi resolution """

    def __init__(self, sigma0, windows_sizes=[1600], downscales_factors=[1], window_step=1):
        """

        Parameters
        ----------
        sigma0 : xarray.DataArray

            sigma0 at medium resolution (i.e 100m), with optional 'pol' dimension.

        windows_sizes: list of int

            list of windows size, like `[1600, 3200]`.

            to have 16km and 32km windows size (if sensor sigma0 resolution is 10m)

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
                            window_size=ws
                        )
                    )

        # 1st gradient define windows_at from window_step for all others gradients
        self.gradients_list[0].window_step = window_step
        self.stacked_gradients = StackedGradients(self.gradients_list)

    @property
    @timing(logger=logger.info)
    def histogram(self):
        """
        xarray.Dataset

            With dims:

            - line, sample : windows centers

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
        __sigma0 = sigma0.isel(line=slice(0, None, factor),
                               sample=slice(0, None, factor)).copy(True)
        __sigma0.values[::] = cv2.resize(
            sigma0.values, __sigma0.shape[::-1], cv2.INTER_AREA)
        return __sigma0


class PlotGradients:
    """Plotting class"""

    def __init__(self, gradients_hist):
        """

        Parameters
        ----------
        gradients_hist : xarray.Dataset

            from `Gradients2D.histogram` or mean from `Gradients.histogram`.

        """
        self.gradients_hist = gradients_hist
        self._spatial_dims = ['sample', 'line']
        # non spatial dims, probably like  ['pol' 'window_dims' 'downscale_factor']
        self._non_spatial_dims = list(
            set(gradients_hist.dims) - set(self._spatial_dims) - set(['angles']))

        # list of dicts, where keys are from self._non_spatial_dims, and values are all possible values for key
        # so by looping this list, all gradients for all non-spatial dims can be accessed
        self.combine_all = [
            dict(zip(self._non_spatial_dims, comb)) for comb in list(
                product(
                    *[self.gradients_hist[k].values for k in self._non_spatial_dims])
            )
        ]

        # styles: only one style allowed per dim
        self.dim_styles = {
            'pol': {'line_dash': ['solid', 'dotted']},
            'downscale_factor': {'line_width': [1, 2, 3, 4]},
            'window_size': {'line_color': ['blue', 'red', 'yellow', 'green']}
        }

        self.styles_names = []

        # add variables named like style, ie self.gradients_hist['line_color']
        # the values are the style values, and the dim name is the dim the style belong to
        for dim, style_dict in self.dim_styles.items():
            for style_name, style_values in style_dict.items():
                try:
                    self.gradients_hist[style_name] = (
                        dim, style_values[:self.gradients_hist[dim].size])
                    self.styles_names.append(style_name)
                except (KeyError, ValueError):
                    # dim is not in self.gradients_hist: ignore
                    pass

        # get maximum histogram
        hist = self.gradients_hist
        iangle = np.abs(hist['weight']).fillna(0).argmax(dim='angles')
        self.peak = hist.angles.isel(angles=iangle).to_dataset(name='angle')
        self.peak['used_ratio'] = hist['used_ratio']
        self.peak['weight'] = hist['weight'].isel(angles=iangle)

        # add styles to self.peak
        for style_name in self.styles_names:
            self.peak[style_name] = self.gradients_hist[style_name]

        self._vectorfield = None

    def _get_style(self, ds):
        # return style for ds, using variables from self.styles_names
        # style is only returned if dim len is 1
        return {st: ds[st].values.item() for st in self.styles_names if (st in ds) and (ds[st].size == 1)}

    def vectorfield(self, tap=True):
        """Show gradients as a `hv.VectorField` object"""
        if self._vectorfield is None:
            vf_list = []
            for sel_one2D in self.combine_all:
                peak2D = self.peak.sel(sel_one2D)
                style = self._get_style(peak2D)
                vf_list.append(
                    hv.VectorField(
                        peak2D,
                        vdims=['angle', 'weight'],
                        kdims=['sample', 'line'],
                    ).opts(pivot='mid', arrow_heads=False, magnitude='weight', aspect='equal', **style)
                )

            # manual legend, to have a style per dimension
            legends = []
            dummy_line = [(0, 0), (0.01, 0)]
            for st in self.styles_names:
                label = self.peak[st].dims[0]
                for item in self.peak[st]:
                    style = {'line_dash': 'solid',
                             'line_width': 1, 'line_color': 'k'}
                    style.update({st: item.item()})
                    legends.append(
                        hv.Curve(
                            dummy_line,
                            label="%s %s" % (label, item[label].item())
                        ).redim.label(x='sample', y='line').opts(**style)
                    )
            self._vectorfield = hv.Overlay(
                vf_list + legends).opts(active_tools=['wheel_zoom', 'pan'])

        if tap:
            line = self.peak.line.values[self.peak.line.size // 2]
            sample = self.peak.sample.values[self.peak.sample.size // 2]
            self._mouse_stream = hv.streams.Tap(
                x=sample, y=line, source=self._vectorfield)
            return self._vectorfield * hv.DynamicMap(self._get_windows, streams=[self._mouse_stream])

        return self._vectorfield

    def mouse_histogram(self, source=None):
        assert self._mouse_stream is not None
        if source is None:
            source = self
        return hv.DynamicMap(source.histogram_plot, streams=[self._mouse_stream]).opts(active_tools=['wheel_zoom'])

    def _get_xline(self, sample=None, line=None, data=None):
        # called by histogram_plot to normalize coords
        if data is not None:
            # called by hv streams (like a mouse tap)
            sample = data[0]
            line = data[1]
        nearest_center = self.peak.sel(
            line=line, sample=sample, method='nearest', tolerance=1e6)
        line = nearest_center.line.values.item()
        sample = nearest_center.sample.values.item()
        return sample, line

    def _get_windows(self, sample=None, line=None, x=None, y=None):

        if x is not None:
            sample = x
        if y is not None:
            line = y

        # line and sample are from mouse or user. get the nearest where histogram is defined
        sample, line = self._get_xline(sample=sample, line=line)

        windows_list = []
        try:
            ws_list = self.gradients_hist['window_size']
        except KeyError:
            # no 'window_size'. compute it from asample neighbors
            ws_list = [
                np.diff(
                    np.array(
                        [[self.gradients_hist[ax].isel({ax: i}).item() for i in [0, 1]] for ax in [
                            'line', 'sample']]
                    )
                ).mean()
            ]

        for ws in ws_list:
            # window as a hv.Path object corresponding to window_size
            amin, amax, xmin, xmax = (
                line - ws / 2, line + ws / 2, sample - ws / 2, sample + ws / 2
            )
            try:
                style = self._get_style(
                    self.gradients_hist.sel(window_size=ws))
            except (IndexError, KeyError):
                style = {}
            windows_list.append(
                hv.Path([[(xmin, amin), (xmin, amax), (xmax, amax),
                        (xmax, amin), (xmin, amin)]]).opts(**style)
            )

        return hv.Overlay(windows_list)

    def histogram_plot(self, sample=None, line=None, x=None, y=None):
        """plot histogram at sample, line"""

        if x is not None:
            sample = x
        if y is not None:
            line = y

        # line and sample are from mouse or user. get the nearest where histogram is defined
        sample, line = self._get_xline(sample=sample, line=line)

        # get histogram
        hist_at = self.gradients_hist.sel(
            line=line, sample=sample, method='nearest', tolerance=500)

        hp_list = []
        for sel_one2D in self.combine_all:
            hist2D_at = hist_at.sel(sel_one2D)
            hist2D360 = circ_hist(hist2D_at['weight'])
            style = self._get_style(hist2D_at)
            hp_list.append(
                hv.Path(hist2D360, kdims=['sample_g', 'line_g']).opts(
                    axiswise=False,
                    framewise=False,
                    aspect='equal', **style)
            )

        return hv.Overlay(hp_list).opts(xlabel='sample %d' % sample, ylabel='line %d' % line, width=200, height=200)


def local_gradients(I):
    """
    compute local multi_gradients

    Parameters
    ----------
    I: xarray.DataArray with dims['line', 'sample']
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

    grad_r = cv2.Scharr(I.values, cv2.CV_64F, 1, 0)
    grad_i = cv2.Scharr(I.values, cv2.CV_64F, 0, 1)

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
    grad3.name = 'G3'
    c = abs(grad2) / (grad3 + 0.00001)
    c = c.where(c <= 1).fillna(0)
    c.name = 'c'

    # return np.sqrt(grad2) ( so angles are in range [-pi/2, pi/2]
    return xr.merge([np.sqrt(grad2), grad3, c])


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
        res.data = in1.data.map_overlap(
            _conv2d, depth=in2.shape, boundary=boundary_map[boundary])
    else:
        res.data = signal.convolve2d(
            in1.data, in2, mode='same', boundary=boundary)

    return res


def smoothing(image):
    # filtre gaussien

    B2 = np.mat('[1,2,1; 2,4,2; 1,2,1]', float) * 1 / 16
    B2 = np.array(B2)

    _image = convolve2d(image, B2, boundary='symm')
    num = convolve2d(xr.ones_like(_image), B2, boundary='symm')
    image = _image / num

    return image


def R2(image):
    """
    reduce image by factor 2, with no moire effect

    Parameters
    ----------
    image: xarray.DataArray with dims ['line', 'sample']

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
    image = image.coarsen({'line': 2, 'sample': 2}, boundary='trim').mean()

    # post-smooth
    _image = convolve2d(image, B2, boundary='symm')
    num = convolve2d(xr.ones_like(_image), B2, boundary='symm')
    image = _image / num

    return image


def Mean(image):
    """
    Local Mean Operator

    Parameters
    ----------
    image: xarray.DataArray with dims ['line', 'sample']

    Returns
    -------
    xarray.DataArray
        smoothed
    """
    B2 = np.mat('[1,2,1; 2,4,2; 1,2,1]', float) * 1 / 16
    B2 = np.array(B2)
    B4 = signal.convolve(B2, B2)

    B22 = np.mat(
        '[1,0,2,0,1;0,0,0,0,0;2,0,4,0,2;0,0,0,0,0;1,0,2,0,1]', float) * 1/16
    B42 = signal.convolve(B22, B22)

    _image = convolve2d(image, B4, boundary='symm')
    num = convolve2d(np.ones_like(_image), B4, boundary='symm')
    image = _image/num

    _image = convolve2d(image, B42, boundary='symm')
    num = convolve2d(np.ones_like(_image), B4, boundary='symm')
    image = _image/num

    return image


def filtering_parameters(image_ori):
    """
    Mask filter parameters definition
    Zhao, Y.; Longépé, N.; Mouche, A.; Husson, R. Automated Rain Detection by Dual-Polarization Sentinel-1 Data.
    Remote Sens. 2021, 13, 3155. https://doi.org/10.3390/rs13163155

    Parameters
    ----------
    image_ori: xarray.DataArray with dims ['line', 'sample']

    Returns
    -------
    list of xarray.DataArray
        f1, f2, f3, f4, F (all between 0 and 1)
    """
    image = np.sqrt(image_ori)  # sqrt de NRCS=amplitude

    # useful parameters
    r2 = R2(image)
    lg = local_gradients(image)
    G3 = lg.G3
    c = lg.c
    J = Mean(r2)

    # P1
    J1 = Mean(r2**2)
    J2 = np.sqrt(J1 - J**2)
    # standart deviation / mean
    P1 = J2/(J+0.00001)
    a1 = -50
    b1 = 2.75

    # P2
    resampl = r2.coarsen({'line': 2, 'sample': 2}, boundary='trim').mean()
    # we compute the exact factor so that the two terms match dimensions in case of odd original dimensions
    K = r2 - ndimage.zoom(smoothing(resampl),
                          (r2.shape[0] / resampl.shape[0], r2.shape[1] / resampl.shape[1]), order=1)
    P2 = K**2 / ((J**2)+0.00001)
    a2 = -5000
    b2 = 3

    # P3
    G4 = Mean(G3)
    P3 = G3/(G4+0.00001)
    a3 = -2.5
    b3 = 4

    # P4
    P4 = np.sqrt(c)
    a4 = -10
    b4 = 6.3

    # set values between 0 & 1
    f1 = np.clip(a1*P1+b1, 0, 1)
    f2 = np.clip(a2*P2+b2, 0, 1)
    f3 = np.clip(a3*P3+b3, 0, 1)
    f4 = np.clip(a4*P4+b4, 0, 1)

    F = np.sqrt(1/4. * (f1**2 + f2**2 + f3**2 + f4**2))

    #  TO CHECK
    if F.shape == image_ori.shape:
        F[F < 0.0015] = 0  # suppress black band # to check

    return f1, f2, f3, f4, F


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

    grads_all = r * c
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
    Bx8 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
                   0, 0, 0, 0, 0, 1], float) * 1 / 4
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


def circ_hist(hist_at):
    """
    convert xarray.Dataset `hist_at` with angles modulo pi to a pandas.Dataframe modulo 2pi

    Parameters
    ----------
    hist_at: xarray.Dataset
        Only one histogram (i.e. at one (line,sample) position.

    Returns
    -------
    pd.DataFrame, with columns ['sample_g', 'line_g']
    """

    # convert histogram to circular histogram
    # convert to complex
    hist_at = hist_at * np.exp(1j * hist_at.angles)

    # central symmetry, to get 360°
    hist_at = xr.concat([hist_at, -hist_at],
                        'angles').drop_vars(['line', 'sample'])
    hist_at['angles'] = np.angle(hist_at)
    hist_at['sample_g'] = np.real(hist_at)
    hist_at['line_g'] = np.imag(hist_at)

    # convert to dataframe (weight no longer needed)
    circ_hist_pts = hist_at.to_dataframe('tmp')[['line_g', 'sample_g']]

    # close path
    circ_hist_pts = pd.concat(
        [circ_hist_pts, pd.DataFrame(circ_hist_pts.iloc[0]).T])

    return circ_hist_pts
