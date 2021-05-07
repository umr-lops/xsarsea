"""
implemented from:
'W. Koch, "Directional analysis of SAR images aiming at wind direction," IEEE Trans. Geosci. Remote Sens., vol. 42, no. 4, pp. 702-710, 2004.'
https://ieeexplore.ieee.org/document/1288365
https://www.climate-service-center.de/imperia/md/content/gkss/institut_fuer_kuestenforschung/ksd/paper/kochw_ieee_2004.pdf
"""

import numpy as np
from scipy import signal
import xarray as xr
import dask.array as da
import warnings
import numba

def streaks_direction(sigma0):
    """

    Parameters
    ----------
    sigma0: xarray.DataArray
        detrended sigma0, at 100m resolution (I1 in Koch(2004))

    Returns
    -------
    xarray.DataArray
        streaks direction, in range [-180,180], at 16km resolution.
        0 deg is azimuth satelite track (not north)

    Notes
    -----
        100m resolution `sigma0_detrend` is not checked.
        Koch(2004) say it should be 100m


    """
    # will work in-memory for numba ufunc
    sigma0 = sigma0.compute()
    if 'pol' in sigma0.dims:
        streaks_dir_list = []
        for pol in sigma0.pol:
            streaks_dir_list.append(_streaks_direction_by_pol(sigma0.sel(pol=pol))
                                    .assign_coords({'pol': pol}))
        streaks_dir = xr.concat(streaks_dir_list, 'pol')
    else:
        streaks_dir = _streaks_direction_by_pol(sigma0)
    return streaks_dir


def _streaks_direction_by_pol(sigma0):
    # internal vectorized function, see streaks_direction

    sigma0 = sigma0.fillna(0).clip(0, None)

    # lower the resolution by a factor 2, without moire effects
    i2 = R2(sigma0, {'atrack': 2, 'xtrack': 2})
    i2 = i2.fillna(0).clip(0, None)

    ampl = np.sqrt(i2)
    G1, G12, G2, G3, c = localGrad(ampl)

    hist = grad_hist(G2, c, window={'atrack': 40, 'xtrack': 40}, n_angles=72)

    smooth_hist = grad_hist_smooth(hist)

    # select best gradient from histogram
    grad_dir = find_gradient(smooth_hist)

    # streaks dir is orthogonal to gradient dir
    streaks_dir = 90 - grad_dir

    # streaks dir is only defined on [-180,180] range (ie no arrow head)
    streaks_dir = xr.where(streaks_dir >= 0, streaks_dir - 90, streaks_dir + 90) % 360 - 180

    return streaks_dir


def convolve2d(in1, in2, boundary='symm', fillvalue=0, dask=True):
    """
    wrapper around scipy.signal.convolve2d for in1 as xarray.DataArray
    mode is forced to 'same', so axes are not changed.
    """

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


def R2(image, reduc):
    """
    resample image by factor

    Parameters
    ----------
    image: xarray.DataArray with dims ['atrack', 'xtrack']
    reduc: dict like { 'atrack' : 2 , 'xtrack' : 2 } (reduce with a factor 2)

    Returns
    -------
    xarray.DataArray
        resampled
    """

    B2 = np.mat('[1,2,1; 2,4,2; 1,2,1]', float) * 1 / 16
    B2 = np.array(B2)
    B4 = signal.convolve(B2, B2)
    try:
        image.data.map_overlap
        ones_like = lambda x: xr.DataArray(da.ones_like(x), dims=x.dims, coords=x.coords)
    except:
        ones_like = xr.ones_like

    # pre smooth
    _image = convolve2d(image, B4, boundary='symm')
    num = convolve2d(ones_like(_image), B4, boundary='symm')
    image = _image / num

    # resample
    image = image.coarsen(reduc, boundary='trim').mean()

    # post-smooth
    _image = convolve2d(image, B2, boundary='symm')
    num = convolve2d(ones_like(_image), B2, boundary='symm')
    image = _image / num

    return image


def localGrad(I):
    """
    compute local gradients

    Parameters
    ----------
    I: xarray.DataArray with dims['atrack', 'xtrack']
        ( from ref article, it's should be 100m resolution )

    Returns
    -------
    tuple of xarray.Dataarray (grad, grad12, grad2, grad3, c)
            - grad : complex gradient, same resolution as I
            - grad12 : grad ** 2
            - grad2 : grad12 resampled by 2 factor
            - grad3 : abs(grad12) resampled by 2 factor
            - c : grad quality

    """
    # local gradient scharr
    Dx = np.mat('[3,0,-3;10,0,-10;3,0,-3]', float) * 1 / 32
    Dy = Dx.T  # transpose
    i = complex(0, 1)
    D = Dx + i * Dy

    # local gradient
    grad = convolve2d(in1=I,in2=D)
    grad.name = 'grad'
    grad12 = grad ** 2  # squared
    grad12.name = 'grad12'
    grad2 = R2(grad12, {'atrack': 2, 'xtrack': 2})
    grad2.name = 'grad2'
    grad3 = R2(abs(grad12), {'atrack': 2, 'xtrack': 2})
    grad3.name = 'grad3'
    # grad quality
    c = abs(grad2) / (grad3 + 0.00001)
    c = c.where(c <= 1).fillna(0)
    c.name = 'c'

    return grad, grad12, grad2, grad3, c


def _grad_hist_one_box(g2, c, angles_bins, grads):
    """
    internal function that compute histogram from localGrad for only on small box.
    this function will be converted to gufunc by numba.

    Parameters
    ----------
    g2: numpy.ndarray
        2D array of g2 values from localGrad
    c: numpy.ndarray
        2D array of g2 values from localGrad
    angles_bins: numpy.ndarray
        1D array of regulary spaced angles from ]-180,  180[
    grads: numpy.ndarray
        *returned* 1D array with same shape as angles_bins, with histogram values
    """
    c_ravel = c.ravel()
    g2_ravel = g2.ravel()
    theta = np.arctan2(g2_ravel.imag, g2_ravel.real)

    # weighted gradients classes
    degree = np.degrees(theta) - 180

    # so given an angle deg, the corresponding index in angles_bin is np.round((deg-angles_start)/angles_step)
    angles_step = angles_bins[1] - angles_bins[0]
    angles_start = angles_bins[0]

    grads[:] = np.complex128(0)

    r = np.abs(g2_ravel) / (np.abs(g2_ravel) + np.median(np.abs(g2_ravel)) + 0.00001)
    r[r > 1] = 0
    for j in range(0, len(degree)):
        deg = degree[j]
        if not np.isnan(deg) and not np.isnan(r[j]) and not np.isnan(c_ravel[j]) and np.abs(
                g2_ravel[j]) != 0:  # evite d avoir des NaN
            # k is the deg index in angles_bins
            k = int(np.round((deg - angles_start) / angles_step))

            grads[k] = grads[k] + r[j] * c_ravel[j] * g2_ravel[j] / np.abs(g2_ravel[j])


# gufunc version of  _grad_hist_one_box that works one many boxes
# g2 and c have shape like [ x, y, bx, by], where bx and by are box shape
_grad_hist_gufunc = numba.guvectorize(
    [(numba.complex128[:, :], numba.float64[:, :], numba.float64[:], numba.complex128[:])], '(n,m),(n,m),(p)->(p)',
    nopython=True)(_grad_hist_one_box)


def grad_hist(g2, c, window, n_angles=72):
    """
    compute gradient histogram from g2 and c by n_angles bins

    Parameters
    ----------
    g2: xarray.DataArray
        2D array from localGrad
    c: xarray.DataArray
        2D array from localGrad, same shape as g2
    window: dict
        window size ie {'atrack': 40, 'xtrack': 40}
    n_angles: angles bins count

    Returns
    -------
    xarray.DataArray
        shape will be reduced by window size, and an 'angle' dim will be added (of size n_angles)

    """

    angles_bins = np.linspace(-180, 180, n_angles + 1)  # one extra bin
    angles_bins = (angles_bins[1:] + angles_bins[:-1]) / 2  # supress extra bin (middle)

    window_dims = {k: "k_%s" % k for k in window.keys()}
    ds = xr.merge([g2.rename('g2'), c.rename('c')])
    try:
        ds_box = ds.rolling(window, center=True).construct(window_dims).sel(
            {k: slice(window[k] // 2, None, window[k]) for k in window.keys()})
    except ValueError as e:
        # too small chunk. adapt message from rolling to usefull infos
        minchunk = { d:  g2.chunks[g2.get_axis_num(d)] for d in window.keys() }
        minwin = { d: window[d] // 2 for d in window.keys()}
        raise ValueError("""Some chunks are too small (%s).
            all chunks must be >= %s. 
            """ % (str(minchunk) , str(minwin)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hist = xr.apply_ufunc(
            _grad_hist_gufunc, ds_box['g2'], ds_box['c'], angles_bins,
            input_core_dims=[window_dims.values(), window_dims.values(), ["angles"]],
            exclude_dims=set(window_dims.values()),
            output_core_dims=[['angles']],
            vectorize=False,
            output_dtypes=[np.complex128]
        )
    hist = hist.rename('angles_hist').assign_coords(angles=angles_bins)

    return hist


def grad_hist_smooth(hist):
    """
    Smooth hist returned by grad_hist with kernels Bx Bx2 Bx4 Bx8.
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


try:
    from xsarsea_protected.streaks import find_gradient
except ImportError:
    def find_gradient(smooth_hist):
        """
         Get maximum gradient from smooth hist.

         Parameters
         ----------
         smooth_hist: xarray.DataArray with 'angles' dim.

         Returns
         -------
         xarray.DataArray with 'angle' dim removed
             selected gradient from smooth_hist (degrees)

         Notes
         _____
         Method from `Koch(2004)`.
         """
        return xr.ufuncs.angle(np.sqrt(smooth_hist).max(dim='angles'), deg=True)
