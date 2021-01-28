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

# map_overlap convolve vs numpy convolve  (memory++ !). TODO: will be deprecated once fixed.
dask_convolve = True

def convolve2d(in1, in2, boundary='symm', fillvalue=0, dask=dask_convolve):
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
        'symm' : 'reflect',
        'wrap' : 'periodic',
        'fill' : fillvalue
    }

    res = in1.copy()
    if parallel and dask:
        # wrapper so every args except in1 are by default
        def _conv2d(in1,in2=in2,mode='same', boundary=boundary, fillvalue=fillvalue):
            return signal.convolve2d(in1, in2, mode=mode, boundary=boundary)
        res.data = in1.data.map_overlap(_conv2d, in2.shape, boundary=boundary_map[boundary])
    else:
        res.data = signal.convolve2d(in1.data, in2, mode='same', boundary=boundary)

    return res

def R2(image, reduc, dask=False):
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
    ones_like = lambda x: xr.DataArray(da.ones_like(x), dims=x.dims,
                                       coords=x.coords)


    # pre smooth
    _image = convolve2d(image, B4, boundary='symm')
    num = convolve2d(ones_like(_image), B4, boundary='symm')
    image = _image / num

    # resample
    image = image.coarsen(reduc, boundary='pad').mean()

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

    def convolve2d(in1=None, in2=None):
        return signal.convolve2d(in1, in2, mode='same', boundary='symm')

    # local gradient
    grad = xr.DataArray(
        I.data.map_overlap(convolve2d, in2=D, depth={'atrack': D.shape[0], 'xtrack': D.shape[0]}, boundary='symm'),
        dims=("atrack", "xtrack"), coords={"atrack": I.atrack, "xtrack": I.xtrack})
    grad.name = 'grad'
    grad = grad.persist()  # persist into memory, to speedup depending vars computations
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
    degree = np.degrees(theta)

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
            # print('k %s for deg %s' % (k, deg))
            # print('grads[%d]=%f' % (k, grads[k]))
            print(c_ravel[j])

# gufunc version of  _grad_hist_one_box that works one many boxes
# g2 and c have shape like [ x, y, bx, by], where bx and by are box shape
_grad_hist_gufunc = numba.guvectorize([(numba.complex128[:,:], numba.float64[:,:], numba.float64[:], numba.complex128[:])], '(n,m),(n,m),(p)->(p)',nopython=True)(_grad_hist_one_box)


