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

#  check map_overlap convolve vs numpy convolve  (memory++ !). TODO: will be deprecated once fixed.
check_convolve = False


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
    depth = {'atrack': B4.shape[0], 'xtrack': B4.shape[1]}
    dataarray = lambda x: xr.DataArray(x, dims=("atrack", "xtrack"),
                                       coords={"atrack": image.atrack, "xtrack": image.xtrack})

    # we will call this function instead of signal.convolve2d,
    # because there is a conflict on `boundary` kwargs with map_overlap
    def convolve2d(in1=None, in2=None, mode='same'):
        return signal.convolve2d(in1, in2, mode=mode, boundary='symm')

    # pre smooth
    _image = dataarray(image.data.map_overlap(convolve2d, in2=B2, depth=depth, boundary=0))
    num = dataarray(da.ones_like(_image).map_overlap(convolve2d, in2=B2, depth=depth, boundary=1))
    image = _image / num

    if check_convolve:  # TODO: remove after fix
        # we check convolve2d called by map_overlap vs numpy
        num = num.persist()
        num_overlap = num.values
        num_numpy = convolve2d(np.ones_like(_image), B4)
        bool_diff = num_numpy != num_overlap
        if np.count_nonzero(bool_diff) == 0:
            warnings.warn('check convolve2d map_overlap vs numpy ok')
        else:
            raise RuntimeError('check convolve2d map_overlap vs numpy failed')

    # resample
    image = image.coarsen(reduc, boundary='pad').mean()

    # post-smooth
    _image = dataarray(image.data.map_overlap(convolve2d, in2=B2, depth=depth, boundary=0))
    num = dataarray(da.ones_like(_image).map_overlap(convolve2d, in2=B2, depth=depth, boundary=1))
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
