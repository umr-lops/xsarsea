import numpy as np
from scipy import signal
import xarray as xr
import dask.array as da


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
    B4 = signal.convolve(B2, B2)
    depth = {'atrack': B2.shape[0], 'xtrack': B2.shape[0]}
    dataarray = lambda x: xr.DataArray(x, dims=("atrack", "xtrack"),
                                       coords={"atrack": image.atrack, "xtrack": image.xtrack})

    def convolve2d(in1=None, in2=None):
        return signal.convolve2d(in1, in2, mode='same', boundary='symm')

    _image = dataarray(image.data.map_overlap(convolve2d, in2=B4, depth=depth, boundary=np.nan))
    num = dataarray(da.ones_like(_image).map_overlap(convolve2d, in2=B4, depth=depth, boundary=np.nan))
    image = _image / num

    image = image.coarsen(reduc, boundary='pad').mean()
    _image = dataarray(image.data.map_overlap(convolve2d, in2=B2, depth=depth, boundary=np.nan))
    num = dataarray(da.ones_like(_image).map_overlap(convolve2d, in2=B2, depth=depth, boundary=np.nan))

    image = _image / num
    return image


def localGrad(I):
    """
    compute local gradients

    Parameters
    ----------
    I: xarray.DataArray with dims['atrack', 'xtrack']

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
    grad12 = grad ** 2  # squared
    grad12.name = 'grad12'
    grad2 = R2(grad12, {'atrack': 2, 'xtrack': 2})
    grad2.name = 'grad2'
    grad3 = R2(abs(grad12), {'atrack': 2, 'xtrack': 2})
    grad3.name = 'grad3'
    # grad quality
    c = abs(grad2) / (grad3 + 0.00001)
    c = c.where(c > 1).fillna(0)
    c.name = 'c'

    return grad, grad12, grad2, grad3, c
