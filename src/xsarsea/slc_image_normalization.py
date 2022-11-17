"""
proposition for Nouguier to normalize SLC SAR image with a low pass filter
May 2022
A Grouazel
"""
import logging
import numpy as np
from numpy.fft import fftn, rfftn, irfftn
from scipy.fftpack.helper import next_fast_len
from scipy.ndimage import uniform_filter
from scipy.ndimage.morphology import binary_dilation
def get_normalized_image(sub_g_slc,im_spacing,lowpass_width=[750., 750.]):
    """

    Parameters
    ----------
    sub_g_slc: xarray.DataArray digital numbers from SLC image complexe numbers
    im_spacing : np.array([ccsd['azi_spacing'], ccsd['grdran_spacing']])
    lowpass_width : list azimuth,range width for the low pass filter in meters

    Returns
    -------

    """
    intensity = (np.abs(sub_g_slc) ** 2.).astype('float64')
    # if nbright != 0:
    #     intensity = np.ma.MaskedArray(intensity, mask=bright_mask)
    int_mean = intensity.mean()
    if lowpass_width is not None:
        lowpass_sigma = np.array(lowpass_width) / im_spacing
        logging.debug('lowpass_sigma = %s',lowpass_sigma)
        lowpass = gaussian_lowpass(intensity, lowpass_sigma)
        logging.debug('lowpass %s mean: %s',lowpass.shape,np.mean(lowpass))
    else:
        lowpass = int_mean
    sub_g_slc /= np.sqrt(lowpass)
    return sub_g_slc


def gaussian_lowpass(values, sigma, axes=None, truncate=4., norm=True):
    """
    Multidimensional Gaussian lowpass using FFT-based convolution.
    """
    # Inputs
    vshape = values.shape
    if axes is None:
        axes = range(-len(vshape), 0)
    else:
        if not hasattr(axes, '__iter__'):
            axes = [axes]
    naxes = len(axes)
    if not hasattr(sigma, '__iter__'):
        sigma = [sigma] * naxes
    else:
        if len(sigma) != naxes:
            raise ValueError('sigma length does not match axes length.')
    ismasked = np.ma.is_masked(values)

    # Shapes
    ghshape = [int(truncate * sigma[i] + 0.5) for i in range(naxes)]
    fftshape = [next_fast_len(vshape[axes[i]] + ghshape[i]) for i in range(naxes)]
    outslice = [slice(0, vshape[i]) for i in range(len(vshape))]
    for i in range(naxes):
        outslice[axes[i]] = slice(ghshape[i], ghshape[i] + vshape[axes[i]])
    #print sigma, axes, ghshape, fftshape, outslice

    # Fourier transform input
    #dt0 = datetime.datetime.utcnow()
    if not ismasked:
        lp = rfftn(values, fftshape, axes)
    else:
        lp = rfftn(values.filled(0.), fftshape, axes)
    #print datetime.datetime.utcnow() - dt0

    # Apply Gaussian filter in Fourier domain
    gtfs, rshapes = [], []
    for i in range(naxes):
        xkern = np.arange(-ghshape[i], ghshape[i] + 1, dtype='float')
        gkern = np.exp(-0.5 * (xkern / sigma[i]) ** 2.)
        gkern /= gkern.sum()
        if i < naxes - 1:
            gtf = fftn(gkern, (fftshape[i],))
        else:
            gtf = rfftn(gkern, (fftshape[i],))
        rshape = [1] * len(vshape)
        rshape[axes[i]] = gtf.size
        lp *= gtf.reshape(rshape)
        gtfs.append(gtf)
        rshapes.append(rshape)

    # Inverse Fourier transform
    #dt0 = datetime.datetime.utcnow()
    outslice = tuple(outslice)
    logging.debug('lp %s',lp.shape)
    logging.debug('outslice %s', outslice)
    logging.debug('fftshape %s', fftshape)
    logging.debug('axes %s', axes)
    tmp = irfftn(lp, fftshape, axes)
    logging.debug('tmp : %s',tmp.shape)
    lp = irfftn(lp, fftshape, axes)[outslice]
    #print datetime.datetime.utcnow() - dt0

    # Normalize
    if norm == True:
        if not ismasked:
            for i in range(naxes):
                lp_norm = rfftn(np.ones(vshape[axes[i]]), (fftshape[i],))
                lp_norm *= gtfs[i][:lp_norm.size]
                lp_norm = irfftn(lp_norm, (fftshape[i],))[outslice[axes[i]]]
                rshape = [1] * len(vshape)
                rshape[axes[i]] = lp_norm.size
                lp /= lp_norm.reshape(rshape)
        else:
            lp_norm = rfftn(1. - np.ma.getmaskarray(values), fftshape, axes)
            for i in range(naxes):
                gtf, rshape = gtfs[i], rshapes[i]
                lp_norm *= gtf.reshape(rshape)
            lp_norm = irfftn(lp_norm, fftshape, axes)[outslice]
            null = np.isclose(lp_norm, 0.)
            lp[~null] /= lp_norm[~null]
            lp = np.ma.MaskedArray(lp, mask=null)

    return lp.astype(values.dtype)