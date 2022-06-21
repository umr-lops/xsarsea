"""
IFREMER LOPS SIAM
original code provided by Frederic Nouguier the 29 March 2022
"""
import numpy as np
import xarray as xr
import scipy
import matplotlib.pyplot as plt
import xrft
import logging

def read_slc(slc):
    """
    change coordinates
    Parameters
    ----------
        slc: (xarray.Dataset)
    :return:
    """
    logging.info('max xtrack val : %s',slc['xtrack'].values[-1])
    slc = slc.rename({'atrack' : 'azimuth','xtrack' : 'range'})
    raxis = np.arange(len(slc['range']))
    aaxis = np.arange(len(slc['azimuth']))
    raxis = raxis * slc.attrs['pixel_xtrack_m'] #this one comes from the annotation
    aaxis = aaxis * slc.attrs['pixel_atrack_m']
    logging.info('range coords before : %s',slc['range'].values)
    slc = slc.assign_coords({'range' : raxis,'azimuth' : aaxis}) # I put back the coords change because otherwise I cannot get the energy pattern expected like Nouguier
    logging.info('max range val : %s',raxis[-1])
    logging.info('range coords after : %s',slc['range'].values)
    return slc

def compute_SAR_cross_spectrum2(gslc, *, N_look=3, look_width=0.25, look_overlap=0., look_window=None,
                                range_spacing=None, welsh_window='hanning', nperseg={'range': 512, 'azimuth': 512},
                                noverlap={'range': 256, 'azimuth': 256}, spacing_tol=1e-3, **kwargs):
    """
    Compute SAR cross spectrum using a 2D Welch method. Looks are centered on the mean Doppler frequency
    If ds contains only one cycle, spectrum wavenumbers are added as coordinates in returned DataSet, othrewise, they are passed as variables (k_range, k_azimuth).

    Args:
        gslc (xarray.DataArray): SAR Single Look Complex image, also called Digital Numbers.

    Keyword Args:

        N_look (int): Number of looks
        look_width (float): Percent of the total bandwidth used for a single look in [0,1]
        look_overlap (float): Percent of look overlaping [0,1]. Negative values means space between two looks
        look_window (xarray): window used in look processing
        range_spacing (float, optional): range spacing used in slant range to ground range projection. Automatically chosen if left as None
        welsh_window (str, optional): name of the window used in welsh
        nperseg (dict, optional): dict with keys 'range' and 'azimuth'. Values are the number of points used to define a look shape
        noverlap (dict, optional): dict with keys 'range' and 'azimuth'. Values are the number of overlapping points between two looks
        spacing_tol (float, optional): spacing tolerance of range azimuth sampling step.
        kwargs (dict): other arguments passed to ground_regularization()

    Returns:
        (xarray): SAR NRCS spectrum
    """

    if np.abs(look_width) >= 1: raise ValueError('look_width must be in [0,1] range')
    gslc = gslc.drop(list(set(gslc.coords).intersection(set(['valid_pulse', 'valid_time', 'padded_tau']))))

    stepra = nperseg['range'] - noverlap['range']
    stepaz = nperseg['azimuth'] - noverlap['azimuth']
    indicesx = np.arange(0, gslc.sizes['range'] - nperseg['range'] + 1, stepra)
    indicesy = np.arange(0, gslc.sizes['azimuth'] - nperseg['azimuth'] + 1, stepaz)

    dir = np.diff(gslc['range'].data)  # range spacing
    dia = np.diff(gslc['azimuth'].data)  # azimuth spacing
    if not (np.allclose(dir, dir[0], rtol=spacing_tol) and np.allclose(dia, dia[0], rtol=spacing_tol)):
        raise ValueError("Can't take Fourier transform because coordinate is not evenly spaced")

    frange = np.fft.fftfreq(nperseg['range'], dir[0] / (2 * np.pi))
    #     fazimuth = np.fft.fftfreq(nperseg['azimuth'], dia[0]/(2*np.pi))
    xspecs = list()
    index = list()
    for kx, indx in enumerate(indicesx):
        for ky, indy in enumerate(indicesy):
            index.append([indx, indx + nperseg['range'], indy, indy + nperseg['azimuth']])

    xspecs = compute_looks_threaded(gslc, index)
    fazimuth = np.fft.fftfreq(xspecs[0]['0tau'][0].sizes['freq_azimuth'], dia[0] / (look_width * 2 * np.pi))
    allspecs = list()
    for tau in range(N_look):
        l = [item.drop_vars(['freq_range', 'freq_azimuth']) for sublist in xspecs for item in sublist[str(tau) + 'tau']]
        l = xr.concat(l, dim=str(tau) + 'tau').rename('cross-spectrum_' + str(tau) + 'tau')
        allspecs.append(l)
    allspecs = xr.merge(allspecs, join='outer', fill_value=np.nan)
    allspecs = allspecs.assign_coords(freq_range=np.fft.fftshift(frange))
    allspecs = allspecs.assign_coords(freq_azimuth=np.fft.fftshift(fazimuth))
    allspecs = allspecs.rename(freq_range='kx', freq_azimuth='ky')
    allspecs.kx.attrs.update({'long_name': 'wavenumber in range direction', 'units': 'rad/m'})
    allspecs.ky.attrs.update({'long_name': 'wavenumber in azimuth direction', 'units': 'rad/m'})
    return allspecs


def compute_looks(gslc, *, N_look=3, look_width=0.25, look_overlap=0., look_window=None, plot=False):
    """
    """


    Np = gslc.sizes['azimuth']
    nlook = int(look_width * Np)  # number of pulse in a look
    noverlap = int(
        np.rint(look_overlap * look_width * Np))  # This noverlap is different from the noverlap of welch_kwargs
    mydop = xrft.fft(gslc, dim=['azimuth'], detrend=None, window=None, shift=True, true_phase=True, true_amplitude=True)

    # Finding an removing Doppler centroid
    weight = xr.DataArray(np.hanning(100), dims=['window'])  # window for smoothing
    weight /= weight.sum()
    smooth_dop = np.abs(mydop).mean(dim='range').rolling(freq_azimuth=len(weight), center=True).construct('window').dot(
        weight)
    i0 = int(np.abs(mydop.freq_azimuth).argmin())  # zero frequency indice
    ishift = int(smooth_dop.argmax()) - i0  # shift of Doppler centroid
    mydop = mydop.roll(freq_azimuth=-ishift, roll_coords=False)

    step = nlook - noverlap
    indices = np.arange(0, Np - nlook + 1, step)
    indices = np.concatenate((-np.flipud(indices)[:-1], indices))
    indices += int(i0)
    indices = indices - int(nlook / 2) if N_look % 2 else indices - int(noverlap / 2)
    indices = indices[np.where(np.logical_and(indices > 0, indices < Np - nlook + 1))]
    indices = np.sort(indices[np.argsort(np.abs(indices + int(nlook / 2) - i0))[0:N_look]])

    if look_window is None:
        win = 1.
    elif isinstance(look_window, str) or type(look_window) is tuple:
        win = scipy.signal.get_window(look_window, nlook)
    else:
        win = np.asarray(look_window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if len(win) > Np:
            raise ValueError('window is longer than x.')
        nlook = win.shape
    if plot: plt.figure()
    looks_spec = list()
    # mylooks=list()
    for ind in indices:
        look = xrft.ifft(mydop[{'freq_azimuth': slice(ind, ind + nlook)}].assign_coords(
            {'freq_azimuth': np.arange(-nlook // 2, nlook // 2)}), dim=['freq_azimuth'], detrend=None, window=False,
                         shift=True, true_phase=False, true_amplitude=True)
        looks_spec.append(xrft.fft(np.abs(look) ** 2, dim=['range', 'azimuth'], detrend='linear'))

    if plot: (np.abs(mydop).mean(dim='range') / np.abs(mydop).mean(dim='range').max()).plot()

    looks_spec = xr.concat(looks_spec, dim='look')
    xspecs = dict()
    for l1 in range(N_look):
        for l2 in range(l1, N_look):
            xspec = looks_spec[{'look': l1}] * np.conj(looks_spec[{'look': l2}])
            #  xspec = np.conj(looks_spec[{'look':l1}])*looks_spec[{'look':l2}]
            if str(l2 - l1) + 'tau' in xspecs.keys():
                xspecs[str(l2 - l1) + 'tau'].append(xspec)
            else:
                xspecs[str(l2 - l1) + 'tau'] = [xspec]
    return xspecs


def compute_looks_threaded(gslc, index, **kwargs):
    """
    """
    import threading, multiprocessing

    out = np.empty(len(index), dtype=dict)

    def my_compute_looks(gslc, ind, j):
        sub = gslc[{'range': slice(ind[0], ind[1]), 'azimuth': slice(ind[2], ind[3])}]
        sub.data = scipy.signal.detrend(sub, sub.get_axis_num('range'))
        sub = sub / np.abs(sub).mean(dim=['range', 'azimuth'])
        out[j] = compute_looks(sub)

    tt = [threading.Thread(target=my_compute_looks, args=(gslc, index[j], j)) for j in range(len(index))]
    [t.start() for t in tt]
    [t.join() for t in tt]
    return out