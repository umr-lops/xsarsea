"""
author Antoine Grouazel:
original code provided by Frederic Nouguier the 29 March 2022
"""
import numpy as np
import xarray as xr
import scipy
import matplotlib.pyplot as plt
import xrft
import logging
import xsar
import os

def get_imagette_indice(onetiff,wv_slc_meta):
    good_indice = None
    # find the indice of the tiff
    imagette_number = os.path.basename(onetiff).split('-')[-1].replace('.tiff','')
    logging.info('imagette_number : %s',imagette_number)
    for ddi,ddname in enumerate(wv_slc_meta.subdatasets) :
        if 'WV_'+imagette_number in ddname :
            logging.info("matching ddname : %s",ddname)
            good_indice = ddi
    logging.info('indice in meta sentinel1 gdal driver is : %s',good_indice)
    return good_indice

def read_slc(onetiff,slice_subdomain=None,resolution=None,resampling=None):
    """

    :param one_wv: SAFE (str) full path without the last /
    :param slice_subdomain : slice objet to define the sub part of imagette to perform cross spectra
    :param resolution: dict for instance for 200m {'atrack' : int(np.round(200 / sar_meta.pixel_atrack_m)), 'xtrack': int(np.round(200 / sar_meta.pixel_xtrack_m))}
    :param rasterio.enums.Resampling.rms for instance
    :return:
    """
    logging.info('tiff: %s',onetiff)
    logging.info('resampling : %s',resampling)
    safepath = os.path.dirname(os.path.dirname(onetiff))
    logging.info('safepath: %s',safepath)
    wv_slc_meta = xsar.Sentinel1Meta(safepath)
    logging.debug('wv_slc_meta %s',wv_slc_meta)
    good_indice = get_imagette_indice(onetiff,wv_slc_meta)
    if resolution is not None:
        if resolution['atrack']==1 and resolution['xtrack']==1:
            #June 2021, a patch because currently resolution 1:1 for image and rasterio returns an error
            slc = xsar.Sentine1Dataset(wv_slc_meta.subdatasets[good_indice],resolution=None,resampling=None)
        else:
            slc = xsar.Sentine1Dataset(wv_slc_meta.subdatasets[good_indice],resolution=resolution,resampling=resampling)
    else:
        slc = xsar.Sentine1Dataset(wv_slc_meta.subdatasets[good_indice])
    logging.info('max xtrack val : %s',slc['xtrack'].values[-1])
    slc = slc.rename({'atrack' : 'azimuth','xtrack' : 'range'})
    azimuthSpacing,rangeSpacing = slc.s1meta.image['slant_pixel_spacing']
    #rangeSpacing = slc.attrs['pixel_xtrack_m']
    #azimuthSpacing= slc.attrs['pixel_atrack_m']

    platform_heading = slc.attrs['platform_heading']
    logging.debug('slant rangeSpacing %s',rangeSpacing)
    #changement des coordinates range and azimuth like Nouguier
    raxis = np.arange(len(slc['range']))
    aaxis = np.arange(len(slc['azimuth']))
    raxis = raxis * rangeSpacing
    aaxis = aaxis * azimuthSpacing
    logging.info('range coords before : %s',slc['range'].values)
    #slc = slc.assign_coords({'range':raxis,'azimuth':aaxis}) # I turn off the coords change performed by Noug because I think it is already done in xsar (not sure but it can explain the problem of grid I have)
    slc = slc.assign_coords({'range' : raxis,'azimuth' : aaxis}) # I put back the coords change because otherwise I cannot get the energy pattern expected like Nouguier
    logging.info('max range val : %s',raxis[-1])
    logging.info('range coords after : %s',slc['range'].values)
    # en WV on attend du 4.1m environ pour le ground range pixel spacing! dixit doc ESA, Nouguier et annotations (en slant)
    slc.attrs.update({'azimuthSpacing' : azimuthSpacing,'rangeGroundSpacing' : rangeSpacing,
                       'heading' : platform_heading})
    # test agrouaze 18 oct 21
    # slc.attrs.update({'azimuthSpacing' : azimuthSpacing,'rangeSpacing' : rangeSpacing,
    #                    'heading' : platform_heading})
    # compute modulation
    #slc['modulation'] = slc['digital_number']/np.mean(abs(slc['digital_number'].values))#*0.0000000001 - 1000000. #division by 1000 for test
    #slc['modulation'] = slc['digital_number']
    if slice_subdomain is not None:
        logging.info('subset in the image: %s',slice_subdomain)
        slc = slc.isel(range=slice_subdomain['range'],azimuth=slice_subdomain['azimuth'],pol=0)
    return slc

def ground_regularization(ds, *,method='nearest', **kwargs):
    """
    pcopy paste from /home1/datahome/fnouguie/research/numeric/R3S/SAR/postprocessing.py (april 2021)
    Compute ground regularization: Interpolation of ds variables on a regular ground grid.

    Args:
        ds (xarray): xarray on (slant_range, azimuth coordinates)

    Keyword Args:
        method (str): method of interpolation (nearest seems to give the best results)
        range_spacing (float, optional): ground spacing [meter]. If not defined, the mean of projected ranges steps is used. You should provide the NOT padded ground range resolution

    Returns:
        (xarray): same as ds but interpolated on (range, azimuth) coordinates
    """
    import warnings
    warnings.warn('Update slant to ground projection with sinc interpolation.')
    logging.debug('type ds: %s',type(ds))
    #tmp = xr.DataArray(np.array([50000]),dims='altitude',coords={'altitude':np.arange(1)}) #added agrouaze for test


    #ds = ds.rename({'xtrack':'slant_range'})
    #ds = ds.rename({'atrack' : 'azimuth'})
    #
    #ds['altitude'] = tmp*
    #ds.attrs['altitude'] = 0.5
    logging.info('ds.coords : %s',ds.coords)
    if 'range' in ds.coords:
        print('Ground regularization: range coordinates already in SLC object. SLC is assumed to be on ground range coordinates already.')
        return ds
    if 'slant_range' not in ds.coords:
        raise ValueError('Please provide ds with slant_range coordinates')

    if 'altitude' in kwargs:
        altitude = kwargs.pop('altitude')
    elif 'altitude' in ds.attrs:
        altitude = ds.altitude
    else:
        raise ValueError('Please provide range spacing value or altitude in kwargs')
    logging.debug('all finite slant ranges? %s',np.isfinite(ds['slant_range'].values).all())
    logging.debug('min slant %s',ds['slant_range'].values.min())
    ground_ranges = np.sqrt(ds['slant_range']**2-altitude**2).rename('ground_range')
    logging.debug('all finite ground ranges? %s',np.isfinite(ground_ranges.values).all())
    logging.debug('ground_ranges !: %s',ground_ranges)
    range_spacing = kwargs.pop('range_spacing', np.diff(ground_ranges).mean().data)
    logging.debug('range spaceing : %s',range_spacing)
    range = np.arange(ground_ranges.min(), ground_ranges.max(), range_spacing)
    logging.debug('range : %s \n %s all ifinite? %s',range.shape,range,np.isfinite(range).all())
    range = xr.DataArray(range, dims=('range',), coords={'range':range})
    tmptmp = ds.assign_coords(slant_range=ground_ranges).interp(**{'slant_range':range},
                                                              method=method)
    logging.debug('tmptmp: %s',tmptmp)
    #return tmptmp.dropna(dim='range').drop('slant_range')
    return tmptmp.drop('slant_range')

def compute_SAR_cross_spectrum2(slc, *, N_look=3, look_width=0.25, look_overlap=0., look_window=None,
                                range_spacing=None, welsh_window='hanning', nperseg={'range': 512, 'azimuth': 512},
                                noverlap={'range': 256, 'azimuth': 256}, spacing_tol=1e-3, **kwargs):
    """
    Compute SAR cross spectrum using a 2D Welch method. Looks are centered on the mean Doppler frequency
    If ds contains only one cycle, spectrum wavenumbers are added as coordinates in returned DataSet, othrewise, they are passed as variables (k_range, k_azimuth).

    Args:
        slc (xarray): SAR Single Look Complex image. Output of sar_processor()

    Keyword Args:

        N_look (int): Number of looks
        look_width (float): Percent of the total bandwidth used for a single look in [0,1]
        look_overlap (float): Percent of look overlaping [0,1]. Negative values means space between two looks
        look_window (xarray): window used in look processing
        range_spacing (float, optional): range spacing used in slant range to ground range projection. Automattically chosen if left as None
        welsh_window (str, optional): name of the window used in welsh
        nperseg (dict, optional): dict with keys 'range' and 'azimuth'. Values are the number of points used to define a look shape
        noverlap (dict, optional): dict with keys 'range' and 'azimuth'. Values are the number of overlaping points between two looks
        spacing_tol (float, optional): spacing tolerance of range azimuth sampling step.
        kwargs (dict): other arguments passed to ground_regularization()

    Returns:
        (xarray): SAR NRCS spectrum
    """

    if np.abs(look_width) >= 1: raise ValueError('look_width must be in [0,1] range')

    with xr.set_options(keep_attrs=True):
        gslc = ground_regularization(slc, range_spacing=range_spacing, **kwargs)

    gslc = gslc.drop(list(set(gslc.coords).intersection(set(['valid_pulse', 'valid_time', 'padded_tau']))))

    if isinstance(welsh_window, str) or type(welsh_window) is tuple:
        winx = scipy.signal.get_window(welsh_window, nperseg['range'])
        winy = scipy.signal.get_window(welsh_window, nperseg['azimuth'])
        win = np.outer(winx, winy)
    else:
        win = np.asarray(welsh_window)
        if len(win.shape) != 2:
            raise ValueError('window must be 2-D')
        if win.shape[0] > x.shape[-2]:
            raise ValueError('window is longer than x.')
        if win.shape[1] > x.shape[-1]:
            raise ValueError('window is longer than y.')
        nperseg['range'] = win.shape[-2]
        nperseg['azimuth'] = win.shape[-1]

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

    #             plot = True if ((kx==0) and (ky==0)) else False
    #             sub = gslc[{'range':slice(indx, indx+nperseg['range']), 'azimuth':slice(indy, indy+nperseg['azimuth'])}]
    #             sub.data = scipy.signal.detrend(sub, sub.get_axis_num('range'))
    #             sub=sub/np.abs(sub).mean(dim=['range', 'azimuth'])
    #             sub = sub*win
    #             xspecs.append(compute_looks(sub, N_look=N_look, look_width=look_width, look_overlap=look_overlap, look_window=look_window, plot=plot))

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
    allspecs.attrs.update({'tau': slc.synthetic_duration * (look_width - look_overlap)})
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