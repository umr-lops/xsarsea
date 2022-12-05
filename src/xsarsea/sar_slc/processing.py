#!/usr/bin/env python
# coding=utf-8
"""
"""
import numpy as np
import xarray as xr
from tqdm import tqdm
from scipy.constants import c as celerity
from xsarsea.sar_slc.tools import xtiling, xndindex
import resource

def compute_subswath_xspectra(dt):
    """
    Main function to compute inter and intra burst spectra. It has to be modified to be able to change Xspectra options
    """
    import datatree
    # from xsarsea.slc_sar.tools import netcdf_compliant
    from xsarsea import netcdf_compliant

    intra_xs = compute_subswath_intraburst_xspectra(dt)

    intra_xs = intra_xs.drop('spatial_ref')
    intra_xs.attrs.update({'start_date': str(intra_xs.start_date)})
    intra_xs.attrs.update({'stop_date': str(intra_xs.stop_date)})
    intra_xs.attrs.update({'footprint': str(intra_xs.footprint)})
    intra_xs.attrs.update({'multidataset': str(intra_xs.multidataset)})
    intra_xs.attrs.update({'land_mask_computed_by_burst': str(intra_xs.land_mask_computed_by_burst)})
    intra_xs.attrs.pop('pixel_line_m')
    intra_xs.attrs.pop('pixel_sample_m')

    inter_xs = compute_subswath_interburst_xspectra(dt,nperseg={'sample':512, 'line':115})

    inter_xs = inter_xs.drop('spatial_ref')
    inter_xs.attrs.update({'start_date': str(inter_xs.start_date)})
    inter_xs.attrs.update({'stop_date': str(inter_xs.stop_date)})
    inter_xs.attrs.update({'footprint': str(inter_xs.footprint)})
    inter_xs.attrs.update({'multidataset': str(inter_xs.multidataset)})
    inter_xs.attrs.update({'land_mask_computed_by_burst': str(inter_xs.land_mask_computed_by_burst)})
    inter_xs.attrs.pop('pixel_line_m')
    inter_xs.attrs.pop('pixel_sample_m')

    dt = datatree.DataTree.from_dict(
        {'interburst_xspectra': netcdf_compliant(inter_xs), 'intraburst_xspectra': netcdf_compliant(intra_xs)})
    return dt


def compute_subswath_intraburst_xspectra(dt, tile_width={'sample': 20.e3, 'line': 20.e3},
                                         tile_overlap={'sample': 10.e3, 'line': 10.e3}, **kwargs):
    """
    Compute IW subswath intra-burst xspectra per tile
    Note: If requested tile is larger than the size of availabe data. tile will be set to maximum available size
    Args:
        dt (xarray.Datatree): datatree contraining subswath information
        tile_width (dict): approximative sizes of tiles in meters. Dict of shape {dim_name (str): width of tile [m](float)}
        tile_overlap (dict): approximative sizes of tiles overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)}
    
    Keyword Args:
        kwargs (dict): keyword arguments passed to tile_burst_to_xspectra()
        
    Return:
        (xarray): xspectra.
    """
    radar_frequency = float(dt['image'].ds['radarFrequency'])
    xspectra = list()
    pbar = tqdm(range(dt['bursts'].sizes['burst']), desc='start intra burst processing', position=1, leave=False)
    for b in pbar:
        str_mem = 'peak memory usage: %s Mbytes', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.
        pbar.set_description('#### total:%s MeM:%s' % (dt['bursts'].sizes['burst'], str_mem))
        burst = crop_burst(dt['measurement'].ds, dt['bursts'].ds, burst_number=b, valid=True).sel(pol='VV')
        deramped_burst = deramp_burst(burst, dt)
        burst = xr.merge([burst, deramped_burst.drop('azimuthTime')], combine_attrs='drop_conflicts')
        burst.load()
        burst_xspectra = tile_burst_to_xspectra(burst, tile_width, tile_overlap, radar_frequency, **kwargs)
        xspectra.append(burst_xspectra.drop(['tile_line', 'tile_sample']))

    xspectra = xr.concat(xspectra, dim='burst')
    return xspectra

def compute_subswath_interburst_xspectra(dt, tile_width={'sample':20.e3, 'line':20.e3}, tile_overlap={'sample':10.e3, 'line':10.e3}, **kwargs):
    """
    Compute IW subswath inter-burst xspectra. No deramping is applied since only magnitude is used.
    
    Note: If requested tile is larger than the size of availabe data. tile will be set to maximum available size
    Note: The overlap is short in azimuth (line) direction. Keeping nperseg = {'line':None} in Xspectra computation
    keeps maximum number of point in azimuth but is not ensuring the same number of overlapping point for all burst
    
    Args:
        dt (xarray.Datatree): datatree contraining subswath information
        tile_width (dict): approximative sizes of tiles in meters. Dict of shape {dim_name (str): width of tile [m](float)}
        tile_overlap (dict): approximative sizes of tiles overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)}
    
    Keyword Args:
        kwargs (dict): keyword arguments passed to tile_bursts_overlap_to_xspectra()
        
    Return:
        (xarray): xspectra.
    """
    azimuth_steering_rate = dt['image'].ds['azimuthSteeringRate'].item()
    azimuth_time_interval = dt['image'].ds['azimuthTimeInterval'].item()
    xspectra = list()
    #for b in range(dt['bursts'].sizes['burst']-1):
    pbar = tqdm(range(dt['bursts'].sizes['burst']-1), desc='start inter burst processing', position=1, leave=False)
    for b in pbar:
        str_mem = 'peak memory usage: %s Mbytes', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.
        pbar.set_description('#### total:%s MeM:%s' % (dt['bursts'].sizes['burst']-1, str_mem))
        burst0 = crop_burst(dt['measurement'].ds, dt['bursts'].ds, burst_number = b, valid=True, merge_burst_annotation = True).sel(pol='VV')
        burst1 = crop_burst(dt['measurement'].ds, dt['bursts'].ds, burst_number = b+1, valid=True, merge_burst_annotation = True).sel(pol='VV')
        
        interburst_xspectra = tile_bursts_overlap_to_xspectra(burst0, burst1, tile_width, tile_overlap, azimuth_steering_rate, azimuth_time_interval, **kwargs)
        xspectra.append(interburst_xspectra.drop(['tile_line','tile_sample']))
    
    xspectra = xr.concat(xspectra, dim='burst')
    return xspectra



def tile_burst_to_xspectra(burst, tile_width, tile_overlap, radar_frequency,
                           lowpass_width={'sample': 1000., 'line': 1000.}, **kwargs):
    """
    Divide burst in tiles and compute intra-burst cross-spectra using compute_intraburst_xspectrum() function.

    Args:
        burst (xarray.Dataset): dataset with deramped digital number variable
        tile_width (dict): approximative sizes of tiles in meters. Dict of shape {dim_name (str): width of tile [m](float)}
        tile_overlap (dict): approximative sizes of tiles overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)}
        radar_frequency (float): radar frequency [Hz]
        lowpass_width (dict): width for low pass filtering [m]. Dict of form {dim_name (str): width (float)}
    
    Keyword Args:
        kwargs: keyword arguments passed to compute_intraburst_xspectrum()
    """
    #from xsarsea.sar_slc.tools import get_corner_tile, get_middle_tile
    from xsarsea import get_corner_tile, get_middle_tile
    burst.load()


    mean_ground_spacing = float(burst['sampleSpacing']/np.sin(np.radians(burst['incidence'].mean())))
    azimuth_spacing = float(burst['lineSpacing'])  
    spacing = {'sample':mean_ground_spacing, 'line':azimuth_spacing}
    
    nperseg = {d:int(np.rint(tile_width[d]/spacing[d])) for d in tile_width.keys()}

    if tile_overlap in (0., None):
        noverlap = {d: 0 for k in nperseg.keys()}
    else:
        noverlap = {d: int(tile_overlap[d] / spacing[d]) for d in tile_width.keys()}

    tiles_index = xtiling(burst, nperseg=nperseg, noverlap=noverlap)
    tiled_burst = burst[tiles_index].drop(['sample', 'line']).swap_dims({'__' + d: d for d in tile_width.keys()})
    tiles_sizes = {d: k for d, k in tiled_burst.sizes.items() if 'tile_' in d}

    xs = np.empty(tuple(tiles_sizes.values()), dtype=object)
    taus = xr.DataArray(np.empty(tuple(tiles_sizes.values()), dtype='float'), dims=tiles_sizes.keys(), name='tau')
    cutoff = xr.DataArray(np.empty(tuple(tiles_sizes.values()), dtype='float'), dims=tiles_sizes.keys(), name='cutoff')

    for i in xndindex(tiles_sizes):
        # sub = tiled_burst[i].swap_dims({'n_line':'line','n_sample':'sample'})
        sub = tiled_burst[i]
        mean_incidence = float(sub.incidence.mean())
        mean_slant_range = float(sub.slant_range_time.mean()) * celerity / 2.
        mean_velocity = float(sub.velocity.mean())
        slant_spacing = float(sub['sampleSpacing'])
        ground_spacing = slant_spacing / np.sin(np.radians(mean_incidence))
        azimuth_spacing = float(sub['lineSpacing'])
        synthetic_duration = celerity * mean_slant_range / (2 * radar_frequency * mean_velocity * azimuth_spacing)
        mod = compute_modulation(sub['deramped_digital_number'], lowpass_width=lowpass_width,
                                 spacing={'sample': ground_spacing, 'line': azimuth_spacing})
        xspecs = compute_intraburst_xspectrum(mod, mean_incidence, slant_spacing, azimuth_spacing, synthetic_duration,
                                              **kwargs)
        xspecs_m = xspecs.mean(dim=['periodo_line', 'periodo_sample'],
                               keep_attrs=True)  # averaging all the periodograms in each tile
        xs[tuple(i.values())] = xspecs_m
        # ------------- tau ----------------
        taus[i] = float(xspecs.attrs['tau'])
        # ------------- cut-off ------------
        cutoff_tau = [str(i) + 'tau' for i in [1, 2, 3, 0] if str(i) + 'tau' in xspecs_m.dims][
            0]  # which tau is used to compute azimuthal cutoff
        k_rg = xspecs_m.k_rg
        k_az = xspecs_m.k_az
        xspecs_m = xspecs_m['xspectra_' + cutoff_tau].mean(dim=cutoff_tau)
        xspecs_m = xspecs_m.assign_coords({'k_rg': k_rg, 'k_az': k_az}).swap_dims(
            {'freq_sample': 'k_rg', 'freq_line': 'k_az'})
        cutoff[i] = compute_azimuth_cutoff(xspecs_m)

    xs = [list(a) for a in list(xs)]  # must be generalized for larger number of dimensions
    xs = xr.combine_nested(xs, concat_dim=tiles_sizes.keys(), combine_attrs='drop_conflicts')
    # xs = xs.assign_coords(tiles.coords)
    # tau
    taus.attrs.update({'long_name': 'delay between two successive looks', 'units': 's'})
    cutoff.attrs.update({'long_name': 'Azimuthal cut-off', 'units': 'm'})

    tiles_corners = get_corner_tile(tiles_index)
    corner_lon = burst['longitude'][tiles_corners].rename('corner_longitude').drop(['line', 'sample'])
    corner_lat = burst['latitude'][tiles_corners].rename('corner_latitude').drop(['line', 'sample'])

    tiles_middle = get_middle_tile(tiles_index)
    middle_lon = burst['longitude'][tiles_middle].rename('longitude')
    middle_lat = burst['latitude'][tiles_middle].rename('latitude')

    xs = xr.merge([xs, taus.to_dataset(), cutoff.to_dataset(), corner_lon.to_dataset(), corner_lat.to_dataset()],
                  combine_attrs='drop_conflicts')
    xs = xs.assign_coords({'longitude': middle_lon,
                           'latitude': middle_lat})  # This line also ensures adding line/sample coordinates too !! DO NOT REMOVE
    xs.attrs.update(burst.attrs)
    xs.attrs.update({'tile_nperseg_' + d: k for d, k in nperseg.items()})
    xs.attrs.update({'tile_noverlap_' + d: k for d, k in noverlap.items()})
    return xs


def burst_valid_indexes(ds):
    """
    Find indexes of valid portion of a burst. Returned line index are relative to burst only !
    
    Args:
        ds (xarray.Dataset): Dataset of one burst
    Return:
        (tuple of int): index of (first valid sample, last valid sample, first valid line, last valid line)
    """
    fvs = ds['firstValidSample']  # first valid samples
    valid_lines = np.argwhere(np.isfinite(fvs).data)  # valid lines
    fvl = int(valid_lines[0])  # first valid line
    lvl = int(valid_lines[-1])  # last valid line
    fvs = int(fvs.max(dim='line'))

    lvs = ds['lastValidSample']  # last valid samples
    valid_lines2 = np.argwhere(np.isfinite(lvs).data)  # valid lines
    if not np.all(valid_lines2 == valid_lines):
        raise ValueError("valid lines are not consistent between first and last valid samples")
    lvs = int(lvs.max(dim='line'))
    return fvs, lvs, fvl, lvl


def crop_burst(ds, burst_annotation, burst_number, valid=True, merge_burst_annotation=True):
    """
    Crop burst from the measurement dataset
    
    Args:
        ds (xarray.Dataset): measurement dataset
        burst_annotation (xarray.dataset): burst annotation dataset
        burst_number (int): burst number
        valid (bool, optional): If true: only return the valid part of the burst
        merge_burst_annotation (bool): If true: annotation of the burst are added to the returned dataset
        
    Return:
        xarray.Dataset : extraction of valid burst portion of provided datatree
    """

    lpb = int(burst_annotation['linesPerBurst'])

    if valid:
        fs, ls, fl, ll = burst_valid_indexes(
            burst_annotation.sel(burst=burst_number))  # first and last line are relative to burst
    else:
        fs, ls = None, None
        fl = 0  # relative to burst
        ll = lpb  # relative to burst

    myburst = ds[{'sample': slice(fs, ls, None), 'line': slice(burst_number * lpb + fl, burst_number * lpb + ll, None)}]

    if merge_burst_annotation:
        annotation = burst_annotation.sel(burst=burst_number)[{'line': slice(fl, ll, None)}]
        myburst = xr.merge([myburst, annotation])

    return myburst.assign_coords({'burst': burst_number})  # This ensures keeping burst number in coordinates


def deramp_burst(burst, dt):
    """
    Deramp burst. Return deramped digital numbers
    
    Args:
        burst (xarray.dataArray or xarray.Dataset): burst or portion of a burst
        dt (xarray.dataTree): datatree containing all informations of the SLC
    Return:
        (xarray.DataArray): deramped digital numbers
    """
    # from sar_slc.deramping import compute_midburst_azimuthtime, compute_slant_range_time, compute_Doppler_centroid_rate, \
    #     compute_reference_time, compute_deramping_phase, compute_DopplerCentroid_frequency
    from xsarsea import compute_midburst_azimuthtime, compute_slant_range_time, compute_Doppler_centroid_rate, \
        compute_reference_time, compute_deramping_phase, compute_DopplerCentroid_frequency

    FMrate = dt['FMrate'].ds
    dcEstimates = dt['doppler_estimate'].ds
    orbit = dt['orbit'].ds
    # radar_frequency = float(dt.ds.radarFrequency)
    radar_frequency = float(dt['image'].ds['radarFrequency'])
    azimuth_steering_rate = float(dt['image'].ds['azimuthSteeringRate'])
    azimuth_time_interval = float(dt['image'].ds['azimuthTimeInterval'])

    midburst_azimuth_time = compute_midburst_azimuthtime(burst, azimuth_time_interval)  # mid burst azimuth time
    slant_range_time = compute_slant_range_time(burst, dt['image'].ds['slantRangeTime'],
                                                dt['image'].ds['rangeSamplingRate'])

    kt = compute_Doppler_centroid_rate(orbit, azimuth_steering_rate, radar_frequency, FMrate, midburst_azimuth_time,
                                       slant_range_time)
    fnc = compute_DopplerCentroid_frequency(dcEstimates, midburst_azimuth_time, slant_range_time)
    eta_ref = compute_reference_time(FMrate, dcEstimates, midburst_azimuth_time, slant_range_time,
                                     int(burst['samplesPerBurst']))
    phi = compute_deramping_phase(burst, kt, eta_ref, azimuth_time_interval)

    with xr.set_options(keep_attrs=True):
        deramped_signal = (burst['digital_number'] * np.exp(-1j * phi)).rename('deramped_digital_number')

    return deramped_signal


def compute_modulation(ds, lowpass_width, spacing):
    """
    Compute modulation map (sig0/low_pass_filtered_sig0)

    Args:
        ds (xarray) : array of (deramped) digital number
        lowpass_width (dict): form {name of dimension (str): width in [m] (float)}. width for low pass filtering [m]
        spacing (dict): form {name of dimension (str): spacing in [m] (float)}. spacing for each filtered dimension


    """
    from scipy.signal import fftconvolve
    # from xsarsea.sar_slc.tools import gaussian_kernel
    from xsarsea import gaussian_kernel
    # ground_spacing = float(ds['sampleSpacing'])/np.sin(np.radians(ds['incidence'].mean()))

    mask = np.isfinite(ds)
    gk = gaussian_kernel(width=lowpass_width, spacing=spacing)
    swap_dims = {d: d + '_' for d in lowpass_width.keys()}
    gk = gk.rename(swap_dims)

    low_pass_intensity = xr.apply_ufunc(fftconvolve, np.abs(ds.where(mask, 0.)) ** 2, gk,
                                        input_core_dims=[lowpass_width.keys(), swap_dims.values()], vectorize=True,
                                        output_core_dims=[lowpass_width.keys()], kwargs={'mode': 'same'})

    normal = xr.apply_ufunc(fftconvolve, mask, gk, input_core_dims=[lowpass_width.keys(), swap_dims.values()],
                            vectorize=True, output_core_dims=[lowpass_width.keys()], kwargs={'mode': 'same'})

    low_pass_intensity = low_pass_intensity / normal

    return ds / np.sqrt(low_pass_intensity)


def compute_intraburst_xspectrum(slc, mean_incidence, slant_spacing, azimuth_spacing, synthetic_duration,
                                 azimuth_dim='line', nperseg={'sample': 512, 'line': 512},
                                 noverlap={'sample': 256, 'line': 256}, **kwargs):
    """
    Compute SAR cross spectrum using a 2D Welch method. Looks are centered on the mean Doppler frequency
    If ds contains only one cycle, spectrum wavenumbers are added as coordinates in returned DataSet, otherwise, they are passed as variables (k_range, k_azimuth).
    
    Args:
        slc (xarray): digital numbers of Single Look Complex image.
        mean_incidence (float): mean incidence on slc
        slant_spacing (float): slant spacing
        azimuth_spacing (float): azimuth spacing
        synthetic_duration (float): synthetic aperture duration (to compute tau)
        azimuth_dim (str): name of azimuth dimension
        nperseg (dict of int): number of point per periodogram. Dict of form {dimension_name(str): number of point (int)}
        noverlap (dict of int): number of overlapping point per periodogram. Dict of form {dimension_name(str): number of point (int)}
    
    Keyword Args:
        kwargs (dict): keyword arguments passed to compute_looks()
        
    Returns:
        (xarray): SLC cross_spectra
    """

    range_dim = list(set(slc.dims) - set([azimuth_dim]))[0]  # name of range dimension

    periodo_slices = xtiling(slc, nperseg=nperseg, noverlap=noverlap, prefix='periodo_')
    periodo = slc[periodo_slices].swap_dims({'__' + d: d for d in [range_dim, azimuth_dim]})
    periodo_sizes = {d: k for d, k in periodo.sizes.items() if 'periodo_' in d}

    out = np.empty(tuple(periodo_sizes.values()), dtype=object)

    for i in xndindex(periodo_sizes):
        image = periodo[i]
        xspecs = compute_looks(image, azimuth_dim=azimuth_dim, synthetic_duration=synthetic_duration,
                               **kwargs)  # .assign_coords(i)
        out[tuple(i.values())] = xspecs

    out = [list(a) for a in list(out)]  # must be generalized for larger number of dimensions
    out = xr.combine_nested(out, concat_dim=periodo_sizes.keys(), combine_attrs='drop_conflicts')
    # out = out.assign_coords(periodo_slices.coords)
    out = out.assign_coords(periodo.coords)

    out.attrs.update({'mean_incidence': mean_incidence})

    # dealing with wavenumbers
    ground_range_spacing = slant_spacing / np.sin(np.radians(out.mean_incidence))
    k_rg = xr.DataArray(
        np.fft.fftshift(np.fft.fftfreq(out.sizes['freq_' + range_dim], ground_range_spacing / (2 * np.pi))),
        dims='freq_' + range_dim, name='k_rg', attrs={'long_name': 'wavenumber in range direction', 'units': 'rad/m'})
    k_az = xr.DataArray(np.fft.fftshift(
        np.fft.fftfreq(out.sizes['freq_' + azimuth_dim], azimuth_spacing / (out.attrs.pop('look_width') * 2 * np.pi))),
                        dims='freq_' + azimuth_dim, name='k_az',
                        attrs={'long_name': 'wavenumber in azimuth direction', 'units': 'rad/m'})
    # out = out.assign_coords({'k_rg':k_rg, 'k_az':k_az}).swap_dims({'freq_'+range_dim:'k_rg', 'freq_'+azimuth_dim:'k_az'})
    out = xr.merge([out, k_rg.to_dataset(), k_az.to_dataset()],
                   combine_attrs='drop_conflicts')  # Adding .to_dataset() ensures promote_attrs=False
    out.attrs.update({'periodogram_nperseg_' + range_dim: nperseg[range_dim],
                      'periodogram_nperseg_' + azimuth_dim: nperseg[azimuth_dim],
                      'periodogram_noverlap_' + range_dim: noverlap[range_dim],
                      'periodogram_noverlap_' + azimuth_dim: noverlap[azimuth_dim]})

    return out.drop(['freq_' + range_dim, 'freq_' + azimuth_dim])


def compute_looks(slc, azimuth_dim, synthetic_duration, nlooks=3, look_width=0.2, look_overlap=0., look_window=None,
                  plot=False):
    """
    Compute the N looks of an slc DataArray.
    Spatial coverage of the provided slc must be small enough to enure an almost constant ground spacing.
    Meaning: ground_spacing ~= slant_spacing / sin(mean_incidence)
    
    Args:
        slc (xarray.DataArray): (bi-dimensional) array to process
        azimuth_dim (str) : name of the azimuth dimension (dimension used to extract the look)
        nlooks (int): number of look
        look_width (float): in [0,1.] width of a look [percentage of full bandwidth] (nlooks*look_width must be < 1)
        look_overlap (float): in [0,1.] look overlapping [percentage of a look]
        look_window (str or tuple): instance that can be passed to scipy.signal.get_window()
    
    Return:
        (dict) : keys are '0tau', '1tau', ... and values are list of corresponding computed spectra
    """
    # import matplotlib.pyplot as plt
    import xrft

    if nlooks < 1:
        raise ValueError('Number of look must be greater than 0')
    if (nlooks * look_width) > 1.:
        raise ValueError('Number of look times look_width must be lower than 1.')

    range_dim = list(set(slc.dims) - set([azimuth_dim]))[0]  # name of range dimension
    freq_azi_dim = 'freq_' + azimuth_dim
    freq_rg_dim = 'freq_' + range_dim

    Np = slc.sizes[azimuth_dim]  # total number of point in azimuth direction
    nperlook = int(look_width * Np)  # number of point perlook in azimuth direction
    noverlap = int(np.rint(look_overlap * look_width * Np))  # number of overlap point

    mydop = xrft.fft(slc, dim=[azimuth_dim], detrend=None, window=None, shift=True, true_phase=True,
                     true_amplitude=True)

    # Finding and removing Doppler centroid
    weight = xr.DataArray(np.hanning(100), dims=['window'])  # window for smoothing
    weight /= weight.sum()
    smooth_dop = np.abs(mydop).mean(dim=range_dim).rolling(**{freq_azi_dim: len(weight), 'center': True}).construct(
        'window').dot(weight)
    i0 = int(np.abs(mydop[freq_azi_dim]).argmin())  # zero frequency indice
    ishift = int(smooth_dop.argmax()) - i0  # shift of Doppler centroid
    mydop = mydop.roll(**{freq_azi_dim: -ishift, 'roll_coords': False})

    # Extracting the useful part of azimuthal Doppler spectrum
    # It removes points on each side to be sure that tiling will operate correctly
    Nused = nlooks * nperlook - (nlooks - 1) * noverlap
    left = (Np - Nused) // 2  # useless points on left side
    mydop = mydop[{freq_azi_dim: slice(left, left + Nused)}]
    look_tiles = xtiling(mydop, nperseg={freq_azi_dim: nperlook}, noverlap={freq_azi_dim: noverlap}, prefix='look_')

    if look_window is not None:
        raise ValueError('Look windowing is not available.')

    looks_spec = list()
    looks = mydop[look_tiles].drop(['freq_' + azimuth_dim]).swap_dims({'__' + d: d for d in ['freq_' + azimuth_dim]})
    looks_sizes = {d: k for d, k in looks.sizes.items() if 'look_' in d}

    # for l in range(look_tiles.sizes[freq_azi_dim]):
    for l in xndindex(looks_sizes):
        look = looks[l]
        look = xrft.ifft(look.assign_coords({freq_azi_dim: np.arange(-(nperlook // 2),
                                                                     -(nperlook // 2) + nperlook) * float(
            mydop[freq_azi_dim].spacing)}), dim=freq_azi_dim, detrend=None, window=False, shift=True, true_phase=False,
                         true_amplitude=True)
        look = np.abs(look) ** 2
        look = look / look.mean(dim=slc.dims)
        looks_spec.append(xrft.fft(look, dim=slc.dims, detrend='constant', true_phase=True, true_amplitude=True))
        # looks_spec.append(xrft.fft(np.abs(look)**2, dim=slc.dims, detrend='linear'))

    looks_spec = xr.concat(looks_spec, dim='look')

    xspecs = {str(i) + 'tau': [] for i in range(nlooks)}  # using .fromkeys() do not work because of common empylist
    for l1 in range(nlooks):
        for l2 in range(l1, nlooks):
            df = float(looks_spec[{'look': l2}][freq_azi_dim].spacing * looks_spec[{'look': l2}][freq_rg_dim].spacing)
            xspecs[str(l2 - l1) + 'tau'].append(looks_spec[{'look': l2}] * np.conj(looks_spec[{'look': l1}]) * df)

    # compute tau = time difference between looks
    look_sep = look_width * (1. - look_overlap)
    tau = synthetic_duration * look_sep

    merged_xspecs = list()
    for i in range(nlooks):
        concat_spec = xr.concat(xspecs[str(i) + 'tau'], dim=str(i) + 'tau').rename('xspectra_{}tau'.format(i))
        concat_spec.attrs.update(
            {'nlooks': nlooks, 'look_width': look_width, 'look_overlap': look_overlap, 'look_window': str(look_window),
             'tau': tau})
        merged_xspecs.append(concat_spec.to_dataset())  # adding to_dataset() ensures promote_attrs=False per default

    merged_xspecs = xr.merge(merged_xspecs, combine_attrs='drop_conflicts')
    merged_xspecs.attrs.update({'look_width': look_width, 'tau': tau})
    return merged_xspecs


def compute_azimuth_cutoff(spectrum, definition='drfab'):
    """
    compute azimuth cutoff
    Args:
        spectrum (xarray): Xspectrum with coordinates k_rg and k_az
        definition (str, optional): ipf (covariance is averaged over range) or drfab (covariance taken at range = 0)
    Return:
        (float): azimuth cutoff [m]
    """
    import xrft
    from scipy.optimize import curve_fit
    coV = xrft.ifft(spectrum, dim=('k_rg', 'k_az'), shift=True, prefix='k_')
    coV = coV.assign_coords({'rg': 2 * np.pi * coV.rg, 'az': 2 * np.pi * coV.az})
    if definition == 'ipf':
        coVRm = coV.real.mean(dim='rg')
    elif definition == 'drfab':
        coVRm = np.real(coV).sel(rg=0.0)
    else:
        raise ValueError("Unknow definition '{}' for azimuth cutoff. It must be 'drfab' or 'ipf'".format(definition))
    coVRm /= coVRm.max()
    coVfit = coVRm.where(np.abs(coVRm.az) < 500, drop=True)

    def fit_gauss(x, a, l):
        return a * np.exp(-(np.pi * x / l) ** 2)

    p, r = curve_fit(fit_gauss, coVfit.az, coVfit.data, p0=[1., 227.])
    return p[1]


def tile_bursts_overlap_to_xspectra(burst0, burst1, tile_width, tile_overlap, azimuth_steering_rate,
                                    azimuth_time_interval, lowpass_width={'sample': 1000., 'line': 1000.}, **kwargs):
    """
    Divide bursts overlaps in tiles and compute inter-burst cross-spectra using compute_interburst_xspectrum() function.

    Args:
        burst0 (xarray.Dataset): first burst (in time) dataset (No need of deramped digital number variable)
        burst1 (xarray.Dataset): second burst (in time) dataset (No need of deramped digital number variable)
        tile_width (dict): approximative sizes of tiles in meters. Dict of shape {dim_name (str): width of tile [m](float)}
        tile_overlap (dict): approximative sizes of tiles overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)}
        azimuth_steering_rate (float) : antenna azimuth steering rate [deg/s]
        azimuth_time_interval (float) : azimuth time spacing [s]
        lowpass_width (dict): width for low pass filtering [m]. Dict of form {dim_name (str): width (float)}
    
    Keyword Args:
        kwargs: keyword arguments passed to compute_interburst_xspectrum()
    """
#    from xsarsea.sar_slc.tools import get_corner_tile, get_middle_tile
    from xsarsea import get_corner_tile, get_middle_tile

    # find overlapping burst portion
    az0 = burst0[{'sample': 0}].azimuth_time.load()
    az1 = burst1.isel(sample=0).azimuth_time[{'line': 0}].load()

    frl = np.argwhere(az0.data >= az1.data)[0].item()  # first overlapping line of first burst
    # Lines below ensures we choose the closest index since azimuth_time are not exactly the same
    t0 = burst0[{'sample': 0, 'line': frl}].azimuth_time
    t1 = burst1[{'sample': 0, 'line': 0}].azimuth_time
    aziTimeDiff = np.abs(t0 - t1)

    if np.abs(burst0[{'sample': 0, 'line': frl - 1}].azimuth_time - t1) < aziTimeDiff:
        frl -= 1
    elif np.abs(burst0[{'sample': 0, 'line': frl + 1}].azimuth_time - t1) < aziTimeDiff:
        frl += 1
    else:
        pass

    burst0 = burst0[{'line': slice(frl, None)}]
    burst1 = burst1[{'line': slice(0, burst0.sizes['line'])}]

    # if overlap0.sizes!=overlap1.sizes:
    #     raise ValueError('Overlaps have different sizes: {} and {}'.format(overlap0.sizes, overlap1.sizes))

    burst0.load()  # loading ensures efficient tiling below
    burst1.load()  # loading ensures efficient tiling below

    burst = burst0
    mean_ground_spacing = float(burst['sampleSpacing']) / np.sin(np.radians(burst['incidence'].mean()))
    azimuth_spacing = float(burst['lineSpacing'])
    spacing = {'sample': mean_ground_spacing, 'line': azimuth_spacing}
    nperseg = {d: int(tile_width[d] / spacing[d]) for d in tile_width.keys()}
    if tile_overlap in (0., None):
        noverlap = {d: 0 for k in nperseg.keys()}
    else:
        noverlap = {d: int(tile_overlap[d] / spacing[d]) for d in tile_width.keys()}
    tiles_index = xtiling(burst, nperseg=nperseg, noverlap=noverlap)
    tiled_burst0 = burst0[tiles_index]  # .drop(['sample','line']).swap_dims({'__'+d:d for d in tile_width.keys()})
    tiled_burst1 = burst1[tiles_index]  # .drop(['sample','line']).swap_dims({'__'+d:d for d in tile_width.keys()})
    tiles_sizes = {d: k for d, k in tiled_burst0.sizes.items() if 'tile_' in d}

    xs = np.empty(tuple(tiles_sizes.values()), dtype=object)
    taus = xr.DataArray(np.empty(tuple(tiles_sizes.values()), dtype='float'), dims=tiles_sizes.keys(), name='tau')
    cutoff = xr.DataArray(np.empty(tuple(tiles_sizes.values()), dtype='float'), dims=tiles_sizes.keys(), name='cutoff')

    for i in xndindex(tiles_sizes):
        sub0 = tiled_burst0[i].swap_dims({'__' + d: d for d in tile_width.keys()})
        sub1 = tiled_burst1[i].swap_dims({'__' + d: d for d in tile_width.keys()})
        sub = sub0
        mean_incidence = float(sub.incidence.mean())
        mean_slant_range = float(sub.slant_range_time.mean()) * celerity / 2.
        slant_spacing = float(sub['sampleSpacing'])
        ground_spacing = slant_spacing / np.sin(np.radians(mean_incidence))
        azimuth_spacing = float(sub['lineSpacing'])

        mod0 = compute_modulation(np.abs(sub0['digital_number']), lowpass_width=lowpass_width,
                                  spacing={'sample': ground_spacing, 'line': azimuth_spacing})
        mod1 = compute_modulation(np.abs(sub1['digital_number']), lowpass_width=lowpass_width,
                                  spacing={'sample': ground_spacing, 'line': azimuth_spacing})

        xspecs = compute_interburst_xspectrum(mod0 ** 2, mod1 ** 2, mean_incidence, slant_spacing, azimuth_spacing,
                                              **kwargs)
        xspecs_m = xspecs.mean(dim=['periodo_line', 'periodo_sample'],
                               keep_attrs=True)  # averaging all the periodograms in each tile
        xs[tuple(i.values())] = xspecs_m
        # ------------- tau ------------------
        antenna_velocity = np.radians(azimuth_steering_rate) * mean_slant_range
        ground_velocity = azimuth_spacing / azimuth_time_interval
        scan_velocity = (ground_velocity + antenna_velocity).item()
        dist0 = (sub0[{'line': sub0.sizes['line'] // 2}]['line'] - sub0['linesPerBurst'] * sub0[
            'burst']) * azimuth_spacing  # distance from begining of the burst
        dist1 = (sub1[{'line': sub1.sizes['line'] // 2}]['line'] - sub1['linesPerBurst'] * sub1[
            'burst']) * azimuth_spacing  # distance from begining of the burst
        tau = (sub1['sensingTime'] - sub0['sensingTime']) / np.timedelta64(1, 's') + (
                    dist1 - dist0) / scan_velocity  # The division by timedelta64(1,s) is to convert in seconds
        taus[i] = tau.item()
        # ------------- cut-off --------------
        k_rg = xspecs_m.k_rg
        k_az = xspecs_m.k_az
        xspecs_m = xspecs_m['interburst_xspectra']
        xspecs_m = xspecs_m.assign_coords({'k_rg': k_rg, 'k_az': k_az}).swap_dims(
            {'freq_sample': 'k_rg', 'freq_line': 'k_az'})
        cutoff[i] = compute_azimuth_cutoff(xspecs_m)

    xs = [list(a) for a in list(xs)]  # must be generalized for larger number of dimensions
    xs = xr.combine_nested(xs, concat_dim=tiles_sizes.keys(), combine_attrs='drop_conflicts')
    # xs = xs.assign_coords(tiles.coords)
    # tau
    taus.attrs.update({'long_name': 'delay between two successive acquisitions', 'units': 's'})
    cutoff.attrs.update({'long_name': 'Azimuthal cut-off', 'units': 'm'})

    tiles_corners = get_corner_tile(tiles_index)
    corner_lon = burst['longitude'][tiles_corners].rename('corner_longitude').drop(['line', 'sample'])
    corner_lat = burst['latitude'][tiles_corners].rename('corner_latitude').drop(['line', 'sample'])

    tiles_middle = get_middle_tile(tiles_index)
    middle_lon = burst['longitude'][tiles_middle].rename('longitude')
    middle_lat = burst['latitude'][tiles_middle].rename('latitude')

    xs = xr.merge([xs, taus.to_dataset(), cutoff.to_dataset(), corner_lon.to_dataset(), corner_lat.to_dataset()],
                  combine_attrs='drop_conflicts')
    xs = xs.assign_coords({'longitude': middle_lon,
                           'latitude': middle_lat})  # This line also ensures adding line/sample coordinates too !! DO NOT REMOVE
    xs.attrs.update(burst.attrs)
    xs.attrs.update({'tile_nperseg_' + d: k for d, k in nperseg.items()})
    xs.attrs.update({'tile_noverlap_' + d: k for d, k in noverlap.items()})
    return xs


def compute_interburst_xspectrum(mod0, mod1, mean_incidence, slant_spacing, azimuth_spacing, azimuth_dim='line',
                                 nperseg={'sample': 512, 'line': None}, noverlap={'sample': 256, 'line': 0}):
    """
    Compute cross spectrum between mod0 and mod1 using a 2D Welch method (periodograms).
    
    Args:
        mod0 (xarray): modulation signal from burst0
        mod1 (xarray): modulation signal from burst1
        mean_incidence (float): mean incidence on slc
        slant_spacing (float): slant spacing
        azimuth_spacing (float): azimuth spacing
        azimuth_dim (str): name of azimuth dimension
        nperseg (dict of int): number of point per periodogram. Dict of form {dimension_name(str): number of point (int)}
        noverlap (dict of int): number of overlapping point per periodogram. Dict of form {dimension_name(str): number of point (int)}
        
    Returns:
        (xarray): concatenated cross_spectra
    """

    range_dim = list(set(mod0.dims) - set([azimuth_dim]))[0]  # name of range dimension

    periodo_slices = xtiling(mod0, nperseg=nperseg, noverlap=noverlap, prefix='periodo_')

    periodo0 = mod0[periodo_slices]  # .swap_dims({'__'+d:d for d in [range_dim, azimuth_dim]})
    periodo1 = mod1[periodo_slices]  # .swap_dims({'__'+d:d for d in [range_dim, azimuth_dim]})
    periodo_sizes = {d: k for d, k in periodo0.sizes.items() if 'periodo_' in d}

    out = np.empty(tuple(periodo_sizes.values()), dtype=object)

    for i in xndindex(periodo_sizes):
        image0 = periodo0[i].swap_dims({'__' + d: d for d in [range_dim, azimuth_dim]})
        image1 = periodo1[i].swap_dims({'__' + d: d for d in [range_dim, azimuth_dim]})
        image0 = (image0 - image0.mean()) / image0.mean()
        image1 = (image1 - image1.mean()) / image1.mean()
        xspecs = xr.DataArray(np.fft.fftshift(np.fft.fft2(image1) * np.conj(np.fft.fft2(image0))),
                              dims=['freq_' + d for d in image0.dims])
        out[tuple(i.values())] = xspecs

    out = [list(a) for a in list(out)]  # must be generalized for larger number of dimensions
    out = xr.combine_nested(out, concat_dim=periodo_sizes.keys(), combine_attrs='drop_conflicts').rename(
        'interburst_xspectra')

    out = out.assign_coords(periodo0.drop(['line', 'sample']).coords)

    out.attrs.update({'mean_incidence': mean_incidence})

    # dealing with wavenumbers
    ground_range_spacing = slant_spacing / np.sin(np.radians(out.mean_incidence))
    k_rg = xr.DataArray(
        np.fft.fftshift(np.fft.fftfreq(out.sizes['freq_' + range_dim], ground_range_spacing / (2 * np.pi))),
        dims='freq_' + range_dim, name='k_rg', attrs={'long_name': 'wavenumber in range direction', 'units': 'rad/m'})
    k_az = xr.DataArray(
        np.fft.fftshift(np.fft.fftfreq(out.sizes['freq_' + azimuth_dim], azimuth_spacing / (2 * np.pi))),
        dims='freq_' + azimuth_dim, name='k_az',
        attrs={'long_name': 'wavenumber in azimuth direction', 'units': 'rad/m'})
    # out = out.assign_coords({'k_rg':k_rg, 'k_az':k_az}).swap_dims({'freq_'+range_dim:'k_rg', 'freq_'+azimuth_dim:'k_az'})
    out = xr.merge([out, k_rg.to_dataset(), k_az.to_dataset()],
                   combine_attrs='drop_conflicts')  # Adding .to_dataset() ensures promote_attrs=False
    out.attrs.update({'periodogram_nperseg_' + range_dim: nperseg[range_dim],
                      'periodogram_nperseg_' + azimuth_dim: nperseg[azimuth_dim],
                      'periodogram_noverlap_' + range_dim: noverlap[range_dim],
                      'periodogram_noverlap_' + azimuth_dim: noverlap[azimuth_dim]})
    return out
