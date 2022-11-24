#!/usr/bin/env python
# coding=utf-8
"""
"""
import numpy as np
import xarray as xr

def compute_slant_range_time(burst, slant_range_time0, range_sampling_rate):
    """
    Compute the slant range time vector
    
    Args:
        burst (xarray.DataArray): burst measurement subportion with dimension sample
        slant_range_time0 (float) : slant range time of the first sample
        range_sampling_rate (float) : range sampling rate
    """
    slant_range_time = slant_range_time0+burst['sample']/range_sampling_rate
    slant_range_time.attrs.update({'units':'s', 'long_name':'slant_range_time'})
    return slant_range_time.rename('slant_range_time')

def compute_midburst_azimuthtime(ds, azimuth_time_interval):
    """
    Compute mid-burst azimuth time
    
    Args:
        ds (xarray.DataArray or xarray.Dataset): Dataset of bursts annotation or cropped burst with stored annotations
        azimuth_time_interval (float): azimuth time interval [second]
    Return:
        (xarray.dataArray): 
    """
    lpb = ds['linesPerBurst']
    midburstati = ds['sensingTime']+lpb/2*np.timedelta64(int(azimuth_time_interval*1e12),'ps').astype('<m8[ns]')
    return midburstati.rename('midBurst_azimuth_time')

def compute_Doppler_rate(orbit, azimuth_steering_rate, radar_frequency):
    """
    Compute the Doppler rate (ks)
    
    Args:
        orbit (xarray.Dataset or xarray.DataArray): orbital dataset
        azimuth_steering_rate (float): azimuth steering rate in [deg/s]
        frequency (float): radar frequency [Hz]
    Return:
        (xarray.DataArray): Doppler rate (ks)
    """
    from scipy.constants import c as celerity
    vs = np.sqrt(orbit['velocity_x']**2+orbit['velocity_y']**2+orbit['velocity_z']**2).rename('orbital_velocity') # orbital velocity
    kphi = np.radians(azimuth_steering_rate) # azimuth steering rate in [rad/s]
    ks = 2.*vs/celerity*radar_frequency*kphi # Doppler rate
    ks.attrs.update({'units':'Hz/s', 'long_name':'Doppler_rate'})
    return ks.rename('ks')

def compute_FMrate(FMrate, midburst_azimuth_time, slant_range_time):
    """
    Compute Doppler FM rate (ka) (slant range dependent) at the closest midburst azimuth time
    
    Args:
        FMrate (xarray.DataArray) : dataArray of FMrate polynomials
        midburst_azimuth_time (xarray.DataArray): array of mid-azimuth time
        slant_range_time (xarray.DataArray) : array of slant range time
    """
    iazi = (np.abs(midburst_azimuth_time-FMrate['azimuthTime'])).argmin(dim='azimuthTime') # index of nearest midburst azimuth time
    poly = FMrate['azimuthFmRatePolynomial'][{'azimuthTime':iazi}]
    t0 = FMrate['t0'][{'azimuthTime':iazi}]
    
    def eval_poly(polynome, slant_range_time, t0):
        return polynome(slant_range_time - t0)
    
    ka = xr.apply_ufunc(eval_poly, poly, slant_range_time, t0, input_core_dims=[[],['sample'],[]], vectorize=True, output_core_dims=[['sample']])
    ka.attrs.update({'units':'Hz/s','long_name':'Doppler_FMrate'})
    return ka

def compute_Doppler_centroid_rate(orbit, azimuth_steering_rate, radar_frequency, FMrate, midburst_azimuth_time, slant_range_time):
    """
    Compute the (range dependent) Doppler centroid rate (kt)
    
    Args:
        orbit (xarray.Dataset or xarray.DataArray): orbital dataset
        azimuth_steering_rate (float): azimuth steering rate in [deg/s]
        frequency (float): radar frequency [Hz]
        FMrate (xarray.DataArray) : dataArray of FMrate polynomials
        midburst_azimuth_time (xarray.DataArray): array of mid-azimuth time
        slant_range_time (xarray.DataArray) : array of slant range time
    Return:
        (xarray.DataArray) : array of Doppler centroid rate
    """
    
    interpolated_orbit = orbit.interp(time=midburst_azimuth_time).drop('time')
    ks = compute_Doppler_rate(interpolated_orbit, azimuth_steering_rate, radar_frequency)
    
    ka = compute_FMrate(FMrate, midburst_azimuth_time, slant_range_time)
    
    kt = ka*ks/(ka-ks)
    kt.attrs.update({'units':'Hz/s','long_name':'Doppler_centroid_rate'})
    return kt.rename('kt')

def compute_DopplerCentroid_frequency(DopplerEstimate, midburst_azimuth_time, slant_range_time):
    """
    Compute Doppler centroid frequency (fnc) (slant range dependent) at the closest midburst azimuth time
    
    Args:
        DopplerEstimate (xarray.DataArray) : DataArray of Doppler estimate polynomials
        midburst_azimuth_time (xarray.DataArray): array of mid-azimuth time
        slant_range_time (xarray.DataArray) : array of slant range time
    """
    iazi = (np.abs(midburst_azimuth_time-DopplerEstimate['azimuthTime'])).argmin(dim='azimuthTime') # index of nearest midburst azimuth time
    poly = DopplerEstimate['dataDcPolynomial'][{'azimuthTime':iazi}]
    t0 = DopplerEstimate['t0'][{'azimuthTime':iazi}]
    
    def eval_poly(polynome, slant_range_time, t0):
        return polynome(slant_range_time - t0)
    
    fnc = xr.apply_ufunc(eval_poly, poly, slant_range_time, t0, input_core_dims=[[],['sample'],[]], vectorize=True, output_core_dims=[['sample']])
    fnc.attrs.update({'units':'Hz','long_name':'Doppler_centroid_frequency'})
    return fnc.rename('fnc')

def compute_reference_time(FMrate, dcEstimates, midburst_azimuth_time, slant_range_time, Ns):
    """
    Compute reference time
    
    Args:
        FMrate (xarray.DataArray) : dataArray of FMrate polynomials
        dcEstimates (xarray.DataArray) : DataArray of Doppler estimate polynomials
        midburst_azimuth_time (xarray.DataArray): array of mid-azimuth time
        slant_range_time (xarray.DataArray) : array of slant range time
        Ns (int): number of samples per burst
    """
    fnc = compute_DopplerCentroid_frequency(dcEstimates, midburst_azimuth_time, slant_range_time)
    ka = compute_FMrate(FMrate, midburst_azimuth_time, slant_range_time)
    nc = -fnc/ka
    nref = nc-nc.sel(sample=Ns//2)
    nref.attrs.update({'long_name':'Reference zero-Doppler azimuth time', 'units':'s'})
    return nref.rename('eta_ref')

def compute_deramping_phase(burst, doppler_centroid_rate, reference_time, azimuth_time_interval):
    """
    Compute deramping phase
    
    Args:
        burst (xarray.Dataset) : dataset of cropped burst
        doppler_centroid_rate (xarray.DataArray) : Doppler centroid rate (kt)
        reference_time (xarray.DataArray) : Reference zero-Doppler azimuth time (eta_ref)
    Return:
    (xarray.DataArray) : Deramped phase
    """
    eta = (burst['line']-burst['burst']*burst['linesPerBurst']-burst['linesPerBurst']/2)*azimuth_time_interval
    phi = np.pi*doppler_centroid_rate*(eta-reference_time)**2
    phi.attrs.update({'long_name':'Deramped_phase', 'units':'rad'})
    return phi.rename('deramping_phase')    