
"""
28 June 2022
Grouazel
"""
import os
import logging
import xsar
import copy
import xsarsea

import numpy as np
from xsarsea.slc_image_normalization import get_normalized_image
import logging
#from xsarsea.cross_spectra_core import read_slc
def from_tiff_to_modulation(tiff_full_path,polarization='VV'):
    """

    Parameters
    ----------
    tiff_full_path str

    Returns
    -------

    """
    logging.debug('tiff_full_path : %s',tiff_full_path)
    fullpathsafeSLC = os.path.dirname(os.path.dirname(tiff_full_path))
    if 'WV' in tiff_full_path:
        tiff_number = os.path.basename(tiff_full_path).split('-')[-1].replace('.tiff', '')
        str_gdal = 'SENTINEL1_DS:%s:WV_%s' % (fullpathsafeSLC, tiff_number)
    elif 'IW' in tiff_full_path and 'SLC' in fullpathsafeSLC:
        tiff_number = os.path.basename(tiff_full_path).split('-')[1].replace('iw', '')
        str_gdal = 'SENTINEL1_DS:%s:IW%s' % (fullpathsafeSLC, tiff_number)
    # imagette_number = os.path.basename(tiff_full_path).split('-')[-1].replace('.tiff', '')
    fullpathsafeSLC = os.path.dirname(os.path.dirname(tiff_full_path))
    logging.debug ('fullpathsafeSLC %s',fullpathsafeSLC)
    subs = xsar.Sentinel1Meta(fullpathsafeSLC).subdatasets
    #s1ds = xsar.Sentinel1Dataset(subs.index[int(tiff_number) - 1])
    s1obj = xsar.Sentinel1Dataset(str_gdal)
    s1obj.add_high_resolution_variables()
    s1dataset = s1obj.dataset
    DN_vv_slc_meters_modulation = from_datasetSLC_to_modulation(s1dataset, polarization=polarization)
    return DN_vv_slc_meters_modulation




def from_datasetSLC_to_modulation(s1dataset,polarization):
    """

    Parameters
    ----------
    s1dataset: xsar xarray Dataset

    Returns
    -------

    """
    slc_meters = xsarsea.cross_spectra_core.read_slc(s1dataset)
    #logging.debug('slc_meters : %s', slc_meters)
    DN_vv_slc_meters = slc_meters['digital_number'].sel(pol=polarization)
    # DN_vv_slc_meters = read_slc(DN_vv_slc)
    lowpassval = [750., 750.]
    # im_spacing = np.array([s1dataset.attrs['pixel_atrack_m'], s1dataset.attrs['pixel_xtrack_m']])
    im_spacing = np.array([s1dataset['lineSpacing'].values, s1dataset['sampleSpacing'].values])
    DN_vv_slc_meters_modulation = get_normalized_image(copy.copy(DN_vv_slc_meters), im_spacing,
                                                                               lowpass_width=lowpassval)
    # DN_vv_slc_meters_modulation.attrs['pixel_atrack_m'] = s1dataset.attrs['pixel_atrack_m']
    # DN_vv_slc_meters_modulation.attrs['pixel_xtrack_m'] = s1dataset.attrs['pixel_xtrack_m']
    DN_vv_slc_meters_modulation = DN_vv_slc_meters_modulation.drop('pol')
    DN_vv_slc_meters_modulation = DN_vv_slc_meters_modulation.drop('spatial_ref')
    return DN_vv_slc_meters_modulation