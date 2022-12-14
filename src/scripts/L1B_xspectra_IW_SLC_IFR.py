# -*- coding: utf-8 -*-
"""
A. Grouazel
18 Nov 2022
purpose: produce nc files from SAFE IW SLC containing cartesian x-spec computed with xsar and xsarsea
 on intra and inter bursts
"""

import xsarsea.sar_slc.processing as proc
import warnings
import xsar
import xsarsea
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import datetime
import logging
import os
import time
import pdb
from yaml import load
from yaml import CLoader as Loader
PRODUCT_VERSION = '0.1'  # version from release 17nov2022 with wavenumbers not aligned
PRODUCT_VERSION = '0.2'  # version from release 5dec2022 with wavenumbers aligned
PRODUCT_VERSION = '0.3'  # add fix for freq_sample + subgroups with subswath
PRODUCT_VERSION = '0.4'  # 12dec22 : only one tiff (one subswath) for one .nc output
PRODUCT_VERSION = '0.5'  # 14dec22 : integration of functions to avoid loading geoloc fields at high resolution
# stream = open(os.path.join(os.path.dirname(__file__), 'configuration_L1B_xspectra_IW_SLC_IFR_v1.yml'), 'r')
# conf = load(stream, Loader=Loader)  # TODO : add argument to compute_subswath_xspectra(conf=conf)
def get_memory_usage():
    try:
        import resource
        memory_used_go = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000./1000.
    except: #on windows resource is not usable
        import psutil
        memory_used_go = psutil.virtual_memory().used / 1000 / 1000 / 1000.
    str_mem = 'RAM usage: %1.1f Go'%memory_used_go
    return str_mem

def generate_IW_L1Bxspec_product(slc_iw_path,output_filename, polarization=None,dev=False):
    """

    :param tiff: str full path
    :param output_filename : str full path
    :param polarization : str : VV VH HH HV [optional]
    :apram dev: bool: allow to shorten the processing
    :return:
    """
    safe = os.path.dirname(os.path.dirname(slc_iw_path))
    logging.info('start loading the datatree %s', get_memory_usage())
    tiff_number = os.path.basename(slc_iw_path).split('-')[1].replace('iw', '')
    str_gdal = 'SENTINEL1_DS:%s:IW%s' % (safe, tiff_number)
    bu = xsar.Sentinel1Meta(str_gdal)._bursts
    chunksize = {'line': int(bu['linesPerBurst'].values), 'sample': int(bu['samplesPerBurst'].values)}
    xsarobj = xsar.Sentinel1Dataset(str_gdal, chunks=chunksize)
    # xsarobj.add_high_resolution_variables(
    #     skip_variables=['land_mask', 'elevation', 'altitude', 'ground_heading', 'range_ground_spacing'],
    #     lazy_loading=False, load_luts=False)
    #xsarobj.datatree['measurement'] = xsarobj.datatree['measurement'].assign(xsarobj.dataset.drop('land_mask'))
    dt = xsarobj.datatree
    dt.load() #took 4min to load and 35Go RAM
    logging.info('datatree loaded %s',get_memory_usage())
    one_subswath_xspectrum_dt = proc.compute_subswath_xspectra(dt,pol=polarization.upper(),dev=dev)
    logging.info('xspec intra and inter ready for %s', slc_iw_path)
    logging.debug('one_subswath_xspectrum = %s', one_subswath_xspectrum_dt)
    one_subswath_xspectrum_dt.attrs['version_xsar'] = xsar.__version__
    one_subswath_xspectrum_dt.attrs['version_xsarsea'] = xsarsea.__version__
    one_subswath_xspectrum_dt.attrs['processor'] = __file__
    one_subswath_xspectrum_dt.attrs['generation_date'] = datetime.datetime.today().strftime('%Y-%b-%d')
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename),0o0775)
        logging.info('makedir %s',os.path.dirname(output_filename))
    one_subswath_xspectrum_dt.to_netcdf(output_filename)
    logging.info('successfuly written %s', output_filename)


if __name__ == '__main__':
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    import argparse
    time.sleep(np.random.rand(1, 1)[0][0])  # to avoid issue with mkdir
    parser = argparse.ArgumentParser(description='L1BwaveIFR_IW_SLC')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwrite the existing outputs [default=False]', required=False)
    parser.add_argument('--tiff', required=True, help='tiff file full path IW SLC')
    parser.add_argument('--subswath', required=False, help='iw1 iw2... [None]', default=None)
    #parser.add_argument('--pol', required=False,choices=['VV','VH','HH','HV'], help='VV HH HV VH [None]', default=None)
    parser.add_argument('--outputdir', required=True, help='directory where to store output netCDF files')
    parser.add_argument('--dev', action='store_true', default=False,help='dev mode stops the computation early')
    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else:
        logging.basicConfig(level=logging.INFO, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    t0 = time.time()

    slc_iw_path = args.tiff
    subswath_number = os.path.basename(slc_iw_path).split('-')[1]
    polarization_from_file = os.path.basename(slc_iw_path).split('-')[3]
    subsath_nickname = '%s_%s' % (subswath_number, polarization_from_file)
    safe_basename = os.path.basename(os.path.dirname(os.path.dirname(slc_iw_path)))
    output_filename = os.path.join(args.outputdir,safe_basename, os.path.basename(
        slc_iw_path).replace('.tiff','') + '_L1B_xspec_IFR_' + PRODUCT_VERSION + '.nc')
    logging.info('mode dev is %s',args.dev)
    logging.info('output filename would be: %s',output_filename)
    if os.path.exists(output_filename) and args.overwrite is False:
        logging.info('%s already exists', output_filename)
    else:
        generate_IW_L1Bxspec_product(slc_iw_path=slc_iw_path,output_filename=output_filename, dev=args.dev,
                                     polarization=polarization_from_file)
    logging.info('peak memory usage: %s Mbytes', get_memory_usage())
    logging.info('done in %1.3f min', (time.time() - t0) / 60.)
