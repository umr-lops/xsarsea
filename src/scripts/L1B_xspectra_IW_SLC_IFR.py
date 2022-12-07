# -*- coding: utf-8 -*-
"""
A Grouazel
18 Nov 2022
purpose: produce nc files from SAFE IW SLC containing cartesian x-spec computed with xsar and xsarsea
 on intra and inter bursts
 first test on
 slc_iw_path = 'SENTINEL1_DS:/home/datawork-cersat-public/project/mpc-sentinel1/data/esa/sentinel-1a/L1/IW/S1A_IW_SLC__1S/2022/250/S1A_IW_SLC__1SDV_20220907T133016_20220907T133043_044899_055CF5_7A7B.SAFE:IW2'
 (the IW1 and IW3 are no working (18nov2022) because of the lack of normalization size for kx and ky)
"""

import xsarsea.sar_slc.processing as proc
import warnings
import xsar
import xsarsea
from tqdm import tqdm
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import datetime
import datatree
import logging
import xarray as xr
import glob
import os
import time
import pdb

from yaml import load, dump
from yaml import CLoader as Loader, CDumper as Dumper
from scipy import interpolate
# nperseg = {'range': 512, 'azimuth': 512}
# noverlap = {'range': 256, 'azimuth': 256}
# N_look = 3
# look_width = 0.2
# look_overlap = 0.1
#POLARIZATION = 'VV'
PRODUCT_VERSION = '0.1' # version from release 17nov2022 with wavenumbers not aligned
PRODUCT_VERSION = '0.2' # version from release 5dec2022 with wavenumbers aligned
PRODUCT_VERSION = '0.3' # add fix for freq_sample + subgroups with subswath
#XSPEC_WL_LIMIT = 40 #m
# ref_kx = np.array([-0.85173872, -0.84841162, -0.84508451, -0.84175741, -0.83843031,
#        -0.8351032 , -0.8317761 , -0.82844899, -0.82512189, -0.82179478,
#        -0.81846768, -0.81514057, -0.81181347, -0.80848637, -0.80515926,
#        -0.80183216, -0.79850505, -0.79517795, -0.79185084, -0.78852374,
#        -0.78519664, -0.78186953, -0.77854243, -0.77521532, -0.77188822,
#        -0.76856111, -0.76523401, -0.7619069 , -0.7585798 , -0.7552527 ,
#        -0.75192559, -0.74859849, -0.74527138, -0.74194428, -0.73861717,
#        -0.73529007, -0.73196297, -0.72863586, -0.72530876, -0.72198165,
#        -0.71865455, -0.71532744, -0.71200034, -0.70867323, -0.70534613,
#        -0.70201903, -0.69869192, -0.69536482, -0.69203771, -0.68871061,
#        -0.6853835 , -0.6820564 , -0.67872929, -0.67540219, -0.67207509,
#        -0.66874798, -0.66542088, -0.66209377, -0.65876667, -0.65543956,
#        -0.65211246, -0.64878536, -0.64545825, -0.64213115, -0.63880404,
#        -0.63547694, -0.63214983, -0.62882273, -0.62549562, -0.62216852,
#        -0.61884142, -0.61551431, -0.61218721, -0.6088601 , -0.605533  ,
#        -0.60220589, -0.59887879, -0.59555169, -0.59222458, -0.58889748,
#        -0.58557037, -0.58224327, -0.57891616, -0.57558906, -0.57226195,
#        -0.56893485, -0.56560775, -0.56228064, -0.55895354, -0.55562643,
#        -0.55229933, -0.54897222, -0.54564512, -0.54231802, -0.53899091,
#        -0.53566381, -0.5323367 , -0.5290096 , -0.52568249, -0.52235539,
#        -0.51902828, -0.51570118, -0.51237408, -0.50904697, -0.50571987,
#        -0.50239276, -0.49906566, -0.49573855, -0.49241145, -0.48908434,
#        -0.48575724, -0.48243014, -0.47910303, -0.47577593, -0.47244882,
#        -0.46912172, -0.46579461, -0.46246751, -0.45914041, -0.4558133 ,
#        -0.4524862 , -0.44915909, -0.44583199, -0.44250488, -0.43917778,
#        -0.43585067, -0.43252357, -0.42919647, -0.42586936, -0.42254226,
#        -0.41921515, -0.41588805, -0.41256094, -0.40923384, -0.40590674,
#        -0.40257963, -0.39925253, -0.39592542, -0.39259832, -0.38927121,
#        -0.38594411, -0.382617  , -0.3792899 , -0.3759628 , -0.37263569,
#        -0.36930859, -0.36598148, -0.36265438, -0.35932727, -0.35600017,
#        -0.35267307, -0.34934596, -0.34601886, -0.34269175, -0.33936465,
#        -0.33603754, -0.33271044, -0.32938333, -0.32605623, -0.32272913,
#        -0.31940202, -0.31607492, -0.31274781, -0.30942071, -0.3060936 ,
#        -0.3027665 , -0.29943939, -0.29611229, -0.29278519, -0.28945808,
#        -0.28613098, -0.28280387, -0.27947677, -0.27614966, -0.27282256,
#        -0.26949546, -0.26616835, -0.26284125, -0.25951414, -0.25618704,
#        -0.25285993, -0.24953283, -0.24620572, -0.24287862, -0.23955152,
#        -0.23622441, -0.23289731, -0.2295702 , -0.2262431 , -0.22291599,
#        -0.21958889, -0.21626179, -0.21293468, -0.20960758, -0.20628047,
#        -0.20295337, -0.19962626, -0.19629916, -0.19297205, -0.18964495,
#        -0.18631785, -0.18299074, -0.17966364, -0.17633653, -0.17300943,
#        -0.16968232, -0.16635522, -0.16302811, -0.15970101, -0.15637391,
#        -0.1530468 , -0.1497197 , -0.14639259, -0.14306549, -0.13973838,
#        -0.13641128, -0.13308418, -0.12975707, -0.12642997, -0.12310286,
#        -0.11977576, -0.11644865, -0.11312155, -0.10979444, -0.10646734,
#        -0.10314024, -0.09981313, -0.09648603, -0.09315892, -0.08983182,
#        -0.08650471, -0.08317761, -0.07985051, -0.0765234 , -0.0731963 ,
#        -0.06986919, -0.06654209, -0.06321498, -0.05988788, -0.05656077,
#        -0.05323367, -0.04990657, -0.04657946, -0.04325236, -0.03992525,
#        -0.03659815, -0.03327104, -0.02994394, -0.02661684, -0.02328973,
#        -0.01996263, -0.01663552, -0.01330842, -0.00998131, -0.00665421,
#        -0.0033271 ,  0.        ,  0.0033271 ,  0.00665421,  0.00998131,
#         0.01330842,  0.01663552,  0.01996263,  0.02328973,  0.02661684,
#         0.02994394,  0.03327104,  0.03659815,  0.03992525,  0.04325236,
#         0.04657946,  0.04990657,  0.05323367,  0.05656077,  0.05988788,
#         0.06321498,  0.06654209,  0.06986919,  0.0731963 ,  0.0765234 ,
#         0.07985051,  0.08317761,  0.08650471,  0.08983182,  0.09315892,
#         0.09648603,  0.09981313,  0.10314024,  0.10646734,  0.10979444,
#         0.11312155,  0.11644865,  0.11977576,  0.12310286,  0.12642997,
#         0.12975707,  0.13308418,  0.13641128,  0.13973838,  0.14306549,
#         0.14639259,  0.1497197 ,  0.1530468 ,  0.15637391,  0.15970101,
#         0.16302811,  0.16635522,  0.16968232,  0.17300943,  0.17633653,
#         0.17966364,  0.18299074,  0.18631785,  0.18964495,  0.19297205,
#         0.19629916,  0.19962626,  0.20295337,  0.20628047,  0.20960758,
#         0.21293468,  0.21626179,  0.21958889,  0.22291599,  0.2262431 ,
#         0.2295702 ,  0.23289731,  0.23622441,  0.23955152,  0.24287862,
#         0.24620572,  0.24953283,  0.25285993,  0.25618704,  0.25951414,
#         0.26284125,  0.26616835,  0.26949546,  0.27282256,  0.27614966,
#         0.27947677,  0.28280387,  0.28613098,  0.28945808,  0.29278519,
#         0.29611229,  0.29943939,  0.3027665 ,  0.3060936 ,  0.30942071,
#         0.31274781,  0.31607492,  0.31940202,  0.32272913,  0.32605623,
#         0.32938333,  0.33271044,  0.33603754,  0.33936465,  0.34269175,
#         0.34601886,  0.34934596,  0.35267307,  0.35600017,  0.35932727,
#         0.36265438,  0.36598148,  0.36930859,  0.37263569,  0.3759628 ,
#         0.3792899 ,  0.382617  ,  0.38594411,  0.38927121,  0.39259832,
#         0.39592542,  0.39925253,  0.40257963,  0.40590674,  0.40923384,
#         0.41256094,  0.41588805,  0.41921515,  0.42254226,  0.42586936,
#         0.42919647,  0.43252357,  0.43585067,  0.43917778,  0.44250488,
#         0.44583199,  0.44915909,  0.4524862 ,  0.4558133 ,  0.45914041,
#         0.46246751,  0.46579461,  0.46912172,  0.47244882,  0.47577593,
#         0.47910303,  0.48243014,  0.48575724,  0.48908434,  0.49241145,
#         0.49573855,  0.49906566,  0.50239276,  0.50571987,  0.50904697,
#         0.51237408,  0.51570118,  0.51902828,  0.52235539,  0.52568249,
#         0.5290096 ,  0.5323367 ,  0.53566381,  0.53899091,  0.54231802,
#         0.54564512,  0.54897222,  0.55229933,  0.55562643,  0.55895354,
#         0.56228064,  0.56560775,  0.56893485,  0.57226195,  0.57558906,
#         0.57891616,  0.58224327,  0.58557037,  0.58889748,  0.59222458,
#         0.59555169,  0.59887879,  0.60220589,  0.605533  ,  0.6088601 ,
#         0.61218721,  0.61551431,  0.61884142,  0.62216852,  0.62549562,
#         0.62882273,  0.63214983,  0.63547694,  0.63880404,  0.64213115,
#         0.64545825,  0.64878536,  0.65211246,  0.65543956,  0.65876667,
#         0.66209377,  0.66542088,  0.66874798,  0.67207509,  0.67540219,
#         0.67872929,  0.6820564 ,  0.6853835 ,  0.68871061,  0.69203771,
#         0.69536482,  0.69869192,  0.70201903,  0.70534613,  0.70867323,
#         0.71200034,  0.71532744,  0.71865455,  0.72198165,  0.72530876,
#         0.72863586,  0.73196297,  0.73529007,  0.73861717,  0.74194428,
#         0.74527138,  0.74859849,  0.75192559,  0.7552527 ,  0.7585798 ,
#         0.7619069 ,  0.76523401,  0.76856111,  0.77188822,  0.77521532,
#         0.77854243,  0.78186953,  0.78519664,  0.78852374,  0.79185084,
#         0.79517795,  0.79850505,  0.80183216,  0.80515926,  0.80848637,
#         0.81181347,  0.81514057,  0.81846768,  0.82179478,  0.82512189,
#         0.82844899,  0.8317761 ,  0.8351032 ,  0.83843031,  0.84175741,
#         0.84508451,  0.84841162])
# ref_ky = np.array([-0.1524951, -0.149505 , -0.1465149, -0.1435248, -0.1405347,
#        -0.1375446, -0.1345545, -0.1315644, -0.1285743, -0.1255842,
#        -0.1225941, -0.119604 , -0.1166139, -0.1136238, -0.1106337,
#        -0.1076436, -0.1046535, -0.1016634, -0.0986733, -0.0956832,
#        -0.0926931, -0.089703 , -0.0867129, -0.0837228, -0.0807327,
#        -0.0777426, -0.0747525, -0.0717624, -0.0687723, -0.0657822,
#        -0.0627921, -0.059802 , -0.0568119, -0.0538218, -0.0508317,
#        -0.0478416, -0.0448515, -0.0418614, -0.0388713, -0.0358812,
#        -0.0328911, -0.029901 , -0.0269109, -0.0239208, -0.0209307,
#        -0.0179406, -0.0149505, -0.0119604, -0.0089703, -0.0059802,
#        -0.0029901,  0.       ,  0.0029901,  0.0059802,  0.0089703,
#         0.0119604,  0.0149505,  0.0179406,  0.0209307,  0.0239208,
#         0.0269109,  0.029901 ,  0.0328911,  0.0358812,  0.0388713,
#         0.0418614,  0.0448515,  0.0478416,  0.0508317,  0.0538218,
#         0.0568119,  0.059802 ,  0.0627921,  0.0657822,  0.0687723,
#         0.0717624,  0.0747525,  0.0777426,  0.0807327,  0.0837228,
#         0.0867129,  0.089703 ,  0.0926931,  0.0956832,  0.0986733,
#         0.1016634,  0.1046535,  0.1076436,  0.1106337,  0.1136238,
#         0.1166139,  0.119604 ,  0.1225941,  0.1255842,  0.1285743,
#         0.1315644,  0.1345545,  0.1375446,  0.1405347,  0.1435248,
#         0.1465149,  0.149505])


stream = open(os.path.join(os.path.dirname(__file__),'configuration_L1B_xspectra_IW_SLC_IFR_v1.yml'), 'r')
conf = load(stream, Loader=Loader) #TODO : add argument to compute_subswath_xspectra(conf=conf)

def generate_IW_L1Bxspec_product(safe,subswath=None,dev=False):
    """

    :param safe: str
    :param subswath: str iw1 iw2 or iw3, if None -> all the subswath are treated
    :return:
    """
    if subswath is not None:
        pattern = '*%s*tiff' %(subswath)
    else:
        pattern = '*tiff'
    lst_tiff = sorted(glob.glob(os.path.join(safe, 'measurement', pattern)))
    logging.info('Nb tiff found : %s',len(lst_tiff))
    pbar = tqdm(range(len(lst_tiff)), desc='start')
    all_subswath_xspec = {}
    for ii in pbar:
        str_mem = 'peak memory usage: %s Mbytes', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.
        slc_iw_path = lst_tiff[ii]
        subswath_number = os.path.basename(slc_iw_path).split('-')[1]
        polarization =os.path.basename(slc_iw_path).split('-')[3]
        tiff_number = os.path.basename(slc_iw_path).split('-')[1].replace('iw', '')
        fullpathsafeSLC = os.path.dirname(os.path.dirname(slc_iw_path))
        str_gdal = 'SENTINEL1_DS:%s:IW%s' % (fullpathsafeSLC, tiff_number)
        bu = xsar.Sentinel1Meta(str_gdal)._bursts
        chunksize = {'line': int(bu['linesPerBurst'].values), 'sample': int(bu['samplesPerBurst'].values)}
        xsarobj = xsar.Sentinel1Dataset(str_gdal,chunks=chunksize)
        xsarobj.add_high_resolution_variables()
        # dt = xsar.open_datatree(slc_iw_path)
        dt = xsarobj.datatree
        pbar.set_description('tiff processing %s #### total:%s' % (os.path.basename(slc_iw_path)[0:6],len(lst_tiff)))
        if dev:
            import pickle
            fid = open('/home1/scratch/agrouaze/l1b/assset_one_subswath_xspectrum_dt.pkl','rb')
            one_subswath_xspectrum_dt = pickle.load(fid)['xs']
            fid.close()
        else:
        
            one_subswath_xspectrum_dt = proc.compute_subswath_xspectra(dt)
        logging.info('xspec intra and inter ready for %s',slc_iw_path)
        #all_subswath_xspec['subswath_%s'%(ii+1)] =one_subswath_xspectrum_dt
        all_subswath_xspec['%s_%s'%(subswath_number,polarization)] = one_subswath_xspectrum_dt
        # if False:
        #     import pdb
        #     import pickle
        #     outfff = '/home1/scratch/agrouaze/l1b/assset_one_subswath_xspectrum_dt.pkl'
        #     fid = open(outfff,'wb')
        #     pickle.dump({'xs':one_subswath_xspectrum_dt},fid)
        #     fid.close()
        #     print('outfff',outfff)
        logging.info('one_subswath_xspectrum = %s', one_subswath_xspectrum_dt)
        logging.info(
            'time to get all X-spectra on WV with N_look:%s look_width %s look_overlap %s nperseg %sx%s  noverlap %sx%s : %1.2f seconds',
            conf['nlooks'], conf['look_width'], conf['look_overlap'], conf['intra']['nperseg']['range'], conf['intra']['nperseg']['azimuth'],
            conf['intra']['noverlap']['range'],
            conf['intra']['noverlap']['azimuth'], time.time() - t0)
   #      ds = xr.Dataset()
   #      ds['filename'] = xr.DataArray(data=[os.path.basename(tiff_full_path)], dims=['n_WV'])
   #      # all_kx.append(allspecs['kx'].values.T)
   #      # all_ky.append(allspecs['ky'].values.T)
   #      for vv in ['cross-spectrum_2tau']:
   #          tmp_xspec = allspecs[vv].mean(dim='2tau')
   #          #resampling of the cartesian xspec
   #          x = tmp_xspec.kx.values
   #          y = tmp_xspec.ky.values
   #          f = interpolate.interp2d(x, y, tmp_xspec.values, kind='linear')
   #          newxspec = f(ref_kx,ref_ky).T
   #          logging.info('newxspec : %s',newxspec.shape)
   #          tmp_xspec_interp = xr.DataArray(data=newxspec,coords={'kx':ref_kx,'ky':ref_ky},dims=['kx','ky'])
   #          tmp_xspec_interp = tmp_xspec_interp.where(np.logical_and(np.abs(tmp_xspec_interp.kx) <= 2*np.pi/XSPEC_WL_LIMIT,
   #                                                                   np.abs(tmp_xspec_interp.ky) <= 2*np.pi/XSPEC_WL_LIMIT),
   #                                         drop=True)
   #          # if store different kx and ky for each WV (ie no interpolation on a common cartesian grid) -> no need to replace coords (below)
   #          # newcoords_x = xr.DataArray(np.arange(tmp_xspec_interp.shape[0]),dims=['kx'])
   #          # newcoords_y = xr.DataArray(np.arange(tmp_xspec_interp.shape[1]), dims=['ky'])
   #          # tmp_xspec_interp = tmp_xspec_interp.assign_coords({'kxi':newcoords_x,
   #          #                          'kyi': newcoords_y,
   #          #                          })
   #          # tmp_xspec_interp = tmp_xspec_interp.swap_dims({'kx':'kxi','ky':'kyi'})
   #          # tmp_xspec_interp = tmp_xspec_interp.drop('ky')
   #          # tmp_xspec_interp = tmp_xspec_interp.drop('kx')
   #          tmp_xspec_interp.attrs['lower_wavelength_limit_m'] = XSPEC_WL_LIMIT
   #          for tt in ['real', 'imag']:
   #              if tt == 'real':
   #                  ds[vv + '_' + tt] = tmp_xspec_interp.real
   #              else:
   #                  ds[vv + '_' + tt] = tmp_xspec_interp.imag
   #
   #      all_ds.append(ds)
   # # final_ds = xr.merge(all_ds)
   #  final_ds = xr.concat(all_ds,dim='n_WV')
    #print('xr.merge(all_kx)',xr.merge(all_kx))
    #print('mat',np.vstack(all_kx).shape)
    #final_ds['kxs'] = xr.DataArray(np.vstack(all_kx),dims=['n_WV','kxlen'],coords={'n_WV':final_ds.n_WV,'kxlen':np.arange(len(all_kx[0]))})#xr.merge(all_kx).name('kxs')
    #final_ds['kys'] = xr.DataArray(np.vstack(all_ky),dims=['n_WV','kylen'],coords={'n_WV':final_ds.n_WV,'kylen':np.arange(len(all_ky[0]))})#xr.merge(all_ky)
#    final_dt = datatree.DataTree.from_dict(all_subwath_xspec)
    final_dt = datatree.DataTree()
    for yy in all_subswath_xspec:
        logging.info('yy = %s',yy)
        final_dt[yy] = all_subswath_xspec[yy]
    final_dt.attrs['version_xsar'] = xsar.__version__
    final_dt.attrs['version_xsarsea'] = xsarsea.__version__
    final_dt.attrs['processor'] = __file__
    final_dt.attrs['generation_date'] = datetime.datetime.today().strftime('%Y-%b-%d')
    final_dt.attrs['N_look'] = conf['nlooks']
    final_dt.attrs['look_width'] = conf['look_width']
    final_dt.attrs['look_overlap'] = conf['look_overlap']
    for ii in conf['intra']['nperseg']:
        final_dt.attrs['nperseg_+%s' % ii] = conf['intra']['nperseg'][ii]
    for ii in conf['intra']['noverlap']:
        final_dt.attrs['noverlap_+%s' % ii] = conf['intra']['noverlap'][ii]
    final_dt.to_netcdf(output_filename)
    logging.info('successfuly written %s', output_filename)

if __name__ == '__main__':
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    import argparse
    import resource
    time.sleep(np.random.rand(1, 1)[0][0])  # to avoid issue with mkdir
    parser = argparse.ArgumentParser(description='L1BwaveIFR_IW_SLC')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwrite the existing outputs [default=False]', required=False)
    parser.add_argument('--safe',required=True, help='SAFE full path IW SLC')
    parser.add_argument('--subswath',required=False,help='iw1 iw2... [None]',default=None)
    parser.add_argument('--outputdir', required=True, help='directory where to store output netCDF files')
    args = parser.parse_args()

    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else:
        logging.basicConfig(level=logging.INFO, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    t0 = time.time()
    if args.safe[-1] == '/':
        safepath = args.safe[0:-1]
    else:
        safepath = args.safe
    output_filename = os.path.join(args.outputdir, os.path.basename(
        safepath) + '_L1B_xspec_IFR_' + PRODUCT_VERSION + '.nc')
    if os.path.exists(output_filename) and args.overwrite is False:
        logging.info('%s already exists',output_filename)
    else:
        generate_IW_L1Bxspec_product(safe=safepath,subswath=args.subswath,dev=False)
    logging.info('peak memory usage: %s Mbytes', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.)
    logging.info('done in %1.3f min', (time.time() - t0) / 60.)
