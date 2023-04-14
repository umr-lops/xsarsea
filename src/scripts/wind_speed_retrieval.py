import xsarsea
from xsarsea import windspeed
import xsar
import datetime 

import xarray as xr
import numpy as np

import os,re
import cv2

import logging

from xsarsea.utils import _load_config
config = _load_config()

# output folder to store L2 products                 

def make_L2(safe_path):
    out_file = config['out_folder'] + safe_path.split("/")[-1] + "/output.nc"
    if os.path.exists(out_file):
        logging.info("OK--", safe_path.split('/')[-1], "already treated")
        return
    logging.info("....treating", safe_path.split("/")[-1])
    #loading metatada
    s1meta = xsar.Sentinel1Meta(safe_path)
    
    ## Land mask 
    s1meta.set_mask_feature('land', config["land_mask"])

    ## Loading ancillary
    # set ecmwf path for ancillary wind
    s1meta.set_raster('ecmwf_0100_1h',config["path_ecmwf_0100_1h"])
    s1meta.set_raster('ecmwf_0125_1h',config["path_ecmwf_0125_1h"])

    # only keep best ecmwf  (FIXME: it's hacky, and xsar should provide a better method to handle this)
    for ecmwf_name in [ 'ecmwf_0125_1h', 'ecmwf_0100_1h' ]:
        ecmwf_infos = s1meta.rasters.loc[ecmwf_name]
        ecmwf_file = ecmwf_infos['get_function'](ecmwf_infos['resource'], date=datetime.datetime.strptime(s1meta.start_date, '%Y-%m-%d %H:%M:%S.%f'))[1]
        if not os.path.isfile(ecmwf_file):
                s1meta.rasters = s1meta.rasters.drop([ecmwf_name])
        else:
            map_model = { '%s_%s' % (ecmwf_name, uv) : 'model_%s' % uv for uv in ['U10', 'V10'] }

    ### Loading dataset & merging ancillary
    try :
        xsar_obj_1000m = xsar.Sentinel1Dataset(s1meta, resolution='1000m')
    except Exception as e :
        logging.warn(e)
        return 
        
    dataset_1000m = xsar_obj_1000m.datatree['measurement'].to_dataset()
    dataset_1000m = dataset_1000m.rename(map_model)


    ### Variables of interest 
    dataset_1000m['land_mask'].values = cv2.dilate(dataset_1000m['land_mask'].values.astype('uint8'),np.ones((3,3),np.uint8),iterations = 3)
    dataset_1000m['sigma0_ocean'] = xr.where(dataset_1000m['land_mask'], np.nan, dataset_1000m['sigma0'].compute()).transpose(*dataset_1000m['sigma0'].dims)
    dataset_1000m['sigma0_ocean'] = xr.where(dataset_1000m['sigma0_ocean'] <= 0, 1e-15, dataset_1000m['sigma0_ocean'])
    dataset_1000m['ancillary_wind'] = (dataset_1000m.model_U10 + 1j * dataset_1000m.model_V10) * np.exp(1j * np.deg2rad(dataset_1000m.ground_heading))
    dataset_1000m['ancillary_wind'] = xr.where(dataset_1000m['land_mask'], np.nan, dataset_1000m['ancillary_wind'].compute()).transpose(*dataset_1000m['ancillary_wind'].dims)
    
    ##ecmwf source
    dataset_1000m.attrs['ancillary_source'] = dataset_1000m['model_U10'].attrs['history'].split('decoded: ')[1].strip()
    
    aux_cal_start = re.search('_V(\d+)T', os.path.basename(xsar_obj_1000m.s1meta.manifest_attrs['aux_cal'])).group(1)
    aux_cal_stop = re.search('_G(\d+)T', os.path.basename(xsar_obj_1000m.s1meta.manifest_attrs['aux_cal'])).group(1)
    ipf_version = xsar_obj_1000m.s1meta.manifest_attrs['ipf_version']
    mission_name = xsar_obj_1000m.s1meta.manifest_attrs["mission"]
    
    
    ## flattening and gmf to use
    if mission_name == "SENTINEL-1":
        logging.info("mission_name : {}, ipf_version : {}, aux_cal_stop : {}".format(mission_name,ipf_version,aux_cal_stop))
        apply_flattening = True
        GMF_VH_NAME = "gmf_s1_v2"
    else : 
        logging.info("mission_name should be SENTINEL-1 since we only know to use S1")
        apply_flattening = False
        GMF_VH_NAME = "gmf_rs2_v2"
        return 

    nesz_cr = dataset_1000m.nesz.isel(pol=1) #(no_flattening)
    if apply_flattening : 
        dataset_1000m=dataset_1000m.assign(nesz_VH_final=(['line','sample'],windspeed.nesz_flattening(nesz_cr, dataset_1000m.incidence)))
        dataset_1000m['nesz_VH_final'].attrs["comment"] = 'nesz has been flattened using windspeed.nesz_flattening'
    else :
        dataset_1000m=dataset_1000m.assign(nesz_VH_final=(['line','sample'],nesz_cr.values))
        dataset_1000m['nesz_VH_final'].attrs["comment"] = 'nesz has not been flattened'
   

   
    ## dsig
    try : 
        dsig_cr = windspeed.get_dsig("gmf_s1_v2", dataset_1000m.incidence,dataset_1000m.sigma0_ocean.sel(pol='VH'),dataset_1000m.nesz_VH_final)
    except Exception as e :
        logging.warn(e)
        return

    ## co & dual inversion
    windspeed_co, windspeed_dual = windspeed.invert_from_model(
        dataset_1000m.incidence,
        dataset_1000m.sigma0_ocean.isel(pol=0),
        dataset_1000m.sigma0_ocean.isel(pol=1),
        #ancillary_wind=-np.conj(dataset_1000m['ancillary_wind']),
        ancillary_wind=-dataset_1000m['ancillary_wind'],
        dsig_cr = dsig_cr,
        model=('cmod5n',GMF_VH_NAME))
        
    dataset_1000m["windspeed_co"] = np.abs(windspeed_co)
    dataset_1000m["windspeed_co"].attrs["comment"] = dataset_1000m["windspeed_co"].attrs["comment"].replace("wind speed and direction","wind speed")

    dataset_1000m["windspeed_dual"] = np.abs(windspeed_dual)
    dataset_1000m["windspeed_dual"].attrs["comment"] = dataset_1000m["windspeed_dual"].attrs["comment"].replace("wind speed and direction","wind speed")
    
    ## cr inversion ##TODO
    windspeed_cr = windspeed.invert_from_model(
        dataset_1000m.incidence.values,
        dataset_1000m.sigma0_ocean.isel(pol=1).values,
        #ancillary_wind=-np.conj(dataset_1000m['ancillary_wind']),
        dsig_cr = dsig_cr.values,
        model=GMF_VH_NAME)

    windspeed_cr = np.abs(windspeed_cr)
    dataset_1000m=dataset_1000m.assign(windspeed_cr=(['line','sample'],windspeed_cr))
    dataset_1000m.windspeed_cr.attrs['comment'] = "wind speed inverted from model %s (%s)" % (GMF_VH_NAME, "VH")
    dataset_1000m.windspeed_cr.attrs['model'] = GMF_VH_NAME
    dataset_1000m.windspeed_cr.attrs['units'] = 'm/s'
    
    
    ## saving 
    dataset_1000m['sigma0_ocean_VV'] = dataset_1000m['sigma0_ocean'].sel(pol='VV')
    dataset_1000m['sigma0_ocean_VH'] = dataset_1000m['sigma0_ocean'].sel(pol='VH')
    
    # prepare dataset for netcdf export
    black_list = ['model_U10', 'model_V10', 'digital_number', 'gamma0_raw', 'negz',
                  'azimuth_time', 'slant_range_time', 'velocity', 'range_ground_spacing',
                  'gamma0', 'time', 'sigma0', 'nesz', 'sigma0_raw', 'sigma0_ocean', 'altitude', 'elevation',
                  'nd_co', 'nd_cr']
    variables = list(set(dataset_1000m) - set(black_list))

    # complex not allowed in netcdf
    dataset_1000m = dataset_1000m[variables]
    
    dataset_1000m['ancillary_wind_spd'] = np.abs(dataset_1000m['ancillary_wind'])
    #dataset_1000m['ancillary_wind_dir'] = xr.ufuncs.angle(dataset_1000m['ancillary_wind'])
    #dataset_1000m['ancillary_wind_dir'].attrs['comment'] = 'angle in radians, anticlockwise, 0=xtrack'
    del dataset_1000m['ancillary_wind']
    
    xsar_obj_1000m.recompute_attrs()
    ds_1000 = dataset_1000m.compute()
    # some type like date or json must be converted to string
    ds_1000.attrs['start_date'] = str(ds_1000.attrs['start_date'])
    ds_1000.attrs['stop_date'] = str(ds_1000.attrs['stop_date'])
    ds_1000.attrs['footprint'] = str(ds_1000.attrs['footprint'])
    # add ipf_version and aux_cal_stop
    ds_1000.attrs['aux_cal_start'] = str(aux_cal_start)
    ds_1000.attrs['aux_cal_stop'] = str(aux_cal_stop)

    # encode gcps as json string
    import json
    class JSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)

    json_gcps = json.dumps(json.loads(json.dumps(ds_1000.line.spatial_ref.gcps,cls=JSONEncoder)))
    ds_1000['line']['spatial_ref'].attrs['gcps'] = json_gcps
    ds_1000['sample']['spatial_ref'].attrs['gcps'] = json_gcps
    
    # remove possible incorect values on swath border
    for name in ['windspeed_co','windspeed_cr','windspeed_dual']:
        ds_1000[name].values[:,0:6] = np.nan
        ds_1000[name].values[:,-6::] = np.nan
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    ds_1000.to_netcdf(out_file)  
    print("OK--",safe_path.split('/')[-1], "OK")     


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                    datefmt='%d/%m/%Y %H:%M:%S')


    ####  to treat a listing 
    # listing_file = open('/home1/datahome/vlheureu/IFREMER/M4_validation/listing_safe_cyclobs_S1_to_18022023.txt', 'r')
    #listing_safe = [line.strip() for line in listing_file.readlines()]
    
    #for safe_path in listing_safe:
    #      make_L2(safe_path)     
          
    #### to treat one file 
    safe_path = config["safe_example"]
    make_L2(safe_path) 