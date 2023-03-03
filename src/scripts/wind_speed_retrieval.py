import xsarsea
from xsarsea import windspeed
import xsar

import xarray as xr
import numpy as np

import os,re
import cv2

# output folder to store L2 products                 
out_folder = "/home1/datawork/vlheureu/IFREMER/M2_L1_retreat/first_try/"

def make_L2(safe_path):
    out_file = out_folder + safe_path.split("/")[-1] + "/output.nc"
    if os.path.exists(out_file):
        print("OK--", safe_path.split('/')[-1], "already treated")
        return
    print("....treating", safe_path.split("/")[-1])
    #loading metatada
    s1meta = xsar.Sentinel1Meta(safe_path)
    
    ## Land mask 
    s1meta.set_mask_feature('land', '/home/datawork-cersat-public/cache/public/ftp/project/sarwing/xsardata/land-polygons-split-4326/land_polygons.shp')

    ## Loading ancillary
    # set ecmwf path for ancillary wind
    s1meta.set_raster('ecmwf_0100_1h','/home/datawork-cersat-public/provider/ecmwf/forecast/hourly/0100deg/netcdf_light/%Y/%j/ECMWF_FORECAST_0100_%Y%m%d%H%M_10U_10V.nc')
    s1meta.set_raster('ecmwf_0125_1h','/home/datawork-cersat-intranet/project/ecmwf/0.125deg/1h/forecasts/%Y/%j/ecmwf_%Y%m%d%H%M.nc')

    # only keep best ecmwf  (FIXME: it's hacky, and xsar should provide a better method to handle this)
    for ecmwf_name in [ 'ecmwf_0125_1h', 'ecmwf_0100_1h' ]:
        ecmwf_infos = s1meta.rasters.loc[ecmwf_name]
        ecmwf_file = ecmwf_infos['get_function'](ecmwf_infos['resource'], date=s1meta.start_date)
        if not os.path.isfile(ecmwf_file):
                s1meta.rasters = s1meta.rasters.drop([ecmwf_name])
        else:
            map_model = { '%s_%s' % (ecmwf_name, uv) : 'model_%s' % uv for uv in ['U10', 'V10'] }

    ### Loading dataset & merging ancillary
    try :
        xsar_obj_1000m = xsar.Sentinel1Dataset(s1meta, resolution='1000m')
    except Exception as e :
        print(e)
        return 
    
    xsar_obj_1000m.dataset = xsar_obj_1000m.dataset.rename(map_model)

    ### Variables of interest 
    xsar_obj_1000m.dataset['land_mask'].values = cv2.dilate(xsar_obj_1000m.dataset['land_mask'].values.astype('uint8'),np.ones((3,3),np.uint8),iterations = 3)
    xsar_obj_1000m.dataset['sigma0_ocean'] = xr.where(xsar_obj_1000m.dataset['land_mask'], np.nan, xsar_obj_1000m.dataset['sigma0'].compute()).transpose(*xsar_obj_1000m.dataset['sigma0'].dims)
    xsar_obj_1000m.dataset['sigma0_ocean'] = xr.where(xsar_obj_1000m.dataset['sigma0_ocean'] <= 0, 1e-15, xsar_obj_1000m.dataset['sigma0_ocean'])
    xsar_obj_1000m.dataset['ancillary_wind'] = (xsar_obj_1000m.dataset.model_U10 + 1j * xsar_obj_1000m.dataset.model_V10) * np.exp(1j * np.deg2rad(xsar_obj_1000m.dataset.ground_heading))
    xsar_obj_1000m.dataset['ancillary_wind'] = xr.where(xsar_obj_1000m.dataset['land_mask'], np.nan, xsar_obj_1000m.dataset['ancillary_wind'].compute()).transpose(*xsar_obj_1000m.dataset['ancillary_wind'].dims)
    
    ##ecmwf source
    xsar_obj_1000m.dataset.attrs['ancillary_source'] = xsar_obj_1000m.dataset['model_U10'].attrs['history'].split('decoded: ')[1].strip()
    
    aux_cal_start = re.search('_V(\d+)T', os.path.basename(xsar_obj_1000m.s1meta.manifest_attrs['aux_cal'])).group(1)
    aux_cal_stop = re.search('_G(\d+)T', os.path.basename(xsar_obj_1000m.s1meta.manifest_attrs['aux_cal'])).group(1)
    ipf_version = xsar_obj_1000m.s1meta.manifest_attrs['ipf_version']
    mission_name = xsar_obj_1000m.s1meta.manifest_attrs["mission"]
    
    
    ## flattening and gmf to use
    if mission_name == "SENTINEL-1":
        print("mission_name : {}, ipf_version : {}, aux_cal_stop : {}".format(mission_name,ipf_version,aux_cal_stop))
        apply_flattening = True
        GMF_VH_NAME = "gmf_s1_v2"
    else : 
        print("mission_name should be SENTINEL-1 since we only know to use S1")
        apply_flattening = False
        GMF_VH_NAME = "gmf_rs2_v2"
        return 

    nesz_cr = xsar_obj_1000m.dataset.nesz.isel(pol=1) #(no_flattening)
    if apply_flattening : 
        xsar_obj_1000m.dataset=xsar_obj_1000m.dataset.assign(nesz_VH_final=(['atrack','xtrack'],windspeed.nesz_flattening(nesz_cr, xsar_obj_1000m.dataset.incidence)))
        xsar_obj_1000m.dataset['nesz_VH_final'].attrs["comment"] = 'nesz has been flattened using windspeed.nesz_flattening'
    else :
        xsar_obj_1000m.dataset=xsar_obj_1000m.dataset.assign(nesz_VH_final=(['atrack','xtrack'],nesz_cr.values))
        xsar_obj_1000m.dataset['nesz_VH_final'].attrs["comment"] = 'nesz has not been flattened'
   
    ## dsig
    try : 
        dsig_cr = windspeed.get_dsig("gmf_s1_v2", dataset_1000m.incidence,dataset_1000m.sigma0_ocean.sel(pol='VH'),dataset_1000m.nesz_VH_final)
    except Exception as e :
        print(e)
        return

    ## co & dual inversion
    windspeed_co, windspeed_dual = windspeed.invert_from_model(
        xsar_obj_1000m.dataset.incidence,
        xsar_obj_1000m.dataset.sigma0_ocean.isel(pol=0),
        xsar_obj_1000m.dataset.sigma0_ocean.isel(pol=1),
        #ancillary_wind=-np.conj(xsar_obj_1000m.dataset['ancillary_wind']),
        ancillary_wind=-xsar_obj_1000m.dataset['ancillary_wind'],
        dsig_cr = dsig_cr,
        model=('cmod5n',GMF_VH_NAME))
        
    xsar_obj_1000m.dataset["windspeed_co"] = np.abs(windspeed_co)
    xsar_obj_1000m.dataset["windspeed_co"].attrs["comment"] = xsar_obj_1000m.dataset["windspeed_co"].attrs["comment"].replace("wind speed and direction","wind speed")

    xsar_obj_1000m.dataset["windspeed_dual"] = np.abs(windspeed_dual)
    xsar_obj_1000m.dataset["windspeed_dual"].attrs["comment"] = xsar_obj_1000m.dataset["windspeed_dual"].attrs["comment"].replace("wind speed and direction","wind speed")
    
    ## cr inversion ##TODO
    windspeed_cr = windspeed.invert_from_model(
        xsar_obj_1000m.dataset.incidence.values,
        xsar_obj_1000m.dataset.sigma0_ocean.isel(pol=1).values,
        #ancillary_wind=-np.conj(xsar_obj_1000m.dataset['ancillary_wind']),
        dsig_cr = dsig_cr.values,
        model=GMF_VH_NAME)

    windspeed_cr = np.abs(windspeed_cr)
    xsar_obj_1000m.dataset=xsar_obj_1000m.dataset.assign(windspeed_cr=(['atrack','xtrack'],windspeed_cr))
    xsar_obj_1000m.dataset.windspeed_cr.attrs['comment'] = "wind speed inverted from model %s (%s)" % (GMF_VH_NAME, "VH")
    xsar_obj_1000m.dataset.windspeed_cr.attrs['model'] = GMF_VH_NAME
    xsar_obj_1000m.dataset.windspeed_cr.attrs['units'] = 'm/s'
    
    
    ## saving 
    
    # prepare dataset for netcdf export
    black_list = ['model_U10', 'model_V10', 'digital_number', 'gamma0_raw', 'negz',
                  'azimuth_time', 'slant_range_time', 'velocity', 'range_ground_spacing',
                  'gamma0', 'time', 'sigma0', 'nesz', 'sigma0_raw', 'sigma0_ocean', 'altitude', 'elevation',
                  'nd_co', 'nd_cr']
    variables = list(set(xsar_obj_1000m.dataset) - set(black_list))

    # complex not allowed in netcdf
    xsar_obj_1000m.dataset = xsar_obj_1000m.dataset[variables]
    
    xsar_obj_1000m.dataset['ancillary_wind_spd'] = np.abs(xsar_obj_1000m.dataset['ancillary_wind'])
    xsar_obj_1000m.dataset['ancillary_wind_dir'] = xr.ufuncs.angle(xsar_obj_1000m.dataset['ancillary_wind'])
    xsar_obj_1000m.dataset['ancillary_wind_dir'].attrs['comment'] = 'angle in radians, anticlockwise, 0=xtrack'
    del xsar_obj_1000m.dataset['ancillary_wind']
    
    xsar_obj_1000m.recompute_attrs()
    ds_1000 = xsar_obj_1000m.dataset.compute()
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

    json_gcps = json.dumps(json.loads(json.dumps(ds_1000.atrack.spatial_ref.gcps,cls=JSONEncoder)))
    ds_1000['atrack']['spatial_ref'].attrs['gcps'] = json_gcps
    ds_1000['xtrack']['spatial_ref'].attrs['gcps'] = json_gcps
    
    # remove possible incorect values on swath border
    for name in ['windspeed_co','windspeed_cr','windspeed_dual']:
        ds_1000[name].values[:,0:6] = np.nan
        ds_1000[name].values[:,-6::] = np.nan
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    ds_1000.to_netcdf(out_file)  
    print("OK--",safe_path.split('/')[-1], "OK")     


if __name__ == "__main__":
    # to treat a listing 
    listing_file = open('/home1/datahome/vlheureu/IFREMER/M4_validation/listing_safe_cyclobs_S1_to_18022023.txt', 'r')
    listing_safe = [line.strip() for line in listing_file.readlines()]
    
    for safe_path in listing_safe:
          make_L2(safe_path)     
          
    #to treat one file 
    #make_L2(safe_path) 