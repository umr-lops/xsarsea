#!/usr/bin/env python
# coding=utf-8
"""
"""
import numpy as np
import xarray as xr


def netcdf_compliant(dataset):
    """
    Create a dataset that can be written on disk with xr.Dataset.to_netcdf() function. It split complex variable in real and imaginary variable

    Args:
        dataset (xarray.Dataset): dataset to be transform
    """
    var_to_rm = list()
    var_to_add = list()
    for i in dataset.variables.keys():
        if dataset[i].dtype==complex:
            re = dataset[i].real
            im = dataset[i].imag
            var_to_add.append({str(i)+'_Re':re, str(i)+'_Im':im})
            var_to_rm.append(str(i))
    return xr.merge([dataset.drop_vars(var_to_rm), *var_to_add])


def gaussian_kernel(width, spacing, truncate = 3.):
    """
    Compute a Gaussian kernel for filtering
    
    Args:
        width (dict): form {name of dimension (str): width in [m] (float)}
        spacing (dict): form {name of dimension (str): spacing in [m] (float)}
        truncate (float): gaussian shape is truncate at +/- (truncate x width) value
    """
    gk = 1.
    for d in width.keys():
        coord = np.arange(-truncate * width[d], truncate*width[d], spacing[d])
        coord = xr.DataArray(coord, dims=d, coords={d:coord})
        gk = gk*np.exp(-coord**2/(2 * width[d]**2))
    gk/=gk.sum()
    return gk

def xndindex(sizes):
    """
    xarray equivalent of np.ndindex iterator with defined dimension names
    
    Args:
        sizes (dict): dict of form {dimension_name (str): size(int)}
    Return:
        list of dict (should be updated to iterator)
    """
    from itertools import repeat
            
    for d,k in zip(repeat(tuple(sizes.keys())),zip(np.ndindex(tuple(sizes.values())))):
        yield {k:l for k,l in zip(d,k[0])}

def xtiling(ds, nperseg, noverlap = 0, centering=False, side='left', prefix='tile_'):
    """
    Define tiles indexes of an xarray of abritrary shape. Name of returned coordinates are prefix+nperseg keys()
    
    Note1: Coordinates of returned arrays depends on type of ds.
    If ds is an xarray instance, returned coordinates are in ds coordinate referential.
    If ds is a dict of shape, returned coordinates are assumed starting from zero.
    
    Note2 : If nperseg is a dict and contains one value set as None (or zero), a tile dimension is created along the corresponding dimension with nperseg set as
    the corresponding shape of ds. This is a diferent behaviour than not providing the dimension in nperseg
    
   
    Args:
        ds (xarray.DataArray or xarray.Dataset or dict). Array to tile or dict with form {dim1 (str):size1 (int), dim2 (str):size2 (int), ...}
        nperseg (int or dict): number of point per tile for each dimension. If defined as int, nperseg is applied on all dimensions of ds. If ds is a dict, form must be {tile_dim1 (str):number_of_point_per_segment(int), tile_dim2 (str):number_of_point_per_segment(int), ...}
        noverlap (int or dict, optional): Number of overlapping points between tiles for each dimension (default is zero). dict of form {tile_dim1 (str):number_of_overlapping_point(int), tile_dim2 (str):number_of_overlapping_point(int), ...}
        centering (dict of bool, optional): If False, first tile starts at first index. If True, number of unused points are equal on each side (see side parameter)
        side (str): 'left' or 'right' If centering is True and unused number of points is odd, side parameter defines on which side there is one more ununsed point
        prefix (str, optional): prefix to add to tiles dimension name in order to define tile coordinates
    Return:
        (dict of xarray.DataArray): keys are the same as nperseg keys and values are xarray indexes of tile in the corresponding dimension.
    """
    import warnings

    sizes = ds.sizes if isinstance(ds, xr.DataArray) or isinstance(ds, xr.Dataset) else ds

    if isinstance(nperseg,int):
        nperseg = dict.fromkeys(sizes.keys(),nperseg)

    if isinstance(noverlap,int):
        noverlap = dict.fromkeys(nperseg,noverlap)
    elif nperseg.keys() != noverlap.keys():
        noverlap = {d: 0 if d not in noverlap.keys() else noverlap[d] for d in nperseg.keys()}
    else:
        pass
    
    if centering is True:
        centering = dict.fromkeys(nperseg, True)
    elif centering is False:
        centering = dict.fromkeys(nperseg, False)
    elif nperseg.keys() != centering.keys():
        centering = {d: False if d not in centering.keys() else centering[d] for d in nperseg.keys()}
    else:
        pass
    
    if side=='left':
        side = dict.fromkeys(nperseg, 'left')
    elif side=='right':
        side = dict.fromkeys(nperseg, 'right')
    elif nperseg.keys() != side.keys():
        side = {d:'left' if d not in side.keys() else side[d] for d in nperseg.keys()}
    else:
        pass

    dims = nperseg.keys() # list of dimensions to work on
    
    for d in dims:
        if nperseg[d] in (0,None):
            nperseg[d] = sizes[d]
            noverlap[d]=0
        if sizes[d]<nperseg[d]:
            warnings.warn("Dimension '{}' ({}) is smaller than required nperseg :{}. nperseg is ajusted accordingly and noverlap forced to zero".format(d, sizes[d],nperseg[d]))
            nperseg[d]=sizes[d]
            noverlap[d]=0
    
    steps = {d:nperseg[d] - noverlap[d] for d in dims} # step between each tile
    indices = {d:np.arange(0, sizes[d] - nperseg[d] + 1, steps[d]) for d in dims} # index of first point of each tile
    
    # For centering option:
    for d in dims:
        if centering[d]:
            ishift = {'left':1,'right':0}[side[d]]
            indices[d] = indices[d]+(sizes[d]-indices[d][-1]-nperseg[d]+ishift)//2

    tiles_index = {d:xr.DataArray(np.empty([len(indices[d]),nperseg[d]], dtype=int), dims = [d,'__'+d]) for d,ind in nperseg.items()}


    for d in dims:
        for i in range(tiles_index[d].sizes[d]):
            tiles_index[d][i] = np.arange(indices[d][i], indices[d][i]+nperseg[d])
    
    if isinstance(ds, xr.DataArray) or isinstance(ds, xr.Dataset):
        coords = {d:ds[d][indices[d]].data+nperseg[d]//2 for d in dims}
    else:
        coords = {d:indices[d]+nperseg[d]//2 for d in dims}
    
    tiles_index = {d:k.assign_coords({d:coords[d]}).rename({d:prefix+d}).rename('tile_{}_index'.format(d)) for d,k in tiles_index.items()}
    
    for d,v in tiles_index.items():
        v[prefix+d].attrs.update({'long_name': 'index of tile middle point','nperseg':nperseg[d], 'noverlap':noverlap[d]})
    
    return tiles_index

def get_corner_tile(tiles):
    """
    Extract corner indexes of tiles
    Args:
        tiles (dict of xarray) : xtiling()
    Return:
        (dict of xarray): same keys as tiles, values ares corners indexes only
    """
    corners = dict()
    for d,v in tiles.items():
        corners[d]=v[{'__'+d:[0,-1]}].rename({'__'+d:'corner_'+d})
    return corners

def get_middle_tile(tiles):
    """
    Extract middle indexes of tiles
    Args:
        tiles (dict of xarray) : xtiling()
    Return:
        (dict of xarray): same keys as tiles, values ares middle indexes
    """
    middle = dict()
    for d,v in tiles.items():
        middle[d]=v[{'__'+d:v.sizes['__'+d]//2}]
    return middle



