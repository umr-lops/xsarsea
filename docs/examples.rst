.. _examples:

#############
Examples
#############

Sigma0_detrend example
======================

Open a SAFE file sentinel 1 and calculate the sigma0_detrend", on pourrait mettre "The Normalized Radar Cross Section (sigma0) as computed from Level-1 SAR data can be detrended in the case of ocean scenes. The goal is to remove the averaged trend (decreasing) of the NRCS with (increasing) incidence angle observed for acquisitions over ocean. The detrend maximizes the  contrasts in the image due to geophysical phenomena and improves the visualization experience of ocean scenes. sigma0_detrend is also  termed "image roughness" or "nice display".

.. code-block:: python

    import xsar
    import xsarsea
    safe_file = 'S1A_IW_GRDH_1SDV_20170907T103020_20170907T103045_018268_01EB76_992F.SAFE'
    ds = xsar.open_dataset(safe_file)
    sar_ds = ds[0]
    sar_ds['sigma0_detrend'] = xsarsea.sigma0_detrend(sar_ds.sigma0, sar_ds.incidence)

Result:

.. code-block:: text

 <xarray.Dataset>
 Dimensions:         (atrack: 16800, pol: 2, xtrack: 25200)
 Coordinates:
  * pol             (pol) object 'VV' 'VH'
  * atrack          (atrack) int64 0 1 2 3 4 5 ... 16795 16796 16797 16798 16799
  * xtrack          (xtrack) int64 0 1 2 3 4 5 ... 25195 25196 25197 25198 25199
 Data variables:
    digital_number  (pol, atrack, xtrack) float32 dask.array<chunksize=(1, 5600, 25200), meta=np.ndarray>
    time            (atrack) datetime64[ns] 2017-09-07T10:30:20.936409 ... 20...
    longitude       (atrack, xtrack) float32 dask.array<chunksize=(5600, 25200), meta=np.ndarray>
    latitude        (atrack, xtrack) float32 dask.array<chunksize=(5600, 25200), meta=np.ndarray>
    incidence       (atrack, xtrack) float32 dask.array<chunksize=(5600, 25200), meta=np.ndarray>
    elevation       (atrack, xtrack) float32 dask.array<chunksize=(5600, 25200), meta=np.ndarray>
    sigma0_raw      (pol, atrack, xtrack) float32 dask.array<chunksize=(1, 5600, 25200), meta=np.ndarray>
    nesz            (pol, atrack, xtrack) float32 dask.array<chunksize=(1, 5600, 25200), meta=np.ndarray>
    gamma0_raw      (pol, atrack, xtrack) float32 dask.array<chunksize=(1, 5600, 25200), meta=np.ndarray>
    negz            (pol, atrack, xtrack) float32 dask.array<chunksize=(1, 5600, 25200), meta=np.ndarray>
    sigma0          (pol, atrack, xtrack) float32 dask.array<chunksize=(1, 5600, 25200), meta=np.ndarray>
    gamma0          (pol, atrack, xtrack) float32 dask.array<chunksize=(1, 5600, 25200), meta=np.ndarray>
    sigma0_detrend  (pol, atrack, xtrack) float32 dask.array<chunksize=(1, 5600, 25200), meta=np.ndarray>
 Attributes:
    footprint:       POLYGON ((-67.84221143971432 20.72564283093837, -70.2216...
    coverage:        251km * 170km (xtrack * atrack )
    pixel_xtrack_m:  10.0
    pixel_atrack_m:  10.2
    ipf_version:     2.84
    swath_type:      IW
    polarizations:   VV VH
    product_type:    GRD
    mission:         SENTINEL-1
    satellite:       A
    start_date:      2017-09-07 10:30:20.936409
    stop_date:       2017-09-07 10:30:45.935264
    path:            /home/datawork-cersat-public/project/mpc-sentinel1/data/...
    denoised:        {'VH': False, 'VV': False}
    subdataset:      IW
    geometry:        {'VH': {'atrack':   atracks xtracks                     ...
    Conventions:     CF-1.7


Streaks example
======================

Use the sigma0_detrend

.. code-block:: python

    from xsarsea.streaks import streaks_direction
    streaks_direction(sar_ds.sigma0_detrend)

Result:

.. code-block:: text

    <xarray.DataArray 'angles_hist' (pol: 2, atrack: 105, xtrack: 158)>
    dask.array<concatenate, shape=(2, 105, 158), dtype=float64, chunksize=(1, 35, 158), chunktype=numpy.ndarray>
    Coordinates:
      * atrack   (atrack) float64 21.5 181.5 341.5 ... 1.634e+04 1.65e+04 1.666e+04
      * xtrack   (xtrack) float64 21.5 181.5 341.5 ... 2.482e+04 2.498e+04 2.514e+04
      * pol      (pol) object 'VV' 'VH'
