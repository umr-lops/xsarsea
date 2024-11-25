import xsarsea
from xsarsea import windspeed
import numpy as np
import xarray as xr
import dask.array as da


@windspeed.gmfs.GmfModel.register(inc_range=[17., 50.], wspd_range=[3., 80.], pol='VH', units='linear', defer=False)
def gmf_dummy(inc, wspd, phi=None):
    a0 = 0.00013106836021008122
    a1 = -4.530598283705591e-06
    a2 = 4.429277425062766e-08
    b0 = 1.3925444179360706
    b1 = 0.004157838450541205
    b2 = 3.4735809771069953e-05

    a = a0 + a1 * inc + a2 * inc ** 2
    b = b0 + b1 * inc + b2 * inc ** 2
    sig = a * wspd ** b

    return sig


def test_available_models():
    models = windspeed.available_models().index
    assert 'gmf_cmod5n' in models
    assert 'gmf_dummy' in models

    nc_luts_subset_path = xsarsea.utils.get_test_file(
        'nc_luts_reduce')
    windspeed.register_nc_luts(nc_luts_subset_path)
    # windspeed.register_nc_luts(nc_luts_path)
    assert 'nc_lut_cmodms1ahw' in windspeed.available_models().index

    nc_luts_path = xsarsea.utils.get_test_file('xsarsea_luts')
    windspeed.models.register_nc_luts(nc_luts_path)
    assert 'nc_lut_sarwing_lut_cmod5n' in windspeed.available_models().index

    assert 'nc_lut_cmodms1ahw' in windspeed.available_models().index


def test_models():
    for model_name, model_row in windspeed.available_models().iterrows():
        model = model_row.model
        print('checking model %s' % model_name)
        lut = model.to_lut()

        # scalar check
        inc = 35
        wspd = 15
        phi = 90
        res = model(inc, wspd, phi)
        assert np.isscalar(res)

        # numpy check
        inc = np.array([35, 40])
        wspd = np.array([15, 17, 20])
        phi = np.array([0., 45., 90., 135., 180.])
        res = model(inc, wspd, phi)

        try:
            # 2D check
            # numpy check
            inc = np.arange(21).reshape(7, 3) + 20
            wspd = np.arange(21).reshape(7, 3)
            phi = np.arange(21).reshape(7, 3) * 9
            res = model(inc, wspd, phi)

            # dask check
            da_inc, da_wspd, da_phi = [
                da.from_array(v) for v in [inc, wspd, phi]]
            xr_inc = xr.DataArray(da_inc, dims=['line', 'sample'])
            xr_wspd = xr.DataArray(da_wspd, dims=['line', 'sample'])
            xr_phi = xr.DataArray(da_phi, dims=['line', 'sample'])
            res = model(xr_inc, xr_wspd, xr_phi)
            res.compute()
        except NotImplementedError:
            pass


def test_inversion():
    sarwing_owi_file = xsarsea.get_test_file(
        's1a-iw-owi-xx-20210909t130650-20210909t130715-039605-04AE83.nc')
    sarwing_ds = xsarsea.read_sarwing_owi(sarwing_owi_file).isel(
        line=slice(0, 50), sample=slice(0, 60))

    owi_ecmwf_wind = sarwing_ds.owiEcmwfWindSpeed * np.exp(
        1j * xsarsea.dir_meteo_to_sample(sarwing_ds.owiEcmwfWindDirection, sarwing_ds.owiHeading))
    sarwing_ds = xr.merge([
        sarwing_ds,
        owi_ecmwf_wind.to_dataset(name='owi_ancillary_wind'),
    ])

    nc_luts_subset_path = xsarsea.get_test_file('nc_luts_reduce')
    windspeed.pickle_luts.register_pickle_luts(nc_luts_subset_path)

    nesz_cross_flat = windspeed.nesz_flattening(
        sarwing_ds.owiNesz_cross, sarwing_ds.owiIncidenceAngle)
    dsig_cr = (1.25 / (sarwing_ds.owiNrcs_cross / nesz_cross_flat)) ** 4.

    windspeed_co, windspeed_dual = windspeed.invert_from_model(
        sarwing_ds.owiIncidenceAngle,
        sarwing_ds.owiNrcs,
        sarwing_ds.owiNrcs_cross,
        ancillary_wind=sarwing_ds.owi_ancillary_wind,
        dsig_cr=dsig_cr,
        model=('gmf_cmod5n', 'nc_lut_cmodms1ahw'))

    assert isinstance(windspeed_co, xr.DataArray)
    assert isinstance(windspeed_dual, xr.DataArray)

    # pure numpy
    windspeed_co, windspeed_dual = windspeed.invert_from_model(
        np.asarray(sarwing_ds.owiIncidenceAngle),
        np.asarray(sarwing_ds.owiNrcs),
        np.asarray(sarwing_ds.owiNrcs_cross),
        ancillary_wind=np.asarray(sarwing_ds.owi_ancillary_wind),
        dsig_cr=np.asarray(dsig_cr),
        model=('gmf_cmod5n', 'nc_lut_cmodms1ahw'))

    assert isinstance(windspeed_co, np.ndarray)
    assert isinstance(windspeed_dual, np.ndarray)

    # dask
    for v in ['owiIncidenceAngle', 'owiNrcs', 'owiNrcs_cross', 'owi_ancillary_wind']:
        sarwing_ds[v].data = da.from_array(sarwing_ds[v].data)

    windspeed_co, windspeed_dual = windspeed.invert_from_model(
        sarwing_ds.owiIncidenceAngle,
        sarwing_ds.owiNrcs,
        sarwing_ds.owiNrcs_cross,
        ancillary_wind=sarwing_ds.owi_ancillary_wind,
        dsig_cr=dsig_cr,
        model=('gmf_cmod5n', 'nc_lut_cmodms1ahw'))

    assert isinstance(windspeed_co.data, da.Array)
    assert isinstance(windspeed_dual.data, da.Array)

    windspeed_co = windspeed_co.compute()
    windspeed_dual = windspeed_dual.compute()

    assert isinstance(windspeed_co.data, np.ndarray)
    assert isinstance(windspeed_dual.data, np.ndarray)

