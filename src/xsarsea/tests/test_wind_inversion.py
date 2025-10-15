import pytest
import numpy as np
import xarray as xr
from unittest.mock import patch, MagicMock
from xsarsea.windspeed import windspeed,register_luts
import xsarsea
nc_luts_path = xsarsea.get_test_file('nc_luts_reduce')
path_cmod7 = xsarsea.get_test_file("cmod7_and_python_script")
register_luts(nc_luts_path, path_cmod7)

@pytest.fixture
def mock_dataset():
    """Crée un dataset xarray avec dsig_cr pour les tests cross-pol."""
    line = np.arange(10)
    sample = np.arange(10)
    incidence = np.ones((10, 10)) * 30
    sigma0_vv = np.ones((10, 10)) * 0.1
    sigma0_vh = np.ones((10, 10)) * 0.15

     # Module aléatoire entre 5 et 15, angle aléatoire entre 0 et 360°
    # module = np.random.uniform(5, 15, (10, 10))
    # angle = np.random.uniform(0, 2*np.pi, (10, 10))
    # ancillary_wind = module * np.exp(1j * angle)

    ancillary_wind = np.ones((10, 10), dtype=np.complex128) * (10 + 10j)

    # Création d'un dsig_cr réaliste : grands nombres, quelques NaN, et valeurs variées
    dsig_cr = np.ones((10, 10)) * 1e9  # Base : grands nombres
    dsig_cr[0, :3] = np.nan  # Quelques NaN
    dsig_cr[-1, -3:] = [1.5, 0.1, 0.7]  # Quelques valeurs plus petites
    dsig_cr[3, 3] = 4.5e16  # Valeur très grande

    ds = xr.Dataset(
        {
            "incidence": (["line", "sample"], incidence),
            "sigma0_ocean": (["line", "sample", "pol"], np.stack([sigma0_vv, sigma0_vh], axis=-1)),
            "ancillary_wind": (["line", "sample"], ancillary_wind),
            "dsig_cr": (["line", "sample"], dsig_cr),
        },
        coords={
            "line": line,
            "sample": sample,
            "pol": ["VV", "VH"],
        },
    )
    return ds

def test_invert_from_model_copol(mock_dataset):
    """Test inversion en mode copol (VV)."""
    with patch("xsarsea.windspeed.invert_from_model") as mock_invert:
        mock_invert.return_value = np.ones((10, 10)) * 5
        result = windspeed.invert_from_model(
            mock_dataset.incidence,
            mock_dataset.sigma0_ocean.isel(pol=0),
            ancillary_wind=mock_dataset["ancillary_wind"],

            model="cmod5n",
        )
        assert result.shape == (10, 10)
        assert np.allclose(result.real, 6.49960225)

def test_invert_from_model_crosspol(mock_dataset):
    """Test inversion en mode crosspol (VH)."""
    with patch("xsarsea.windspeed.invert_from_model") as mock_invert:
        GMF_VH_NAME = "gmf_s1_v2"
        mock_invert.return_value = np.ones((10, 10)) * 7
        result = windspeed.invert_from_model(
            mock_dataset.incidence,
            mock_dataset.sigma0_ocean.isel(pol=1),
            model=GMF_VH_NAME,
            # dsig_cr=mock_dataset['dsig_cr']
        )
        assert result.shape == (10, 10)
        assert np.allclose(result, 80)

def test_invert_from_model_dualpol(mock_dataset):
    """Test inversion en mode dualpol (VV + VH)."""
    with patch("xsarsea.windspeed.invert_from_model") as mock_invert:
        GMF_VV_NAME = "gmf_cmod5n"
        GMF_VH_NAME = "gmf_s1_v2"
        model=(GMF_VV_NAME,GMF_VH_NAME)
        mock_invert.return_value = (np.ones((10, 10)) * 6, np.ones((10, 10)) * 8)
        result_co, result_dual = windspeed.invert_from_model(
            mock_dataset.incidence,
            mock_dataset.sigma0_ocean.isel(pol=0),
            mock_dataset.sigma0_ocean.isel(pol=1),
            ancillary_wind=mock_dataset['ancillary_wind'],
            model=model,
        )
        assert result_co.shape == (10, 10)
        assert result_dual.shape == (10, 10)
        assert np.allclose(result_co.real, 6.49960225)
        assert np.allclose(result_dual.real, 48.14520185)



# if __name__ =='__main__':
#     import logging
#     logging.basicConfig(level=logging.INFO)
#     ds = mock_dataset()
#     logging.info('ds : %s',ds)
#     # test_invert_from_model_copol(ds)
#     # test_invert_from_model_crosspol(ds)
#     test_invert_from_model_dualpol(ds)