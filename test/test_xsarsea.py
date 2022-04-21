import xsarsea
from xsarsea import windspeed
import numpy as np

@windspeed.gmfs.GmfModel.register(inc_range=[17., 50.], wspd_range=[3., 80.], pols=['VH'], units='linear')
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
    models = windspeed.available_models()
    assert 'gmf_cmod5n' in models
    assert 'gmf_dummy' in models

    sarwing_luts_subset_path = xsarsea.utils.get_test_file('sarwing_luts_subset')
    windspeed.sarwing_luts.register_all_sarwing_luts(sarwing_luts_subset_path)

    assert 'sarwing_lut_cmodms1ahw' in models


def test_models():
    for model_name, model in windspeed.available_models().items():
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