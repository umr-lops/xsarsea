# CONSTANTS #
from xsarsea.gmfs import Gmfs_Loader
import numpy as np

du = 2
dv = 2
dsig_co = 0.1
dsig_cr = 0.1
du10_fg = 2


gmfs_loader = Gmfs_Loader()
dims = {}
dims["inc_1d"] = np.arange(17, 50.1, 0.1)
dims["phi_1d"] = np.arange(0, 360, 1)
dims["wspd_1d"] = np.arange(0.3, 50.1, 0.1)
dims["fct_number"] = 1

gmfs_loader.load_lut(pol="copol", tabulated=True,
                     path="/home/vincelhx/Documents/ifremer/data/gmfs/lut_co.nc", dims=dims)
gmfs_loader.load_lut(pol="crpol", tabulated=True,
                     path="/home/vincelhx/Documents/ifremer/data/gmfs/lut_cr.nc", dims=dims)


# "point_by_point"
# "iterative"
# "third"

inversion_parameters = {
    "inversion_method": "third",
    "lut_co_dict": gmfs_loader.lut_co_dict,
    "lut_cr_dict": gmfs_loader.lut_cr_dict,
    "dims": dims,
    "is_rs2": False,
}
