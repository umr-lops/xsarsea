import os

import numpy as np
import xarray as xr

from xsarsea.windspeed.models import LutModel
from xsarsea.windspeed.utils import logger


class Cmod7Model(LutModel):

    _name_prefix = "gmf_"
    _priority = 1

    def __init__(self, name, path, **kwargs):
        super().__init__(name, **kwargs)
        self.path = path

    def _raw_lut(self, **kwargs):
        if not os.path.isdir(self.path):
            raise FileNotFoundError(self.path)

        logger.info(f"load gmf lut from {self.path}")

        sigma0_path = os.path.join(self.path, "gmf_cmod7_vv.dat_little_endian")
        try:
            sigma0 = np.fromfile(sigma0_path, dtype=np.float32)
        except FileNotFoundError:
            raise FileNotFoundError(sigma0_path)

        # Dimensions of GMF table
        m = 250  # wind speed min/max = 0.2-50 (step 0.2) [m/s] --> 250 pts
        n = 73  # dir min/max = 0-180 (step 2.5) [deg]   -->  73 pts
        p = 51  # inc min/max = 16-66 (step 1) [deg]     -->  51 pts
        # Remove head and tail
        sigma0 = sigma0[1:-1]

        # To access the table as a three-dimensional Fortran-ordered m x n x p matrix,
        # reshape it
        sigma0 = sigma0.reshape((m, n, p), order="F")

        self.wspd_step_lr = 0.2
        self.inc_step_lr = 1
        self.phi_step_lr = 2.5

        self.inc_range_lr = [16, 66]
        self.wspd_range_lr = [0.2, 50.0]
        self.phi_range_lr = [0, 180]

        wspd = np.arange(
            self.wspd_range_lr[0], self.wspd_range_lr[1] + self.wspd_step_lr, self.wspd_step_lr
        )
        inc = np.arange(
            self.inc_range_lr[0], self.inc_range_lr[1] + self.inc_step_lr, self.inc_step_lr
        )
        phi = np.arange(
            self.phi_range_lr[0], self.phi_range_lr[1] + self.phi_step_lr, self.phi_step_lr
        )

        dims = ["wspd", "phi", "incidence"]
        final_dims = ["incidence", "wspd", "phi"]
        coords = {"incidence": inc, "phi": phi, "wspd": wspd}

        self.wspd_range = self.wspd_range_lr
        self.inc_range = self.inc_range_lr
        self.phi_range = self.phi_range_lr

        da_sigma0_db = xr.DataArray((sigma0), dims=dims, coords=coords)

        da_sigma0_db.name = "sigma0_gmf"
        da_sigma0_db.attrs["units"] = "linear"
        da_sigma0_db.attrs["model"] = self.name
        da_sigma0_db.attrs["resolution"] = "low"

        return da_sigma0_db.transpose(*final_dims)


def register_cmod7(topdir):
    """
    Register cmod7.

    This function return nothing. See `xsarsea.windspeed.available_models` to see registered models.

    Parameters
    ----------
    topdir: str
        top dir path to cmod7 lut.

    Examples
    --------

    Notes
    _____
    Source : https://scatterometer.knmi.nl/cmod7

    See Also
    --------
    xsarsea.windspeed.available_models
    xsarsea.windspeed.gmfs.GmfModel.register

    """

    path = topdir
    name = Cmod7Model._name_prefix + "cmod7"

    cmod7_model = Cmod7Model(name, path, pol="VV")
