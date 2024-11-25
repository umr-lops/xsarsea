import os
import pickle as pkl

import numpy as np
import xarray as xr

from xsarsea.windspeed.models import LutModel
from xsarsea.windspeed.utils import logger


class PickleLutModel(LutModel):

    _name_prefix = "sarwing_lut__"
    _priority = 10

    def __init__(self, name, path, **kwargs):
        super().__init__(name, **kwargs)
        self.path = path

    def _raw_lut(self, **kwargs):
        if not os.path.isdir(self.path):
            raise FileNotFoundError(self.path)

        logger.info(f"load pickle lut from {self.path}")

        sigma0_db_path = os.path.join(self.path, "sigma.npy")
        sigma0_db = np.ascontiguousarray(np.transpose(np.load(sigma0_db_path)))
        inc = pkl.load(
            open(os.path.join(self.path, "incidence_angle.pkl"), "rb"), encoding="iso-8859-1"
        )
        try:
            phi, wspd = pkl.load(
                open(os.path.join(self.path, "wind_speed_and_direction.pkl"), "rb"),
                encoding="iso-8859-1",
            )
        except FileNotFoundError:
            phi = None
            wspd = pkl.load(
                open(os.path.join(self.path, "wind_speed.pkl"), "rb"), encoding="iso-8859-1"
            )

        self.wspd_step = np.round(np.unique(np.diff(wspd)), decimals=2)[0]
        self.inc_step = np.round(np.unique(np.diff(inc)), decimals=2)[0]
        self.inc_range = [np.round(np.min(inc), decimals=2), np.round(np.max(inc), decimals=2)]
        self.wspd_range = [np.round(np.min(wspd), decimals=2), np.round(np.max(wspd), decimals=2)]

        if phi is not None:
            dims = ["wspd", "phi", "incidence"]
            final_dims = ["incidence", "wspd", "phi"]
            coords = {"incidence": inc, "phi": phi, "wspd": wspd}
            self.phi_step = np.round(np.unique(np.diff(phi)), decimals=2)[0]
            # low res parameters, for downsampling
            self.inc_step_lr = 1.0
            self.wspd_step_lr = 0.4
            self.phi_step_lr = 2.5
            self.phi_range = [np.round(np.min(phi), decimals=2), np.round(np.max(phi), decimals=2)]
        else:
            dims = ["wspd", "incidence"]
            final_dims = ["incidence", "wspd"]
            coords = {"incidence": inc, "wspd": wspd}
            # low res parameters, for downsampling. those a close to high res, as crosspol lut has quite small
            self.inc_step_lr = 1.0
            self.wspd_step_lr = 0.1
            self.phi_step_lr = 1

        da_sigma0_db = xr.DataArray(sigma0_db, dims=dims, coords=coords)

        da_sigma0_db.name = "sigma0_gmf"
        da_sigma0_db.attrs["units"] = "dB"
        da_sigma0_db.attrs["model"] = self.name
        da_sigma0_db.attrs["resolution"] = "high"

        return da_sigma0_db.transpose(*final_dims)


def register_pickle_luts(path):
    """
    Register LUTs from a specified path. The path can be a directory containing multiple LUTs or a single LUT file.

    This function returns nothing. See `xsarsea.windspeed.available_models` to see registered models.

    Parameters
    ----------
    path : str
        Path to a LUT or a directory containing multiple LUTs.

    Examples
    --------
    Register a single LUT:

    >>> xsarsea.windspeed.register_pickle_luts(
    ...     xsarsea.get_test_file("sarwing_luts_subset/GMF_cmodms1ahw")
    ... )

    Register all LUTs from a directory:

    >>> xsarsea.windspeed.register_pickle_luts(
    ...     "/home/datawork-cersat-public/cache/project/sarwing/GMFS/v1.6"
    ... )

    Notes
    -----
    LUTs can be downloaded from https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsardata/sarwing_luts

    See Also
    --------
    xsarsea.windspeed.available_models
    xsarsea.windspeed.gmfs.GmfModel.register
    """

    def register_lut(file_path):
        name = os.path.basename(file_path)
        name = name.replace("GMF_", PickleLutModel._name_prefix)
        # Guess available pols from filenames
        if os.path.exists(os.path.join(file_path, "wind_speed_and_direction.pkl")):
            pol = "VV"
        elif os.path.exists(os.path.join(file_path, "wind_speed.pkl")):
            pol = "VH"
        else:
            pol = None

        pickleLutmodel = PickleLutModel(name, file_path, pol=pol)

    last_folder_name = os.path.basename(os.path.normpath(path))
    if last_folder_name.startswith("GMF_"):
        register_lut(path)
    else:
        if os.path.isdir(path):
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                # Handle directories within the top directory
                if os.path.isdir(file_path) and filename.startswith("GMF_"):
                    register_lut(file_path)
