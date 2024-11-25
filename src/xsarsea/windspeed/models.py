# class that represent a model (lut or gmf)

import glob
import logging
import os
from abc import abstractmethod

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger('xsarsea.windspeed.models')


class Model:
    """
    Model class is an abstact class, for handling GMF or LUT.
    It should not be instanciated by the user.

    All registered models can be retreived with :func:`~xsarsea.windspeed.available_models`
    """

    _available_models = {}
    _name_prefix = ""
    _priority = None

    @abstractmethod
    def __init__(self, name, **kwargs):
        logger.debug('name model : %s',name)
        self.name = name
        self.pol = kwargs.pop("pol", None)
        self.units = kwargs.pop("units", None)
        self.phi_range = kwargs.pop("phi_range", None)
        self.wspd_range = kwargs.pop("wspd_range", None)
        self.__dict__.update(kwargs)
        self.resolution = kwargs.pop("resolution", None)

        if not hasattr(self, "inc_range"):
            # self.inc_range = [17., 50.]
            self.inc_range = [16.0, 66.0]
        # steps for generated luts
        self.inc_step_lr = kwargs.pop("inc_step_lr", 1.0)
        self.wspd_step_lr = kwargs.pop("wspd_step_lr", 0.2)
        self.phi_step_lr = kwargs.pop("phi_step_lr", 2.5)

        self.inc_step = kwargs.pop("inc_step", 0.1)
        self.wspd_step = kwargs.pop("wspd_step", 0.1)
        self.phi_step = kwargs.pop("phi_step", 1.0)

        self.__class__._available_models[name] = self

        logger.debug(
            f"register model {name}"
            + f" pol={self.pol}"
            + f" units={self.units}"
            + "\n"
            + f" inc_range={self.inc_range}"
            + f" wspd_range={self.wspd_range}"
            + f" phi_range={self.phi_range}"
            + "\n"
            + f" inc_step={self.inc_step}"
            + f" wspd_step={self.wspd_step}"
            + f" phi_step={self.phi_step}"
            + "\n"
            + f" inc_step_lr={self.inc_step_lr}"
            + f" wspd_step_lr={self.wspd_step_lr}"
            + f" phi_step_lr={self.phi_step_lr}"
        )

    @property
    @abstractmethod
    def short_name(self):
        if self.__class__._name_prefix and self.name.startswith(self.__class__._name_prefix):
            return self.name.replace(self.__class__._name_prefix, "", 1)
        else:
            return None

    @abstractmethod
    def _raw_lut(self):
        pass

    def _normalize_lut(self, lut, **kwargs):
        # check that lut is correct
        assert isinstance(lut, xr.DataArray)
        try:
            units = lut.attrs["units"]
        except KeyError:
            raise KeyError("lut has no lut.attrs['units']")

        allowed_units = ["linear", "dB"]
        if units not in allowed_units:
            raise ValueError(f"Unknown lut units '{units}'. Allowed are '{allowed_units}'")

        if lut.ndim == 2:
            good_dims = ("incidence", "wspd")
            if not (lut.dims == good_dims):
                raise IndexError(f"Bad dims '{lut.dims}'. Should be '{good_dims}'")
        elif lut.ndim == 3:
            good_dims = ("incidence", "wspd", "phi")
            if not (lut.dims == good_dims):
                raise IndexError(f"Bad dims '{lut.dims}'. Should be '{good_dims}'")
        else:
            raise IndexError(f"Bad dims '{lut.dims}'")

        assert "resolution" in lut.attrs

        # we check if the lut needs interpolation
        resolution = kwargs.pop("resolution", "high")

        if resolution is None:
            # high res by default
            resolution = "high"

        lut_resolution = lut.attrs["resolution"]
        logger.debug(f"lut_resolution {self.name} from _raw_lut : {lut_resolution}")
        logger.debug(f"desired {self.name} resolution : {resolution}")

        #  forcing resolution to be the one of
        if resolution == "high" and lut_resolution == "high":
            do_interp = self.inc_step != kwargs.get(
                "inc_step", self.inc_step
            ) or self.wspd_step != kwargs.get("wspd_step", self.wspd_step)
            if self.iscopol:
                do_interp = do_interp or self.phi_step != kwargs.get("phi_step", self.phi_step)

        elif resolution == "low" and lut_resolution == "low":
            do_interp = self.inc_step_lr != kwargs.get(
                "inc_step_lr", self.inc_step_lr
            ) or self.wspd_step_lr != kwargs.get("wspd_step_lr", self.wspd_step_lr)
            if self.iscopol:
                do_interp = do_interp or self.phi_step_lr != kwargs.get(
                    "phi_step_lr", self.phi_step_lr
                )
        else:
            do_interp = False

        if do_interp:
            logger.debug(
                f"Even if lut_resolution is already set to {lut_resolution}, lut {self.name} needs interpolation to match your desired resolution"
            )

        if (resolution is not None and resolution != lut_resolution) or do_interp:
            if resolution == "high":
                # high resolution steps
                inc_step = kwargs.pop("inc_step", self.inc_step)
                wspd_step = kwargs.pop("wspd_step", self.wspd_step)
                phi_step = kwargs.pop("phi_step", self.phi_step)
            elif resolution == "low":
                # low resolution steps
                inc_step = kwargs.pop("inc_step_lr", self.inc_step_lr)
                wspd_step = kwargs.pop("wspd_step_lr", self.wspd_step_lr)
                phi_step = kwargs.pop("phi_step_lr", self.phi_step_lr)

            inc, wspd, phi = (
                r and np.linspace(r[0], r[1], num=int(np.round((r[1] - r[0]) / step) + 1))
                for r, step in zip(
                    [self.inc_range, self.wspd_range, self.phi_range],
                    [inc_step, wspd_step, phi_step],
                )
            )
            logger.debug(f"interp lut {self.name} to high res")
            interp_kwargs = {
                k: v
                for k, v in zip(["incidence", "wspd", "phi"], [inc, wspd, phi])
                if v is not None
            }
            lut = lut.interp(**interp_kwargs, kwargs=dict(bounds_error=True))
            lut.attrs["resolution"] = resolution
        else:
            logger.debug(
                f"lut {self.name} already at desired resolution {resolution} with exact same params : no interpolation needed"
            )

        return lut

    @property
    def iscopol(self):
        """True if model is copol"""
        return len(set(self.pol)) == 1

    @property
    def iscrosspol(self):
        """True if model is crosspol"""
        return len(set(self.pol)) == 2

    def to_lut(self, units="linear", **kwargs):
        """
        Get the model lut

        Parameters
        ----------
        units: str
            'linear', 'dB'

        Returns
        -------
        xarray.DataArray

        """

        lut = self._raw_lut(**kwargs)

        lut = self._normalize_lut(lut, **kwargs)

        final_lut = lut

        if units is None:
            return final_lut

        if units == "dB":
            if lut.attrs["units"] == "dB":
                final_lut = lut
            elif lut.attrs["units"] == "linear":
                # clip with epsilon to avoid nans
                final_lut = 10 * np.log10(lut + 1e-15)
                final_lut.attrs["units"] = "dB"
        elif units == "linear":
            if lut.attrs["units"] == "linear":
                final_lut = lut
            elif lut.attrs["units"] == "dB":
                final_lut = 10.0 ** (lut / 10.0)
                final_lut.attrs["units"] = "linear"
        else:
            raise ValueError(f"Unit not known: {units}. Known are 'dB' or 'linear' ")

        final_lut.attrs["model"] = self.name
        final_lut.attrs["pol"] = self.pol
        final_lut.name = "sigma0_model"

        return final_lut

    def to_netcdf(self, file):
        """
        Save model as a lut netcdf file.

        Parameters
        ----------
        file: str
        """

        if self.iscopol:
            resolution = "low"
        else:
            resolution = "high"

        lut = self.to_lut(resolution=resolution, units="dB")
        ds_lut = lut.to_dataset(promote_attrs=True)
        ds_lut.sigma0_model.attrs.clear()
        ds_lut.attrs["pol"] = self.pol
        ds_lut.attrs["inc_range"] = self.inc_range
        ds_lut.attrs["wspd_range"] = self.wspd_range
        ds_lut.attrs["resolution"] = resolution
        ds_lut.attrs["model"] = self.short_name
        if "phi" in lut.dims:
            ds_lut.attrs["phi_range"] = self.phi_range

        ds_lut.attrs["wspd_step"] = np.round(np.unique(np.diff(lut.wspd)), decimals=2)[0]
        ds_lut.attrs["inc_step"] = np.round(np.unique(np.diff(lut.incidence)), decimals=2)[0]
        if "phi" in lut.dims:
            ds_lut.attrs["phi_step"] = np.round(np.unique(np.diff(lut.phi)), decimals=2)[0]

        ds_lut.to_netcdf(file)

    @abstractmethod
    def __call__(self, inc, wspd, phi=None, broadcast=False):
        """

        Parameters
        ----------
        inc: array-like
            incidence
        wspd: array-like
            windspeed
        phi: array-like or None
            wind direction relative to antenna
        broadcast: bool
            | If True, input arrays will be broadcasted to the same dimensions, and output will have the same dimension.
            | This option is only available for :func:`~xsarsea.windspeed.gmfs.GmfModel`

        Examples
        --------

        >>> cmod5 = xsarsea.windspeed.get_model("cmod5")
        >>> cmod5(np.arange(20, 22), np.arange(10, 12))
        <xarray.DataArray (incidence: 2, wspd: 2)>
        array([[0.00179606, 0.00207004],
        [0.0017344 , 0.00200004]])
        Coordinates:
        * incidence  (incidence) int64 20 21
        * wspd       (wspd) int64 10 11
        Attributes:
        units:    linear

        Returns
        -------
        xarray.DataArray
        """
        raise NotImplementedError(self.__class__)

    def __repr__(self):
        return f"<{self.__class__.__name__}('{self.name}') pol={self.pol}>"


class LutModel(Model):
    """
    Abstract class for handling Lut models. See :func:`~Model`

    Examples
    --------

    >>> isinstance(xsarsea.windspeed.get_model("sarwing_lut_cmodms1ahw"), LutModel)
    True
    """

    _name_prefix = "nc_lut_"
    _priority = None

    def __call__(self, inc, wspd, phi=None, units=None, **kwargs):

        all_scalar = all(np.isscalar(v) for v in [inc, wspd, phi] if v is not None)

        all_1d = False
        try:
            all_1d = all(v.ndim == 1 for v in [inc, wspd, phi] if v is not None)
        except AttributeError:
            all_1d = False

        if not (all_scalar or all_1d):
            raise NotImplementedError("Only scalar or 1D array are implemented for LutModel")

        lut = self.to_lut(units=units, **kwargs)
        if "phi" in lut.dims:
            sigma0 = lut.interp(incidence=inc, wspd=wspd, phi=phi)
        else:
            sigma0 = lut.interp(incidence=inc, wspd=wspd)

        try:
            sigma0.name = "sigma0_gmf"
            sigma0.attrs["model"] = self.name
            sigma0.attrs["units"] = self.units
        except AttributeError:
            pass

        if all_scalar:
            return sigma0.item()
        else:
            return sigma0


class NcLutModel(LutModel):
    """
    Class to handle luts in netcdf xsarsea format
    """

    _priority = 10

    @property
    def short_name(self):
        return self._short_name

    def __init__(self, path, **kwargs):
        name = os.path.splitext(os.path.basename(path))[0]
        # we do not want to read full dataset at this step, we just want global attributes
        # with netCDF4.Dataset(path) as ncfile:
        logger.debug('path model : %s',path)
        with xr.open_dataset(path) as ncfile:

            for attr in [
                "units",
                "pol",
                "model",
                "resolution",
                "inc_range",
                "wspd_range",
                "phi_range",
                "inc_step",
                "wspd_step",
                "phi_step",
            ]:
                if attr in ncfile.attrs:
                    try:

                        kwargs[attr] = ncfile.attrs[attr]
                        if isinstance(kwargs[attr], np.ndarray):
                            kwargs[attr] = list(kwargs[attr])
                    except AttributeError:
                        if "phi" not in attr:
                            raise AttributeError(f"Attr {attr} not found in {path}")
                else:
                    logger.debug('no %s attribute for model : %s',attr,name)
        self._short_name = kwargs.pop("model")
        if kwargs["resolution"] == "low":
            kwargs["inc_step_lr"] = kwargs.pop("inc_step")
            kwargs["wspd_step_lr"] = kwargs.pop("wspd_step")
            kwargs["phi_step_lr"] = kwargs.pop("phi_step", None)

        super().__init__(name, **kwargs)
        self.path = path

    def _raw_lut(self, **kwargs):
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)
        ds_lut = xr.open_dataset(self.path)

        lut = ds_lut.sigma0_model
        lut.attrs["units"] = ds_lut.attrs["units"]
        lut.attrs["model"] = ds_lut.attrs["model"]
        lut.attrs["resolution"] = ds_lut.attrs["resolution"]

        return lut


def register_nc_luts(topdir, gmf_names=None):
    """
    Register all netcdf luts found under `topdir`.

    This function return nothing. See `xsarsea.windspeed.available_models` to see registered models.

    Parameters
    ----------
    topdir: str
        top dir path to netcdf luts.

        gmf_names (list, optional): List of names to filter the registrated gmfs.
            If None, all registrated gmfs are processed.

    Examples
    --------
    register a subset of sarwing luts

    >>> xsarsea.windspeed.register_nc_luts(xsarsea.get_test_file("nc_luts_subset"))

    register all pickle lut from ifremer path

    >>> xsarsea.windspeed.register_pickle_luts(
    ...     "/home/datawork-cersat-public/cache/project/sarwing/xsardata/nc_luts"
    ... )


    See Also
    --------
    xsarsea.windspeed.available_models
    xsarsea.windspeed.gmfs.GmfModel.register

    """
    for path in glob.glob(os.path.join(topdir, f"{NcLutModel._name_prefix}*.nc")):
        path = os.path.abspath(os.path.join(topdir, path))
        name = os.path.basename(path).replace(".nc", "")
        if gmf_names is None or name in gmf_names:
            ncLutModel = NcLutModel(path)


def available_models(pol=None):
    """
    get available models

    Parameters
    ----------
    pol: str or None
        Filter models by pol (ie pol='VV')
        if None, no filters

    Returns
    -------
    pandas.DataFrame

    """

    avail_models = pd.DataFrame(columns=["short_name", "priority", "pol", "model"], index=[])

    for name, model in Model._available_models.items():
        avail_models.loc[name, "model"] = model
        avail_models.loc[name, "pol"] = model.pol
        avail_models.loc[name, "priority"] = model._priority
        avail_models.loc[name, "short_name"] = model.short_name

    aliased = (
        avail_models.sort_values("priority", ascending=True)
        .drop_duplicates("short_name")
        .rename(columns=dict(short_name="alias"))
        .drop(columns="priority")
    )

    non_aliased = (
        avail_models.drop(aliased.index)
        .drop(columns="priority")
        .rename(columns=dict(short_name="alias"))
    )
    non_aliased["alias"] = None
    # aliased.merge(non_aliased)
    # aliased
    avail_models = pd.concat([aliased, non_aliased])

    # filter
    if pol is not None:
        avail_models = avail_models[avail_models.pol == pol]

    return avail_models

    if pol is None:
        return Model._available_models
    else:
        models_found = {}
        for name, model in Model._available_models.items():
            if pol == model.pol:
                models_found[name] = model
        return models_found


def get_model(name):
    """
    get model by name

    Parameters
    ----------
    name: str
        model name

    Returns
    -------
    Model
    """

    if isinstance(name, Model):
        # name is already a model
        return name

    avail_models = available_models()

    try:
        model = avail_models.loc[name].model
    except KeyError:
        try:
            model = avail_models[avail_models.alias == name].model.item()
        except ValueError:
            raise KeyError(f"model {name} not found")

    return model


def register_luts(topdir=None, topdir_cmod7=None):
    """
    Register gmfModel luts and ncLutModel luts

    Parameters
    ----------
    topdir: str
        top dir path to nc luts.

    topdir_cmod7: str
        top dir path to cmod7 luts.

    kwargs: dict
        kwargs to pass to register_nc_luts
    """

    # register gmf luts
    import xsarsea.windspeed as windspeed

    windspeed.GmfModel.activate_gmfs_impl()

    # register nc luts
    if topdir is not None:
        register_nc_luts(topdir)

    # register cmod7
    if topdir_cmod7 is not None:
        windspeed.register_cmod7(topdir_cmod7)
