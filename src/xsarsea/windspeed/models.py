# class that represent a model (lut or gmf)

from abc import abstractmethod
import numpy as np
import xarray as xr
import os
import glob
import netCDF4
import logging
import pandas as pd

logger = logging.getLogger('xsarsea.windspeed')


class Model:
    """
    Model class is an abstact class, for handling GMF or LUT.
    It should not be instanciated by the user.

    All registered models can be retreived with :func:`~xsarsea.windspeed.available_models`
    """

    _available_models = {}
    _name_prefix = ''
    _priority = None

    @abstractmethod
    def __init__(self, name, **kwargs):
        self.name = name
        self.pol = kwargs.pop('pol', None)
        self.units = kwargs.pop('units', None)
        self.phi_range = kwargs.pop('phi_range', None)
        self.wspd_range = kwargs.pop('wspd_range', None)
        self.__dict__.update(kwargs)
        if not hasattr(self, 'inc_range'):
            self.inc_range = [17., 50.]

        # steps for generated luts
        self.inc_step_lr = kwargs.pop('inc_step_lr', 1.)
        self.wspd_step_lr = kwargs.pop('wspd_step_lr', 0.4)
        self.phi_step_lr = kwargs.pop('phi_step_lr', 2.5)

        self.inc_step = kwargs.pop('inc_step', 0.1)
        self.wspd_step = kwargs.pop('wspd_step', 0.1)
        self.phi_step = kwargs.pop('phi_step', 1)

        self.__class__._available_models[name] = self

    @property
    @abstractmethod
    def short_name(self):
        if self.__class__._name_prefix and self.name.startswith(self.__class__._name_prefix):
            return self.name.replace(self.__class__._name_prefix, '', 1)
        else:
            return None

    @abstractmethod
    def _raw_lut(self):
        pass

    def _normalize_lut(self, lut, **kwargs):
        # check that lut is correct
        assert isinstance(lut, xr.DataArray)
        try:
            units = lut.attrs['units']
        except KeyError:
            raise KeyError("lut has no lut.attrs['units']")

        allowed_units = ['linear', 'dB']
        if units not in allowed_units:
            raise ValueError(
                "Unknown lut units '%s'. Allowed are '%s'" % (units, allowed_units))

        if lut.ndim == 2:
            good_dims = ('incidence', 'wspd')
            if not (lut.dims == good_dims):
                raise IndexError("Bad dims '%s'. Should be '%s'" %
                                 (lut.dims, good_dims))
        elif lut.ndim == 3:
            good_dims = ('incidence', 'wspd', 'phi')
            if not (lut.dims == good_dims):
                raise IndexError("Bad dims '%s'. Should be '%s'" %
                                 (lut.dims, good_dims))
        else:
            raise IndexError("Bad dims '%s'" % lut.dims)

        assert 'resolution' in lut.attrs

        # we check if the lut needs interpolation
        resolution = kwargs.pop('resolution', 'high')
        if resolution is None:
            # high res by default
            resolution = 'high'

        lut_resolution = lut.attrs['resolution']
        if resolution is not None and resolution != lut_resolution:
            if resolution == 'high':
                # high resolution steps
                inc_step = kwargs.pop('inc_step', self.inc_step)
                wspd_step = kwargs.pop('wspd_step', self.wspd_step)
                phi_step = kwargs.pop('phi_step', self.phi_step)
            elif resolution == 'low':
                # low resolution steps
                inc_step = kwargs.pop('inc_step_lr', self.inc_step_lr)
                wspd_step = kwargs.pop('wspd_step_lr', self.wspd_step_lr)
                phi_step = kwargs.pop('phi_step_lr', self.phi_step_lr)

            inc, wspd, phi = [
                r and np.linspace(r[0], r[1], num=int(
                    np.round((r[1] - r[0]) / step) + 1))
                for r, step in zip(
                    [self.inc_range, self.wspd_range, self.phi_range],
                    [inc_step, wspd_step, phi_step]
                )
            ]
            logger.debug('interp lut %s to high res' % self.name)
            interp_kwargs = {k: v for k, v in zip(['incidence', 'wspd', 'phi'], [
                                                  inc, wspd, phi]) if v is not None}
            lut = lut.interp(**interp_kwargs, kwargs=dict(bounds_error=True))
            lut.attrs['resolution'] = resolution

        return lut

    @property
    def iscopol(self):
        """True if model is copol"""
        return len(set(self.pol)) == 1

    @property
    def iscrosspol(self):
        """True if model is crosspol"""
        return len(set(self.pol)) == 2

    def to_lut(self, units='linear', **kwargs):
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

        if units == 'dB':
            if lut.attrs['units'] == 'dB':
                final_lut = lut
            elif lut.attrs['units'] == 'linear':
                # clip with epsilon to avoid nans
                final_lut = 10 * np.log10(lut + 1e-15)
                final_lut.attrs['units'] = 'dB'
        elif units == 'linear':
            if lut.attrs['units'] == 'linear':
                final_lut = lut
            elif lut.attrs['units'] == 'dB':
                final_lut = 10. ** (lut / 10.)
                final_lut.attrs['units'] = 'linear'
        else:
            raise ValueError(
                "Unit not known: %s. Known are 'dB' or 'linear' " % units)

        final_lut.attrs['model'] = self.name
        final_lut.attrs['pol'] = self.pol
        final_lut.name = 'sigma0_model'

        return final_lut

    def to_netcdf(self, file):
        """
        Save model as a lut netcdf file.

        Parameters
        ----------
        file: str
        """

        if self.iscopol:
            resolution = 'low'
        else:
            resolution = 'high'
        lut = self.to_lut(resolution=resolution, units='dB')
        ds_lut = lut.to_dataset(promote_attrs=True)
        ds_lut.sigma0_model.attrs.clear()
        ds_lut.attrs['pol'] = self.pol
        ds_lut.attrs['inc_range'] = self.inc_range
        ds_lut.attrs['wspd_range'] = self.wspd_range
        ds_lut.attrs['resolution'] = resolution
        ds_lut.attrs['model'] = self.short_name
        if 'phi' in lut.dims:
            ds_lut.attrs['phi_range'] = self.phi_range

        ds_lut.attrs['wspd_step'] = np.round(
            np.unique(np.diff(lut.wspd)), decimals=2)[0]
        ds_lut.attrs['inc_step'] = np.round(
            np.unique(np.diff(lut.incidence)), decimals=2)[0]
        if 'phi' in lut.dims:
            ds_lut.attrs['phi_step'] = np.round(
                np.unique(np.diff(lut.phi)), decimals=2)[0]

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
            wind direction, in **gmf convention**
        broadcast: bool
            | If True, input arrays will be broadcasted to the same dimensions, and output will have the same dimension.
            | This option is only available for :func:`~xsarsea.windspeed.gmfs.GmfModel`

        Examples
        --------

        >>> cmod5 = xsarsea.windspeed.get_model('cmod5')
        >>> cmod5(np.arange(20,22), np.arange(10,12))
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
        return "<%s('%s') pol=%s>" % (self.__class__.__name__, self.name, self.pol)


class LutModel(Model):
    """
    Abstract class for handling Lut models. See :func:`~Model`

    Examples
    --------

    >>> isinstance(xsarsea.windspeed.get_model('sarwing_lut_cmodms1ahw'), LutModel)
    True
    """

    _name_prefix = 'nc_lut_'
    _priority = None

    def __call__(self, inc, wspd, phi=None, units=None, **kwargs):

        all_scalar = all(np.isscalar(v)
                         for v in [inc, wspd, phi] if v is not None)

        all_1d = False
        try:
            all_1d = all(v.ndim == 1 for v in [
                         inc, wspd, phi] if v is not None)
        except AttributeError:
            all_1d = False

        if not (all_scalar or all_1d):
            raise NotImplementedError(
                'Only scalar or 1D array are implemented for LutModel')

        lut = self.to_lut(units=units, **kwargs)
        if 'phi' in lut.dims:
            sigma0 = lut.interp(incidence=inc, wspd=wspd, phi=phi)
        else:
            sigma0 = lut.interp(incidence=inc, wspd=wspd)

        try:
            sigma0.name = 'sigma0_gmf'
            sigma0.attrs['model'] = self.name
            sigma0.attrs['units'] = self.units
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
        with netCDF4.Dataset(path) as ncfile:
            for attr in ['units', 'pol', 'model', 'resolution', 'inc_range', 'wspd_range', 'phi_range', 'inc_step', 'wspd_step', 'phi_step']:
                try:
                    kwargs[attr] = ncfile.getncattr(attr)
                    if isinstance(kwargs[attr], np.ndarray):
                        kwargs[attr] = list(kwargs[attr])
                except AttributeError as e:
                    if 'phi' not in attr:
                        raise AttributeError(
                            'Attr %s not found in %s' % (attr, path))
        self._short_name = kwargs.pop('model')
        if kwargs['resolution'] == 'low':
            kwargs['inc_step_lr'] = kwargs.pop('inc_step')
            kwargs['wspd_step_lr'] = kwargs.pop('wspd_step')
            kwargs['phi_step_lr'] = kwargs.pop('phi_step', None)

        super().__init__(name, **kwargs)
        self.path = path

    def _raw_lut(self, **kwargs):
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)
        ds_lut = xr.open_dataset(self.path)

        lut = ds_lut.sigma0_model
        lut.attrs['units'] = ds_lut.attrs['units']
        lut.attrs['model'] = ds_lut.attrs['model']
        lut.attrs['resolution'] = ds_lut.attrs['resolution']

        return lut


def register_all_nc_luts(topdir):
    """
    Register all netcdf luts found under `topdir`.

    This function return nothing. See `xsarsea.windspeed.available_models` to see registered models.

    Parameters
    ----------
    topdir: str
        top dir path to netcdf luts.

    Examples
    --------
    register a subset of sarwing luts

    >>> xsarsea.windspeed.register_all_nc_luts(xsarsea.get_test_file('nc_luts_subset'))

    register all sarwing lut from ifremer path

    >>> xsarsea.windspeed.register_all_sarwing_luts('/home/datawork-cersat-public/cache/project/sarwing/xsardata/nc_luts')

    Notes
    _____
    Sarwing lut can be downloaded from https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsardata/nc_luts

    See Also
    --------
    xsarsea.windspeed.available_models
    xsarsea.windspeed.gmfs.GmfModel.register

    """
    for path in glob.glob(os.path.join(topdir, "%s*.nc" % NcLutModel._name_prefix)):
        path = os.path.abspath(os.path.join(topdir, path))

        sarwing_model = NcLutModel(path)


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

    avail_models = pd.DataFrame(
        columns=['short_name', 'priority', 'pol', 'model'], index=[])

    for name, model in Model._available_models.items():
        # print('%s: %s' % (name, model))
        avail_models.loc[name, 'model'] = model
        avail_models.loc[name, 'pol'] = model.pol
        avail_models.loc[name, 'priority'] = model._priority
        avail_models.loc[name, 'short_name'] = model.short_name

    aliased = avail_models.sort_values('priority', ascending=False).drop_duplicates('short_name').rename(
        columns=dict(short_name='alias')).drop(columns='priority')

    non_aliased = avail_models.drop(aliased.index).drop(
        columns='priority').rename(columns=dict(short_name='alias'))
    non_aliased['alias'] = None
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
            raise KeyError('model %s not found' % name)

    return model
