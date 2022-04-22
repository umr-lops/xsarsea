# class that represent a model (lut or gmf)

from abc import abstractmethod
import numpy as np
import xarray as xr


class Model:

    available_models = {}
    """dict of available registered models"""

    @abstractmethod
    def __init__(self, name, **kwargs):
        self.name = name
        self.pol = kwargs.pop('pol', None)
        self.units = kwargs.pop('units', None)
        self.inc_range = kwargs.pop('inc_range', None)
        self.phi_range = kwargs.pop('phi_range', None)
        self.wspd_range = kwargs.pop('wspd_range', None)
        self.__class__.available_models[name] = self

    @abstractmethod
    def _raw_lut(self):
        pass

    @staticmethod
    def _check_lut(lut):
        # check that lut is correct
        assert isinstance(lut, xr.DataArray)
        try:
            units = lut.attrs['units']
        except KeyError:
            raise KeyError("lut has no lut.attrs['units']")

        allowed_units = ['linear', 'dB']
        if units not in allowed_units:
            raise ValueError("Unknown lut units '%s'. Allowed are '%s'" % (units, allowed_units))

        if lut.ndim == 2:
            good_dims = ('incidence', 'wspd')
            if not (lut.dims == good_dims):
                raise IndexError("Bad dims '%s'. Should be '%s'" % (lut.dims, good_dims))
        elif lut.ndim == 3:
            good_dims = ('incidence', 'wspd', 'phi')
            if not (lut.dims == good_dims):
                raise IndexError("Bad dims '%s'. Should be '%s'" % (lut.dims, good_dims))
        else:
            raise IndexError("Bad dims '%s'" % lut.dims)

        return True

    @property
    def iscopol(self):
        """True if model is copol"""
        return len(set(self.pol)) == 1

    @property
    def iscrosspol(self):
        """True if model is crosspol"""
        return len(set(self.pol)) == 2

    def to_lut(self, units='linear'):
        """Get the model's lut"""

        lut = self._raw_lut()

        self._check_lut(lut)


        final_lut = lut

        if units is None:
            return final_lut

        if units == 'dB':
            if lut.attrs['units'] == 'dB':
                final_lut = lut
            elif lut.attrs['units'] == 'linear':
                final_lut = 10 * np.log10(lut + 1e-15)  # clip with epsilon to avoid nans
                final_lut.attrs['units'] = 'dB'
        elif units == 'linear':
            if lut.attrs['units'] == 'linear':
                final_lut = lut
            elif lut.attrs['units'] == 'dB':
                final_lut = 10. ** (lut / 10.)
                final_lut.attrs['units'] = 'linear'
        else:
            raise ValueError("Unit not known: %s. Known are 'dB' or 'linear' " % units)

        final_lut.attrs['model'] = self.name

        return final_lut

    @abstractmethod
    def __call__(self, inc, wspd, phi=None):
        raise NotImplementedError(self.__class__)

    def __repr__(self):
        return "<%s('%s') pols=%s>" % (self.__class__.__name__, self.name, self.pols)


class LutModel(Model):

    def __call__(self, inc, wspd, phi=None, units=None):
        try:
            ndim = inc.ndim
        except AttributeError:
            ndim = 0

        if ndim == 2:
            raise NotImplementedError('2D array are implemented')
        lut = self.to_lut(units=units)
        if 'phi' in lut.dims:
            res = lut.interp(incidence=inc, wspd=wspd, phi=phi)
        else:
            res = lut.interp(incidence=inc, wspd=wspd)

        if ndim == 0:
            return res.item()
        else:
            return res


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
    dict
        dict of available models. Key is name, value is model.

    """
    if pol is None:
        return Model.available_models
    else:
        models_found = {}
        for name, model in Model.available_models.items():
            if pol == model.pol:
                models_found[name] = model
        return models_found

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

    return available_models()[name]
