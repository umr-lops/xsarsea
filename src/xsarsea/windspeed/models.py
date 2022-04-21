# class that represent a model (lut or gmf)

from abc import abstractmethod
import numpy as np


class Model:

    available_models = {}
    """dict of available registered models"""

    @abstractmethod
    def __init__(self, name, **kwargs):
        self.name = name
        self.pols = kwargs.pop('pols', None)
        self.units = kwargs.pop('units', None)
        self.inc_range = kwargs.pop('inc_range', None)
        self.phi_range = kwargs.pop('phi_range', None)
        self.wspd_range = kwargs.pop('wspd_range', None)
        self.__class__.available_models[name] = self

    @abstractmethod
    def _raw_lut(self):
        pass

    def to_lut(self, units='linear'):
        """Get the model's lut"""

        lut = self._raw_lut()

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

    def __repr__(self):
        return "<%s('%s') pols=%s>" % (self.__class__.__name__, self.name, self.pols)


def available_models(pol=None):

    return Model.available_models

    models_found = {}
    for name, model in Model.available_models.items():
        models_found[name] = model

    return models_found
