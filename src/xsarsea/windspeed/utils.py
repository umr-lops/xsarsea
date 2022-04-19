import os
import warnings
import numpy as np

def wind_to_img(u, v, ground_heading, convention='antenna'):
    wind_azi = np.sqrt(u ** 2 + v ** 2) * \
               np.exp(1j * (np.arctan2(u, v) - np.deg2rad(ground_heading)))

    wind_azi.attrs['comment'] = """
        Ancillary wind, as a complex number.
        complex angle is the wind direction relative to azimuth (atrack)
        module is windspeed
        real part is atrack wind component
        imag part is xtrack wind component
        """

    if convention == 'antenna':
        # transpose real and img to get antenna convention
        wind_antenna = np.imag(wind_azi) + 1j * np.real(wind_azi)
        wind_antenna.attrs['comment'] = """
                Ancillary wind, as a complex number.
                complex angle is the wind direction relative to antenna (xtrack)
                module is windspeed
                real part is antenna (xtrack) wind component 
                imag part is atrack wind component
                """
        return wind_antenna
    else:
        return wind_azi


def nesz_flattening(noise, inc):
    """

    Parameters
    ----------
    noise: array-like
        noise array (nesz), with shape (atrack, xtrack)
    inc: array-like
        incidence array

    Returns
    -------
    array-like
        flattened noise

    """

    if noise.ndim != 2:
        raise IndexError('Only 2D noise allowed')

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*empty.*', category=RuntimeWarning)
        noise_mean = np.nanmean(noise, axis=0)

    try:
        # unlike np.mean, np.nanmean return a numpy array, even if noise is an xarray
        # if this behaviour change in the future, we want to be sure to have a numpy array
        noise_mean = noise_mean.values
    except AttributeError:
        pass

    def _noise_flattening_1row(noise_row, inc_row):

        noise_flat = noise_row.copy()

        # replacing nan values by nesz mean value for concerned incidence
        noise_flat[np.isnan(noise_flat)] = noise_mean[np.isnan(noise_flat)]

        # to dB
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            noise_db = 10. * np.log10(noise_flat)

        try:
            _coef = np.polyfit(inc_row[np.isfinite(noise_db)],
                               noise_db[np.isfinite(noise_db)], 1)
        except TypeError:
            # noise is all nan
            return np.full(noise_row.shape, np.nan)

        # flattened, to linear
        noise_flat = 10. ** ((inc_row * _coef[0] + _coef[1] - 1.0) / 10.)

        return noise_flat

    # incidence is almost constant along atrack dim, so we can make it 1D
    return np.apply_along_axis(_noise_flattening_1row, 1, noise, np.nanmean(inc, axis=0))