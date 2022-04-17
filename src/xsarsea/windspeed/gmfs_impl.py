from .gmfs import register_gmf
import numpy as np

# analytical functions


def gmf_cmod5_generic(neutral=False):
    # return cmod5 or cmod5n (neutral) function

    # Coefficients and constants

    # CMOD5 coefficients
    c = np.array([0., -0.688, -0.793, 0.338, -0.173, 0., 0.004, 0.111,
                  0.0162, 6.34, 2.57, -2.18, 0.4, -0.6, 0.045, 0.007,
                  0.33, 0.012, 22., 1.95, 3., 8.39, -3.44, 1.36, 5.35,
                  1.99, 0.29, 3.80, 1.53])
    name = 'gmf_cmod5'
    if neutral:
        # CMOD5h coefficients
        c = np.array([0., -0.6878, -0.7957, 0.338, -0.1728, 0., 0.004, 0.1103,
                      0.0159, 6.7329, 2.7713, -2.2885, 0.4971, -0.725, 0.045, 0.0066,
                      0.3222, 0.012, 22.7, 2.0813, 3., 8.3659, -3.3428, 1.3236,
                      6.2437, 2.3893, 0.3249, 4.159, 1.693])
        name = 'gmf_cmod5n'

    @register_gmf(name, inc_range=[17., 50.], wspd_range=[0.2, 50.], phi_range=[0., 180.])
    def gmf_cmod5(inc, wspd, phi):
        zpow = 1.6
        thetm = 40.
        thethr = 25.
        y0 = c[19]
        pn = c[20]
        a = y0 - (y0 - 1.) / pn
        b = 1. / (pn * (y0 - 1.) ** (pn - 1.))

        # Angles
        cosphi = np.cos(np.deg2rad(phi))
        x = (inc - thetm) / thethr
        x2 = x ** 2.

        # B0 term
        a0 = c[1] + c[2] * x + c[3] * x2 + c[4] * x * x2
        a1 = c[5] + c[6] * x
        a2 = c[7] + c[8] * x
        gam = c[9] + c[10] * x + c[11] * x2
        s0 = c[12] + c[13] * x
        s = a2 * wspd
        a3 = 1. / (1. + np.exp(-s0))

        if s < s0:
            a3 = a3 * (s / s0) ** (s0 * (1. - a3))
        else:
            a3 = 1. / (1. + np.exp(-s))

        b0 = (a3 ** gam) * 10. ** (a0 + a1 * wspd)

        # B1 term
        b1 = c[15] * wspd * (0.5 + x - np.tanh(4. * (x + c[16] + c[17] * wspd)))
        b1 = (c[14] * (1. + x) - b1) / (np.exp(0.34 * (wspd - c[18])) + 1.)

        # B2 term
        v0 = c[21] + c[22] * x + c[23] * x2
        d1 = c[24] + c[25] * x + c[26] * x2
        d2 = c[27] + c[28] * x
        v2 = (wspd / v0 + 1.)
        if v2 < y0:
            v2 = a + b * (v2 - 1.) ** pn

        b2 = (-d1 + d2 * v2) * np.exp(-v2)

        # Sigma0 according to Fourier terms
        sig = b0 * (1. + b1 * cosphi + b2 * (2. * cosphi ** 2. - 1.)) ** zpow
        return sig

    return gmf_cmod5

# register gmfs gmf_cmod5 and gmf_cmod5n
gmf_cmod5_generic(neutral=False)
gmf_cmod5_generic(neutral=True)


#
# gmf_dummy example
#
#@register_gmf(inc_range=[17., 50.], wspd_range=[3., 80.], phi_range=None)
#def gmf_dummy(incidence, speed, phi=None):
#    a0 = 0.00013106836021008122
#    a1 = -4.530598283705591e-06
#    a2 = 4.429277425062766e-08
#    b0 = 1.3925444179360706
#    b1 = 0.004157838450541205
#    b2 = 3.4735809771069953e-05
#
#    a = a0 + a1 * incidence + a2 * incidence ** 2
#    b = b0 + b1 * incidence + b2 * incidence ** 2
#    sig = a * speed ** b
#
#    return sig

