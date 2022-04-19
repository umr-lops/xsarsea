from numba import njit
import numpy as np


@njit
def cmod5(u10, phi, inc, neutral=True):
    """
    """
    # Coefficients and constants
    if neutral is True:  # CMOD5.n coefficients
        c = [0., -0.6878, -0.7957, 0.338, -0.1728, 0., 0.004, 0.1103,
             0.0159, 6.7329, 2.7713, -2.2885, 0.4971, -0.725, 0.045, 0.0066,
             0.3222, 0.012, 22.7, 2.0813, 3., 8.3659, -3.3428, 1.3236,
             6.2437, 2.3893, 0.3249, 4.159, 1.693]
    else:  # CMOD5 coefficients
        c = [0., -0.688, -0.793, 0.338, -0.173, 0., 0.004, 0.111,
             0.0162, 6.34, 2.57, -2.18, 0.4, -0.6, 0.045, 0.007,
             0.33, 0.012, 22., 1.95, 3., 8.39, -3.44, 1.36, 5.35,
             1.99, 0.29, 3.80, 1.53]
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
    s = a2 * u10
    a3 = 1. / (1. + np.exp(-s0))
    slts0 = s < s0
    a3[~slts0] = 1. / (1. + np.exp(-s[~slts0]))
    a3[slts0] = a3[slts0] * (s[slts0] / s0[slts0]
                             ) ** (s0[slts0] * (1. - a3[slts0]))
    b0 = (a3 ** gam) * 10. ** (a0 + a1 * u10)

    # B1 term
    b1 = c[15] * u10 * (0.5 + x - np.tanh(4. * (x + c[16] + c[17] * u10)))
    b1 = (c[14] * (1. + x) - b1) / (np.exp(0.34 * (u10 - c[18])) + 1.)

    # B2 term
    v0 = c[21] + c[22] * x + c[23] * x2
    d1 = c[24] + c[25] * x + c[26] * x2
    d2 = c[27] + c[28] * x
    v2 = (u10 / v0 + 1.)
    v2lty0 = v2 < y0
    v2[v2lty0] = a + b * (v2[v2lty0] - 1.) ** pn
    b2 = (-d1 + d2 * v2) * np.exp(-v2)

    # Sigma0 according to Fourier terms
    sig = b0 * (1. + b1 * cosphi + b2 * (2. * cosphi ** 2. - 1.)) ** zpow
    return sig


@njit
def cmod_like_CR(inc, u10):
    # inc = xdata[0]
    # u10 = xdata[1]

    c1 = -3.14993013e+00
    c2 = -5.97976767e-01
    c3 = -3.27075281e-01
    c4 = -4.69016576e-01
    c5 = 2.52596490e-02
    c6 = 1.05453695e-02
    c7 = 8.23746078e+09
    c8 = -1.70926452e+09
    c9 = -6.50638418e+05
    c10 = 7.50378262e+18
    c11 = -7.97374621e+18
    c12 = 1.63073350e+12
    c13 = -4.22692526e+17

    zpow = 1.6
    thetm = 40.
    thethr = 25.

    # y0 = c[19]
    # pn = c[20]
    # a = y0 - (y0 - 1.) / pn
    # b = 1. / (pn * (y0 - 1.) ** (pn - 1.))

    # Angles
    x = (inc - thetm) / thethr
    x2 = x ** 2.

    # B0 term
    a0 = c1 + c2 * x + c3 * x2 + c4 * x * x2
    a1 = c5 + c6 * x
    a2 = c7 + c8 * x
    gam = c9 + c10 * x + c11 * x2
    s0 = c12 + c13 * x
    s = a2 * u10
    a3 = 1. / (1. + np.exp(-s0))
    slts0 = s < s0
    a3[~slts0] = 1. / (1. + np.exp(-s[~slts0]))
    a3[slts0] = a3[slts0] * (s[slts0] / s0[slts0]
                             ) ** (s0[slts0] * (1. - a3[slts0]))
    a3 = a3 * (s / s0) ** (s0 * (1. - a3))
    # a3=1
    b0 = (a3 ** gam) * 10. ** (a0 + a1 * u10)
    # b0 = 10*np.log10(a3 ** gam) + a0 + a1*u10
    # b0 = 10. ** ( a0 + a1*u10 )

    return b0


@njit
def cmod_like_CR_2(inc, u10):
    # inc = xdata[0]
    # u10 = xdata[1]

    a0 = 3.01372545e-05
    a1 = -4.75452800e-07
    b0 = 1.96134137e+00
    b1 = -1.26396011e-03
    c0 = 1.30450788e+01
    c1 = -2.60843967e+00
    d0 = -4.30657986e-03
    d1 = 7.52195823e-05

    #,d,e,f,g):#
    # ,p0):

    # c = d + e*(u10-f)**g
    # sig_dB = a + b*u10**c
    # return sig_dB
    # b = c+d*u+e*u**2
    # sig = a*u10**b
    a = a0 + a1*inc  # + a2*inc**2
    b = b0 + b1*inc  # + b2*inc**2
    c = c0 + c1*inc  # + c2*inc**2
    d = d0 + d1*inc  # + d2*inc**2

    # 0.000006*xx**(1.85+(xx-30)*(-1*0.001))
    sig = a*u10**(b+d*(u10-c))

    return sig


### OLD smap & smos wrond res (3km) in coloc files ###

@njit
def cmod_like_CR_3(inc, u10):

    a0 = 6.04514162e-05
    a1 = -8.35047917e-07
    b0 = 1.51102385e+00
    b1 = -7.89429818e-04

    # c = d + e*(u10-f)**g
    # sig_dB = a + b*u10**c
    # return sig_dB
    # b = c+d*u+e*u**2
    # sig = a*u10**b
    a = a0 + a1*inc  # + a2*inc**2
    b = b0 + b1*inc  # + b2*inc**2
    # c = c0 + c1*inc #+ c2*inc**2
    # d = d0 + d1*inc #+ d2*inc**2

    # 0.000006*xx**(1.85+(xx-30)*(-1*0.001))
    # sig = a*u10**(b+d*(u10-c))
    sig = a*u10**(b)

    return sig


@njit
def cmod_like_CR_4(inc, u10):
    # FOR RS2
    a0 = 1.49874540e-04
    a1 = -5.22780344e-06
    a2 = 4.70774846e-08
    b0 = 1.44620078e+00
    b1 = -9.53090689e-03
    b2 = 3.57441209e-04

    #,d,e,f,g):#
    # ,p0):

    # c = d + e*(u10-f)**g
    # sig_dB = a + b*u10**c
    # return sig_dB
    # b = c+d*u+e*u**2
    # sig = a*u10**b
    a = a0 + a1*inc + a2*inc**2
    b = b0 + b1*inc + b2*inc**2
    # a = a0 + a1*inc# + a2*inc**2
    # b = b0 + b1*inc# + b2*inc**2
    # c = c0 + c1*inc #+ c2*inc**2
    # d = d0 + d1*inc #+ d2*inc**2

    # 0.000006*xx**(1.85+(xx-30)*(-1*0.001))
    # sig = a*u10**(b+d*(u10-c))
    sig = a*u10**(b)

    return sig


@njit
def cmod_like_CR_5(inc, u10):
    # FOR S1A/S1B

    a0 = 0.00013106836021008122
    a1 = -4.530598283705591e-06
    a2 = 4.429277425062766e-08
    b0 = 1.3925444179360706
    b1 = 0.004157838450541205
    b2 = 3.4735809771069953e-05

    #,d,e,f,g):#
    # ,p0):

    # c = d + e*(u10-f)**g
    # sig_dB = a + b*u10**c
    # return sig_dB
    # b = c+d*u+e*u**2
    # sig = a*u10**b
    a = a0 + a1*inc + a2*inc**2
    b = b0 + b1*inc + b2*inc**2
    # a = a0 + a1*inc# + a2*inc**2
    # b = b0 + b1*inc# + b2*inc**2
    # c = c0 + c1*inc #+ c2*inc**2
    # d = d0 + d1*inc #+ d2*inc**2

    # 0.000006*xx**(1.85+(xx-30)*(-1*0.001))
    # sig = a*u10**(b+d*(u10-c))
    sig = a*u10**(b)

    return sig


@njit
def cmod_like_CR_7(inc, u10):
    # array([ 1.95413174e-05, -1.97502988e-08, -6.20960769e-09,  2.43508866e+00,
    #   -4.05945173e-02,  6.57808362e-04])
    # FOR RS2
    a0 = 1.95413174e-05
    a1 = -1.97502988e-08
    a2 = -6.20960769e-09
    b0 = 2.43508866e+00
    b1 = -4.05945173e-02
    b2 = 6.57808362e-04

    a = a0 + a1*inc + a2*inc**2
    b = b0 + b1*inc + b2*inc**2

    sig = a*u10**(b)

    return sig


@njit
def cmod_like_CR_8(inc, u10):
    # array([ 1.37636993e-04, -4.71175564e-06,  4.12477619e-08,  1.50892081e+00,
    #   -1.30471202e-02,  4.23936064e-04])
    # FOR RS2
    a0 = 1.37636993e-04
    a1 = -4.71175564e-06
    a2 = 4.12477619e-08
    b0 = 1.50892081e+00
    b1 = -1.30471202e-02
    b2 = 4.23936064e-04

    a = a0 + a1*inc + a2*inc**2
    b = b0 + b1*inc + b2*inc**2

    sig = a*u10**(b)

    return sig

### NEW smap & smos good res in coloc files ###


@njit
def cmod_like_CR_10(inc, u10):
    # array([ 1.32442623e-04, -4.77000091e-06,  4.56020016e-08,  1.39503529e+00,
    #    -1.13913155e-03,  1.97108148e-04])

    # FOR RS2 based on SMAP

    a0 = 1.32442623e-04
    a1 = -4.77000091e-06
    a2 = 4.56020016e-08
    b0 = 1.39503529e+00
    b1 = -1.13913155e-03
    b2 = 1.97108148e-04

    a = a0 + a1*inc + a2*inc**2
    b = b0 + b1*inc + b2*inc**2

    sig = a*u10**(b)

    return sig


@njit
def cmod_like_CR_11(inc, u10):
    # array([ 1.53368802e-04, -5.37145074e-06,  4.86660756e-08,  1.42770612e+00,
    #     -8.65922095e-03,  3.43272988e-04])

    # FOR RS2 based on SMAP + SFMR
    a0 = 1.53368802e-04
    a1 = -5.37145074e-06
    a2 = 4.86660756e-08
    b0 = 1.42770612e+00
    b1 = -8.65922095e-03
    b2 = 3.43272988e-04

    a = a0 + a1*inc + a2*inc**2
    b = b0 + b1*inc + b2*inc**2

    sig = a*u10**(b)

    return sig


@njit
def cmod_like_CR_12(inc, u10):
    # array([ 1.82826880e-05,  1.55856688e-07, -8.75021268e-09,  2.38330363e+00,
    #  -3.84172887e-02,  5.97105405e-04])

    # FOR RS2 based on SMOS
    a0 = 1.82826880e-05
    a1 = 1.55856688e-07
    a2 = -8.75021268e-09
    b0 = 2.38330363e+00
    b1 = -3.84172887e-02
    b2 = 5.97105405e-04

    a = a0 + a1*inc + a2*inc**2
    b = b0 + b1*inc + b2*inc**2

    sig = a*u10**(b)

    return sig


@njit
def cmod_like_CR_13(inc, u10):
    # array([1.34760283e-04, -4.56427601e-06,  3.94618073e-08,  1.52220140e+00,
    #        -1.36549252e-02,  4.29236850e-04])

    # FOR RS2 based on SMOS and SFMR
    a0 = 1.37636993e-04
    a1 = -4.71175564e-06
    a2 = 4.12477619e-08
    b0 = 1.50892081e+00
    b1 = -1.30471202e-02
    b2 = 4.23936064e-04

    a = a0 + a1*inc + a2*inc**2
    b = b0 + b1*inc + b2*inc**2

    sig = a*u10**(b)

    return sig


@njit
def cmod_like_CR_15(inc, u10):
    # array([ 1.69102352e-04, -8.62868059e-06,  1.61024724e-07, -1.08600013e-09,
    #        1.44889415e+00, -9.24902315e-03,  6.11390671e-04, -4.61515791e-06])

    # FOR RS2 based on SMOS and SFMR
    a0 = 1.69102352e-04
    a1 = -8.62868059e-06
    a2 = 1.61024724e-07
    a3 = -1.08600013e-09

    b0 = 1.44889415e+00
    b1 = -9.24902315e-03
    b2 = 6.11390671e-04
    b3 = -4.61515791e-06

    a = a0 + a1*inc + a2*inc**2 + a3*inc**3
    b = b0 + b1*inc + b2*inc**2 + b3*inc**3

    sig = a*u10**(b)

    return sig


corresponding_gmfs = {
    1: cmod_like_CR,
    2: cmod_like_CR_2,
    3: cmod_like_CR_3,
    4: cmod_like_CR_4,
    5: cmod_like_CR_5,
    7: cmod_like_CR_7,
    8: cmod_like_CR_8,

    # sar at the good resolution
    10: cmod_like_CR_10,
    11: cmod_like_CR_11,
    12: cmod_like_CR_12,
    13: cmod_like_CR_13,
    15: cmod_like_CR_15
}
