import numpy as np

from xsarsea.windspeed.gmfs import GmfModel

# analytical functions


def gmf_cmod5_generic(neutral=False):
    # return cmod5 or cmod5n (neutral) function

    # Coefficients and constants

    # CMOD5 coefficients
    c = np.array(
        [
            0.0,
            -0.688,
            -0.793,
            0.338,
            -0.173,
            0.0,
            0.004,
            0.111,
            0.0162,
            6.34,
            2.57,
            -2.18,
            0.4,
            -0.6,
            0.045,
            0.007,
            0.33,
            0.012,
            22.0,
            1.95,
            3.0,
            8.39,
            -3.44,
            1.36,
            5.35,
            1.99,
            0.29,
            3.80,
            1.53,
        ]
    )
    name = "gmf_cmod5"
    if neutral:
        # CMOD5h coefficients
        c = np.array(
            [
                0.0,
                -0.6878,
                -0.7957,
                0.338,
                -0.1728,
                0.0,
                0.004,
                0.1103,
                0.0159,
                6.7329,
                2.7713,
                -2.2885,
                0.4971,
                -0.725,
                0.045,
                0.0066,
                0.3222,
                0.012,
                22.7,
                2.0813,
                3.0,
                8.3659,
                -3.3428,
                1.3236,
                6.2437,
                2.3893,
                0.3249,
                4.159,
                1.693,
            ]
        )
        name = "gmf_cmod5n"

    @GmfModel.register(name, wspd_range=[0.2, 50.0], pol="VV", units="linear", defer=False)
    def gmf_cmod5(inc, wspd, phi):
        zpow = 1.6
        thetm = 40.0
        thethr = 25.0
        y0 = c[19]
        pn = c[20]
        a = y0 - (y0 - 1.0) / pn
        b = 1.0 / (pn * (y0 - 1.0) ** (pn - 1.0))

        # Angles
        cosphi = np.cos(np.deg2rad(phi))
        x = (inc - thetm) / thethr
        x2 = x**2.0

        # B0 term
        a0 = c[1] + c[2] * x + c[3] * x2 + c[4] * x * x2
        a1 = c[5] + c[6] * x
        a2 = c[7] + c[8] * x
        gam = c[9] + c[10] * x + c[11] * x2
        s0 = c[12] + c[13] * x
        s = a2 * wspd
        a3 = 1.0 / (1.0 + np.exp(-s0))

        if s < s0:
            a3 = a3 * (s / s0) ** (s0 * (1.0 - a3))
        else:
            a3 = 1.0 / (1.0 + np.exp(-s))

        b0 = (a3**gam) * 10.0 ** (a0 + a1 * wspd)

        # B1 term
        b1 = c[15] * wspd * (0.5 + x - np.tanh(4.0 * (x + c[16] + c[17] * wspd)))
        b1 = (c[14] * (1.0 + x) - b1) / (np.exp(0.34 * (wspd - c[18])) + 1.0)

        # B2 term
        v0 = c[21] + c[22] * x + c[23] * x2
        d1 = c[24] + c[25] * x + c[26] * x2
        d2 = c[27] + c[28] * x
        v2 = wspd / v0 + 1.0
        if v2 < y0:
            v2 = a + b * (v2 - 1.0) ** pn

        b2 = (-d1 + d2 * v2) * np.exp(-v2)

        # Sigma0 according to Fourier terms
        sig = b0 * (1.0 + b1 * cosphi + b2 * (2.0 * cosphi**2.0 - 1.0)) ** zpow
        return sig

    return gmf_cmod5


# register gmfs gmf_cmod5 and gmf_cmod5n
gmf_cmod5_generic(neutral=False)
gmf_cmod5_generic(neutral=True)


@GmfModel.register(wspd_range=[0.2, 50.0], pol="VV", units="linear", defer=False)
def gmf_cmodifr2(inc_angle, wind_speed, wind_dir):

    C = np.zeros(26)

    # init CMOD-IFR2 coef
    C[0] = 0.0
    C[1] = -2.437597
    C[2] = -1.5670307
    C[3] = 0.3708242
    C[4] = -0.040590
    C[5] = 0.404678
    C[6] = 0.188397
    C[7] = -0.027262
    C[8] = 0.064650
    C[9] = 0.054500
    C[10] = 0.086350
    C[11] = 0.055100
    C[12] = -0.058450
    C[13] = -0.096100
    C[14] = 0.412754
    C[15] = 0.121785
    C[16] = -0.024333
    C[17] = 0.072163
    C[18] = -0.062954
    C[19] = 0.015958
    C[20] = -0.069514
    C[21] = -0.062945
    C[22] = 0.035538
    C[23] = 0.023049
    C[24] = 0.074654
    C[25] = -0.014713

    T = inc_angle
    wind = wind_speed

    tetai = (T - 36.0) / 19.0
    xSQ = tetai * tetai
    # P0 = 1.0
    P1 = tetai
    P2 = (3.0 * xSQ - 1.0) / 2.0
    P3 = (5.0 * xSQ - 3.0) * tetai / 2.0
    ALPH = C[1] + C[2] * P1 + C[3] * P2 + C[4] * P3
    BETA = C[5] + C[6] * P1 + C[7] * P2
    ang = wind_dir
    cosi = np.cos(np.deg2rad(ang))
    cos2i = 2.0 * cosi * cosi - 1.0
    tetamin = 18.0
    tetamax = 58.0
    tetanor = (2.0 * T - (tetamin + tetamax)) / (tetamax - tetamin)
    vmin = 3.0
    vmax = 25.0
    vitnor = (2.0 * wind - (vmax + vmin)) / (vmax - vmin)
    pv0 = 1.0
    pv1 = vitnor
    pv2 = 2 * vitnor * pv1 - pv0
    pv3 = 2 * vitnor * pv2 - pv1
    pt0 = 1.0
    pt1 = tetanor
    pt2 = 2 * tetanor * pt1 - pt0
    # pt3 = 2 * tetanor * pt2 - pt1
    b1 = C[8] + C[9] * pv1 + (C[10] + C[11] * pv1) * pt1 + (C[12] + C[13] * pv1) * pt2
    tetamin = 18.0
    tetamax = 58.0
    tetanor = (2.0 * T - (tetamin + tetamax)) / (tetamax - tetamin)
    vmin = 3.0
    vmax = 25.0
    vitnor = (2.0 * wind - (vmax + vmin)) / (vmax - vmin)
    pv0 = 1.0
    pv1 = vitnor
    pv2 = 2 * vitnor * pv1 - pv0
    pv3 = 2 * vitnor * pv2 - pv1
    pt0 = 1.0
    pt1 = tetanor
    pt2 = 2 * tetanor * pt1 - pt0
    # pt3 = 2 * tetanor * pt2 - pt1
    result = (
        C[14]
        + C[15] * pt1
        + C[16] * pt2
        + (C[17] + C[18] * pt1 + C[19] * pt2) * pv1
        + (C[20] + C[21] * pt1 + C[22] * pt2) * pv2
        + (C[23] + C[24] * pt1 + C[25] * pt2) * pv3
    )
    b2 = result

    b0 = np.power(10.0, (ALPH + BETA * np.sqrt(wind)))
    sig = b0 * (1.0 + b1 * cosi + np.tanh(b2) * cos2i)

    return sig


#
# gmf_dummy example
#
# @GmfModel.register(wspd_range=[3., 80.], pol='VH', units='linear')
# def gmf_dummy(incidence, speed, phi=None):
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


@GmfModel.register(wspd_range=[3.0, 80.0], pol="VH", units="linear", defer=False)
def gmf_rs2_v2(incidence, speed, phi=None):
    """
    Radarsat-2 VH GMF : relation between sigma0, incidence and windspeed.

    Parameters
    ----------
    incidence: xarray.DataArray
        incidence angle [deg]
    speed: xarray.DataArray
        wind speed [m/s]

    Returns
    -------
    sigma0: xarray.DataArray
        linear sigma0

    """
    # constants params

    Z1_p = np.array([6.55519203e-06, 2.49753154e00, -1.35734881e-02])
    Z2_p = np.array(
        [
            1.47342197e-04,
            -4.07334797e-06,
            3.43593382e-08,
            1.10188639e00,
            1.40782758e-02,
            -1.53748743e-04,
        ]
    )
    Final_p = np.array([-0.18675905, 24.48859492, 0.19185442, 25.38275738])

    # Z1

    a0_Z1 = Z1_p[0]
    b0_Z1 = Z1_p[1]
    b1_Z1 = Z1_p[2]

    a_Z1 = a0_Z1
    b_Z1 = b0_Z1 + b1_Z1 * incidence
    sig_Z1 = a_Z1 * speed ** (b_Z1)

    # Z2
    a0_Z2 = Z2_p[0]
    a1_Z2 = Z2_p[1]
    a2_Z2 = Z2_p[2]
    b0_Z2 = Z2_p[3]
    b1_Z2 = Z2_p[4]
    b2_Z2 = Z2_p[5]

    a_Z2 = a0_Z2 + a1_Z2 * incidence + a2_Z2 * incidence**2
    b_Z2 = b0_Z2 + b1_Z2 * incidence + b2_Z2 * incidence**2
    sig_Z2 = a_Z2 * speed ** (b_Z2)

    c0, c1 = Final_p[0], Final_p[1]
    c2, c3 = Final_p[2], Final_p[3]
    sigmoid_Z1 = 1 / (1 + np.exp(-c0 * (speed - c1)))
    sigmoid_Z2 = 1 / (1 + np.exp(-c2 * (speed - c3)))
    sig_Final = sig_Z1 * sigmoid_Z1 + sig_Z2 * sigmoid_Z2
    return sig_Final


@GmfModel.register(wspd_range=[3.0, 80.0], pol="VH", units="linear", defer=False)
def gmf_s1_v2(incidence, speed, phi=None):
    """
    Sentinel-1 VH GMF : relation between sigma0, incidence and windspeed.

    Parameters
    ----------
    incidence: xarray.DataArray
        incidence angle [deg]
    speed: xarray.DataArray
        wind speed [m/s]

    Returns
    -------
    sigma0: xarray.DataArray
        linear sigma0
    """

    # constants params

    Z1_p = np.array([2.13755392e-06, 2.47395267e00, -2.85775085e-03])

    Z2_p = np.array(
        [
            6.54058552e-05,
            -2.43845137e-06,
            2.87698338e-08,
            1.14509104e00,
            3.41828829e-02,
            -4.79715441e-04,
        ]
    )
    Final_p = np.array([-0.23257086, 12.39717002, 0.21667263, 12.22862991])

    # Z1
    a0_Z1 = Z1_p[0]
    b0_Z1 = Z1_p[1]
    b1_Z1 = Z1_p[2]

    a_Z1 = a0_Z1
    b_Z1 = b0_Z1 + b1_Z1 * incidence
    sig_Z1 = a_Z1 * speed ** (b_Z1)

    # Z2
    a0_Z2 = Z2_p[0]
    a1_Z2 = Z2_p[1]
    a2_Z2 = Z2_p[2]
    b0_Z2 = Z2_p[3]
    b1_Z2 = Z2_p[4]
    b2_Z2 = Z2_p[5]

    a_Z2 = a0_Z2 + a1_Z2 * incidence + a2_Z2 * incidence**2
    b_Z2 = b0_Z2 + b1_Z2 * incidence + b2_Z2 * incidence**2
    sig_Z2 = a_Z2 * speed ** (b_Z2)

    c0, c1 = Final_p[0], Final_p[1]
    c2, c3 = Final_p[2], Final_p[3]
    sigmoid_Z1 = 1 / (1 + np.exp(-c0 * (speed - c1)))
    sigmoid_Z2 = 1 / (1 + np.exp(-c2 * (speed - c3)))
    sig_Final = sig_Z1 * sigmoid_Z1 + sig_Z2 * sigmoid_Z2
    return sig_Final


@GmfModel.register(wspd_range=[3.0, 80.0], pol="VH", units="linear", defer=False)
def gmf_rcm_noaa(incidence, speed, phi=None):
    """
    Radarsat Consteallation Mission VH GMF : relation between sigma0, incidence and windspeed.
    From NOAA, named gmf_rcm_v1.
    Parameters
    ----------
    incidence: xarray.DataArray
        incidence angle [deg]
    speed: xarray.DataArray
        wind speed [m/s]

    Returns
    -------
    sigma0: xarray.DataArray
        linear sigma0

    """
    # constants params

    Z1_p = np.array([2.2309436836414871e-12, 8.3374911282878728, -0.033443488982800210])
    Z2_p = np.array(
        [
            7.7945050373193260e-05,
            -2.4425748662769216e-06,
            2.7625550632547159e-08,
            1.2524896108831316,
            0.019203092214131894,
            -0.00028408046502692580,
        ]
    )
    Final_p = np.array(
        [-0.34498737004629487, 12.558975188752012, 0.12713502524515713, 4.2806865431046752]
    )

    # Z1
    a0_Z1 = Z1_p[0]
    b0_Z1 = Z1_p[1]
    b1_Z1 = Z1_p[2]

    a_Z1 = a0_Z1
    b_Z1 = b0_Z1 + b1_Z1 * incidence
    sig_Z1 = a_Z1 * speed ** (b_Z1)

    # Z2
    a0_Z2 = Z2_p[0]
    a1_Z2 = Z2_p[1]
    a2_Z2 = Z2_p[2]
    b0_Z2 = Z2_p[3]
    b1_Z2 = Z2_p[4]
    b2_Z2 = Z2_p[5]

    a_Z2 = a0_Z2 + a1_Z2 * incidence + a2_Z2 * incidence**2
    b_Z2 = b0_Z2 + b1_Z2 * incidence + b2_Z2 * incidence**2
    sig_Z2 = a_Z2 * speed ** (b_Z2)

    c0, c1 = Final_p[0], Final_p[1]
    c2, c3 = Final_p[2], Final_p[3]
    sigmoid_Z1 = 1 / (1 + np.exp(-c0 * (speed - c1)))
    sigmoid_Z2 = 1 / (1 + np.exp(-c2 * (speed - c3)))
    sig_Final = sig_Z1 * sigmoid_Z1 + sig_Z2 * sigmoid_Z2
    return sig_Final
