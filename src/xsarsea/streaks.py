"""
implemented from:
'W. Koch, "Directional analysis of SAR images aiming at wind direction," IEEE Trans. Geosci. Remote Sens., vol. 42, no. 4, pp. 702-710, 2004.'
https://ieeexplore.ieee.org/document/1288365
https://www.climate-service-center.de/imperia/md/content/gkss/institut_fuer_kuestenforschung/ksd/paper/kochw_ieee_2004.pdf
"""

import numpy as np
from scipy import signal
import xarray as xr
import dask.array as da
import warnings
import numba

# map_overlap convolve vs numpy convolve  (memory++ !). TODO: will be deprecated once fixed.
dask_convolve = True


def streaks_direction(sigma0):
    """

    Parameters
    ----------
    sigma0: xarray.DataArray
        detrended sigma0, at 100m resolution (I1 in Koch(2004))

    Returns
    -------
    xarray.DataArray
        streaks direction, in range [-180,180], at 16km resolution.
        0 deg is azimuth satelite track (not north)

    Notes
    -----
        100m resolution `sigma0_detrend` is not checked.
        Koch(2004) say it should be 100m


    """

    if 'pol' in sigma0.dims:
        streaks_dir_list = []
        for pol in sigma0.pol:
            streaks_dir_list.append(_streaks_direction_by_pol(sigma0.sel(pol=pol))
                                    .assign_coords({'pol': pol}))
        streaks_dir = xr.concat(streaks_dir_list, 'pol')
    else:
        streaks_dir = _streaks_direction_by_pol(sigma0)
    return streaks_dir


def _streaks_direction_by_pol(sigma0):
    # internal vectorized function, see streaks_direction

    sigma0 = sigma0.fillna(0).clip(0, None)

    # lower the resolution by a factor 2, without moire effects
    i2 = R2(sigma0, {'atrack': 2, 'xtrack': 2})
    i2 = i2.fillna(0).clip(0, None)

    ampl = np.sqrt(i2)
    G1, G12, G2, G3, c = localGrad(ampl)

    hist = grad_hist(G2, c, window={'atrack': 40, 'xtrack': 40}, n_angles=72)

    smooth_hist = grad_hist_smooth(hist)

    grad_dir = find_gradient(smooth_hist)

    # streaks dir is orthogonal to gradient dir
    streaks_dir = 90 - grad_dir

    # streaks dir is only defined on [-180,180] range (ie no arrow head)
    streaks_dir = xr.where(streaks_dir >= 0, streaks_dir - 90, streaks_dir + 90) % 360 - 180

    # many computations where done to compute streaks_dir.
    # it's small, so we can persist it into memory to speed up future computation
    streaks_dir = streaks_dir.persist()

    return streaks_dir


def convolve2d(in1, in2, boundary='symm', fillvalue=0, dask=dask_convolve):
    """
    wrapper around scipy.signal.convolve2d for in1 as xarray.DataArray
    mode is forced to 'same', so axes are not changed.
    """

    try:
        _ = in1.data.map_overlap
        parallel = True
    except:
        parallel = False

    # dict mapping boundary convolve to map_overlap option
    boundary_map = {
        'symm': 'reflect',
        'wrap': 'periodic',
        'fill': fillvalue
    }

    res = in1.copy()
    if parallel and dask:
        # wrapper so every args except in1 are by default
        def _conv2d(in1, in2=in2, mode='same', boundary=boundary, fillvalue=fillvalue):
            return signal.convolve2d(in1, in2, mode=mode, boundary=boundary)

        # make sure the smallest in1 chunk size is >= in2.shape.
        min_in1_chunk = tuple([min(c) for c in in1.chunks])
        if np.min(np.array(min_in1_chunk) - np.array(in2.shape)) < 0:
            in1.data = in1.data.rechunk(chunks=[int(np.median(c)) for c in in1.chunks], balance=True)

        res.data = in1.data.map_overlap(_conv2d, in2.shape, boundary=boundary_map[boundary])
    else:
        res.data = signal.convolve2d(in1.data, in2, mode='same', boundary=boundary)

    return res


def R2(image, reduc):
    """
    resample image by factor

    Parameters
    ----------
    image: xarray.DataArray with dims ['atrack', 'xtrack']
    reduc: dict like { 'atrack' : 2 , 'xtrack' : 2 } (reduce with a factor 2)

    Returns
    -------
    xarray.DataArray
        resampled
    """

    B2 = np.mat('[1,2,1; 2,4,2; 1,2,1]', float) * 1 / 16
    B2 = np.array(B2)
    B4 = signal.convolve(B2, B2)
    ones_like = lambda x: xr.DataArray(da.ones_like(x), dims=x.dims,
                                       coords=x.coords)

    # pre smooth
    _image = convolve2d(image, B4, boundary='symm')
    num = convolve2d(ones_like(_image), B4, boundary='symm')
    image = _image / num

    # resample
    image = image.coarsen(reduc, boundary='pad').mean()

    # post-smooth
    _image = convolve2d(image, B2, boundary='symm')
    num = convolve2d(ones_like(_image), B2, boundary='symm')
    image = _image / num

    return image


def localGrad(I):
    """
    compute local gradients

    Parameters
    ----------
    I: xarray.DataArray with dims['atrack', 'xtrack']
        ( from ref article, it's should be 100m resolution )

    Returns
    -------
    tuple of xarray.Dataarray (grad, grad12, grad2, grad3, c)
            - grad : complex gradient, same resolution as I
            - grad12 : grad ** 2
            - grad2 : grad12 resampled by 2 factor
            - grad3 : abs(grad12) resampled by 2 factor
            - c : grad quality

    """
    # local gradient scharr
    Dx = np.mat('[3,0,-3;10,0,-10;3,0,-3]', float) * 1 / 32
    Dy = Dx.T  # transpose
    i = complex(0, 1)
    D = Dx + i * Dy

    def convolve2d(in1=None, in2=None):
        return signal.convolve2d(in1, in2, mode='same', boundary='symm')

    # local gradient
    grad = xr.DataArray(
        I.data.map_overlap(convolve2d, in2=D, depth={'atrack': D.shape[0], 'xtrack': D.shape[0]}, boundary='symm'),
        dims=("atrack", "xtrack"), coords={"atrack": I.atrack, "xtrack": I.xtrack})
    grad.name = 'grad'
    grad = grad.persist()  # persist into memory, to speedup depending vars computations
    grad12 = grad ** 2  # squared
    grad12.name = 'grad12'
    grad2 = R2(grad12, {'atrack': 2, 'xtrack': 2})
    grad2.name = 'grad2'
    grad3 = R2(abs(grad12), {'atrack': 2, 'xtrack': 2})
    grad3.name = 'grad3'
    # grad quality
    c = abs(grad2) / (grad3 + 0.00001)
    c = c.where(c <= 1).fillna(0)
    c.name = 'c'

    return grad, grad12, grad2, grad3, c


def _grad_hist_one_box(g2, c, angles_bins, grads):
    """
    internal function that compute histogram from localGrad for only on small box.
    this function will be converted to gufunc by numba.

    Parameters
    ----------
    g2: numpy.ndarray
        2D array of g2 values from localGrad
    c: numpy.ndarray
        2D array of g2 values from localGrad
    angles_bins: numpy.ndarray
        1D array of regulary spaced angles from ]-180,  180[
    grads: numpy.ndarray
        *returned* 1D array with same shape as angles_bins, with histogram values
    """
    c_ravel = c.ravel()
    g2_ravel = g2.ravel()
    theta = np.arctan2(g2_ravel.imag, g2_ravel.real)

    # weighted gradients classes
    degree = np.degrees(theta) - 180

    # so given an angle deg, the corresponding index in angles_bin is np.round((deg-angles_start)/angles_step)
    angles_step = angles_bins[1] - angles_bins[0]
    angles_start = angles_bins[0]

    grads[:] = np.complex128(0)

    r = np.abs(g2_ravel) / (np.abs(g2_ravel) + np.median(np.abs(g2_ravel)) + 0.00001)
    r[r > 1] = 0
    for j in range(0, len(degree)):
        deg = degree[j]
        if not np.isnan(deg) and not np.isnan(r[j]) and not np.isnan(c_ravel[j]) and np.abs(
                g2_ravel[j]) != 0:  # evite d avoir des NaN
            # k is the deg index in angles_bins
            k = int(np.round((deg - angles_start) / angles_step))

            grads[k] = grads[k] + r[j] * c_ravel[j] * g2_ravel[j] / np.abs(g2_ravel[j])


# gufunc version of  _grad_hist_one_box that works one many boxes
# g2 and c have shape like [ x, y, bx, by], where bx and by are box shape
_grad_hist_gufunc = numba.guvectorize(
    [(numba.complex128[:, :], numba.float64[:, :], numba.float64[:], numba.complex128[:])], '(n,m),(n,m),(p)->(p)',
    nopython=True)(_grad_hist_one_box)


def grad_hist(g2, c, window, n_angles=72):
    """
    compute gradient histogram from g2 and c by n_angles bins

    Parameters
    ----------
    g2: xarray.DataArray
        2D array from localGrad
    c: xarray.DataArray
        2D array from localGrad, same shape as g2
    window: dict
        window size ie {'atrack': 40, 'xtrack': 40}
    n_angles: angles bins count

    Returns
    -------
    xarray.DataArray
        shape will be reduced by window size, and an 'angle' dim will be added (of size n_angles)

    """

    angles_bins = np.linspace(-180, 180, n_angles + 1)  # one extra bin
    angles_bins = (angles_bins[1:] + angles_bins[:-1]) / 2  # supress extra bin (middle)

    # make a rolling dataset with window
    window_dims = {k: "k_%s" % k for k in window.keys()}
    ds = xr.merge([g2.rename('g2'), c.rename('c')]).rolling(window, center=True).construct(window_dims).sel(
        {k: slice(window[k] // 2, None, window[k]) for k in window.keys()})

    # FIXME: hard to make xr.apply_ufunc works with dask. If ok, following will be not required
    ds = ds.persist()
    ds = ds.compute()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hist = xr.apply_ufunc(
            _grad_hist_gufunc, ds['g2'], ds['c'], angles_bins,
            input_core_dims=[window_dims.values(), window_dims.values(), "angles"],
            exclude_dims=set(window_dims.values()),
            output_core_dims=[['angles']],
            # doesn't works with dask
            dask='parallelized',
            dask_gufunc_kwargs={
                'output_sizes': {
                    'angles': angles_bins.size
                }
            },
            output_dtypes=[np.complex128]
        )
    hist = hist.rename('angles_hist').assign_coords(angles=angles_bins)

    return hist


def grad_hist_smooth(hist):
    """
    Smooth hist returned by grad_hist with kernels Bx Bx2 Bx4 Bx8.
    Histogram coordinates are angles, so begin and end are circulary wrapped.

    Parameters
    ----------
    hist: xarray.DatArray, with 'angles' dim.

    Returns
    -------
    xarray.DataArray
      same as hist, but smoothed.

    """
    Bx = np.array([1, 2, 1], float) * 1 / 4
    Bx2 = np.array([1, 0, 2, 0, 1], float) * 1 / 4
    Bx4 = np.array([1, 0, 0, 0, 2, 0, 0, 0, 1], float) * 1 / 4
    Bx8 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1], float) * 1 / 4
    Bs = [Bx, Bx2, Bx4, Bx8]

    # circular wrap
    maxsize_B = max([len(B) for B in Bs])
    smooth_hist = hist.pad({'angles': maxsize_B}, mode='wrap')

    for B in Bs:
        smooth_hist = xr.apply_ufunc(
            signal.convolve, smooth_hist, B, kwargs={'mode': 'same'},
            input_core_dims=[["angles"], ["kernel_len"]],
            output_core_dims=[['angles']],
            vectorize=True,
            output_dtypes=[np.complex128])

    # unwrap
    smooth_hist = smooth_hist.isel(angles=slice(maxsize_B, -maxsize_B))

    return smooth_hist


def _find_gradient(smooth_hist):
    hist2 = np.sqrt(smooth_hist)
    mode = find_mode(np.abs(hist2), 0., 5., circ=True)
    wpic1 = np.argmax(mode['val'])
    indi = mode['loc'][wpic1]
    gradient = hist2[indi]
    return np.angle(gradient, deg=True)


def find_gradient(smooth_hist):
    """
     Get main gradient from smooth hist.

     Parameters
     ----------
     smooth_hist: xarray.DataArray with 'angles' dim.

     Returns
     -------
     xarray.DataArray with 'angle' dim removed
         selected gradient from smooth_hist (degrees)

     Notes
     _____
     Method it different from `Koch(2004)`.
     Original method from Koch is ot take maximum histogram value.

     """
    deg = xr.apply_ufunc(_find_gradient, smooth_hist,
                         input_core_dims=[["angles"]],
                         output_core_dims=[[]],
                         vectorize=True,
                         # doesn't works with dask
                         # dask='parallelized',
                         # dask_gufunc_kwargs={
                         #    'output_sizes': {
                         #        'angles': angles_bins.size
                         #    }
                         # },
                         output_dtypes=[np.float64])
    return deg


def find_mode(tab00, smooth_fac, noise, circ=False):
    tab0 = tab00.copy()
    tab = np.asarray(tab0, dtype='float')
    mm = np.max(tab)

    if circ:
        ## duplicate and center
        arg_max = np.argmax(tab)
        sz = np.size(tab)
        tab_o = tab.copy()
        tab = np.hstack([tab, tab, tab])

        # ;; TEST the input data
    if min(tab) < 0:
        logger.info('input data must be positive')
        raise ValueError('input data must be positive')

    minzero = 0
    if min(tab) == 0:
        minzero = 1
        tab = tab + 1

    sm = 0
    # ;; marge in Percentage
    if np.size(tab) < smooth_fac:
        smooth_fac = 0

    if smooth_fac > 0:
        cnt = np.size(tab)
        s_tab = tab.copy()
        if cnt > smooth_fac:
            if circ:
                s_tab = smooth(np.hstack([tab_o, tab, tab_o]), smooth_fac)
                s_tab = s_tab[sz:np.size(tab) + sz]
            else:
                s_tab = smooth(tab, smooth_fac)
    else:
        s_tab = tab.copy()

    # ;; Gives all the modes without taking into account the possible
    # ;; noise. Discrimation is done after
    mode = find_mode_intern(np.copy(s_tab), smooth=0, circ=circ)

    siz = np.size(mode['val'])
    final_val = np.zeros(siz)  # replicate(0., siz)
    final_loc = np.zeros(siz, np.int32)  # replicate(0, siz)
    final_beg_loc = np.zeros(siz, np.int32)  # replicate(0, siz)
    final_end_loc = np.zeros(siz, np.int32)  # replicate(0, siz)

    if noise > 0:
        _val = mode['val'][np.argsort(mode['loc'])]
        val_max = np.max(_val)
        noise_abs = noise * val_max / 100.
        h = 0

        # while size(where(s_tab > 0), / dimensions) NE 0 do begin
        while np.count_nonzero(s_tab) > 0:
            mode = find_mode_intern(np.copy(s_tab), smooth=0, circ=circ)

            siz = np.size(mode['val'])
            final_val_tmp = 0.
            final_loc_tmp = 0
            final_beg_loc_tmp = 0
            final_end_loc_tmp = 0

            s_mode_loc = np.argsort(mode['loc'])
            val_max = np.max(mode['val'][s_mode_loc])
            imax = np.argmax(mode['val'][s_mode_loc])
            loc_max = mode['loc'][s_mode_loc][imax]
            beg_loc_max = mode['beg_loc'][s_mode_loc][imax]
            end_loc_max = mode['end_loc'][s_mode_loc][imax]
            final_val_tmp = val_max
            final_loc_tmp = loc_max
            final_beg_loc_tmp = beg_loc_max
            final_end_loc_tmp = end_loc_max

            if siz >= 2:
                # ;; Is this a real new mode or just tiny fluctuations?
                # ;; Let's consider a noise of 10% relatively to the
                # ;; highest maximum

                # ;; Backward
                if imax > 0:
                    val = mode['val'][s_mode_loc][0:imax]
                    loc = mode['loc'][s_mode_loc][0:imax]
                    beg_loc = mode['beg_loc'][s_mode_loc][0:imax]
                    end_loc = mode['end_loc'][s_mode_loc][0:imax]
                    ind = np.size(loc)
                    while ind >= 0:
                        if s_tab[final_beg_loc_tmp - 1] != 0:
                            dif = np.abs(s_tab[loc[ind - 1]] - s_tab[final_beg_loc_tmp])
                            if dif <= noise_abs:
                                # ;; the new mode is extended
                                final_beg_loc_tmp = int(np.floor(beg_loc[ind - 1]))
                                # ;;final_end_loc_tmp = floor(end_loc_max)
                                if ind >= 2:
                                    ind = ind - 1
                                    val = val[0:ind]
                                    loc = loc[0:ind]
                                    end_loc = end_loc[0:ind]
                                    beg_loc = beg_loc[0:ind]
                                else:
                                    break
                            else:
                                break
                        else:
                            break

                if imax < siz - 1:
                    val = mode['val'][s_mode_loc][imax + 1:siz]
                    loc = mode['loc'][s_mode_loc][imax + 1:siz]
                    beg_loc = mode['beg_loc'][s_mode_loc][imax + 1:siz]
                    end_loc = mode['end_loc'][s_mode_loc][imax + 1:siz]
                    ind = 0
                    n = np.size(loc)
                    while n >= 1:
                        if s_tab[final_end_loc_tmp + 1] != 0:
                            dif = abs(s_tab[loc[0]] - s_tab[final_end_loc_tmp])
                            if dif <= noise_abs:
                                # ;; the new mode is extended
                                final_end_loc_tmp = end_loc[0]
                                if n >= 2:
                                    val = val[1:n]
                                    loc = loc[1:n]
                                    end_loc = end_loc[1:n]
                                    beg_loc = beg_loc[1:n]
                                    n = np.size(loc)
                                else:
                                    break
                            else:
                                break
                        else:
                            break

            if np.size(final_end_loc_tmp) > 1:
                final_end_loc_tmp = final_end_loc_tmp[0]
            if np.size(final_beg_loc_tmp) > 1:
                final_beg_loc_tmp = final_beg_loc_tmp[0]

            final_beg_loc[h] = final_beg_loc_tmp
            final_end_loc[h] = final_end_loc_tmp
            final_val[h] = final_val_tmp
            final_loc[h] = final_loc_tmp
            h = h + 1

            # ;; We now repeat the operation for all the remaining modes
            # pt = indgen(final_end_loc_tmp-final_beg_loc_tmp+1)+final_beg_loc_tmp
            pt = np.arange(final_end_loc_tmp - final_beg_loc_tmp + 1) + final_beg_loc_tmp
            if pt[0] > 0:
                if s_tab[pt[0] - 1] > 0:
                    pt = pt[1:np.size(pt)]

            nn = np.size(pt)
            if pt[nn - 1] < np.size(s_tab) - 1:
                if s_tab[pt[nn - 1] + 1] > 0:
                    pt = pt[0:np.size(pt) - 1]
            s_tab[pt] = 0.

        final_beg_loc = final_beg_loc[0:h]
        final_end_loc = final_end_loc[0:h]
        final_val = final_val[0:h]
        final_loc = final_loc[0:h]
    else:
        final_beg_loc = mode.beg_loc
        final_end_loc = mode.end_loc
        final_val = mode.val
        final_loc = mode.loc

    s_final_loc = np.argsort(final_loc)

    # ;; Elements are given by increasing location
    final_beg_loc = final_beg_loc[s_final_loc]
    final_end_loc = final_end_loc[s_final_loc]
    final_val = final_val[s_final_loc]
    final_loc = final_loc[s_final_loc]

    res = dict(val=final_val,
               loc=final_loc,
               beg_loc=final_beg_loc,
               end_loc=final_end_loc)

    if circ:
        ## Arrange outputs:
        wres = np.where((res['loc'] >= sz) & (res['loc'] < 2 * sz))[0]
        if np.size(wres) >= 1:
            final_loc = np.mod(res['loc'][wres], sz)
            final_val = res['val'][wres]
            final_beg_loc = np.mod(res['beg_loc'][wres], sz)
            final_end_loc = np.mod(res['end_loc'][wres], sz)
            final_val = res['val'][wres]
            wres_min = np.argmin(wres)
            wres_max = np.argmax(wres)
            final_beg_loc[0] = np.mod(res['end_loc'][wres[wres_min] - 1], sz)
            final_end_loc[-1] = np.mod(res['beg_loc'][wres[wres_max] + 1], sz)

            res = dict(val=final_val,
                       loc=final_loc,
                       beg_loc=final_beg_loc,
                       end_loc=final_end_loc)
    if minzero == 1:
        res['val'] = res['val'] - 1

    # ;;; To a check
    for a in range(0, np.size(res['val']) - 1):
        if res['end_loc'][a] != res['beg_loc'][a + 1]:
            logger.error('Bug in the system - check the find_mode analysis')
            # print 'Bug in the system - check the find_mode analysis'
            raise AssertionError('Bug in the system - check the find_mode analysis')

    return res


def find_mode_intern(hist, smooth=None, circ=False):
    if smooth > 0:
        hist = smooth(hist, smooth)

    hist_tmp = np.asarray(hist)
    Opt = []
    Oval = []
    cond = np.size(np.argwhere(hist != 0)) > 0
    while cond:  # .ma.masked_equal(hist_tmp,0).count()>0:
        m = max(hist_tmp)
        i = np.argmax(hist_tmp)
        # ;; The max of the mode is found, let's find its extension
        pt_tmp = np.zeros(np.size(hist_tmp), np.int32)  # replicate(0, n_elements(hist_tmp))
        pt_tmp[0] = i
        val_tmp = np.zeros(np.size(hist_tmp))  # replicate(0., n_elements(hist_tmp))
        val_tmp[0] = m

        # ;; Backward
        h = 1
        iloc = i
        out = search_mode(hist_tmp, pt_tmp, val_tmp, -1, h, iloc)
        h = out['h']  # in these case we need to get back the h
        # ;; Reverse order
        val_tmp = invert_tab(out['val'], h)
        pt_tmp = invert_tab(out['pt'], h)
        # ;; Forward
        out = search_mode(hist_tmp, pt_tmp, val_tmp, 1, out['h'], i)
        pt = out['pt'][0:out['h']]
        val = out['val'][0:out['h']]
        Opt.append(pt)
        Oval.append(val)

        if pt[0] > 0:
            if hist_tmp[pt[0] - 1] > 0:
                pt = pt[1:np.size(pt)]

        nn = np.size(pt)
        if pt[nn - 1] < np.size(hist_tmp) - 1:
            if hist_tmp[pt[nn - 1] + 1] > 0:
                pt = pt[0:np.size(pt) - 1]  # -2

        for p in pt:
            hist_tmp[p] = 0.
        cond = np.size(np.argwhere(hist != 0)) > 0

    value = np.zeros(len(Oval))  # replicate(0., Oval->n_elements())
    location = np.zeros(len(Oval), np.int32)  # replicate(0, Oval->n_elements())
    beg_location = np.zeros(len(Oval), np.int32)  # replicate(0, Oval->n_elements())
    end_location = np.zeros(len(Oval), np.int32)  # replicate(0, Oval->n_elements())
    # ;;mean = replicate(0., Oval->n_elements())
    # ;;std_dev = replicate(0., Oval->n_elements())
    # ;;Pearson_md_sk = replicate(0., Oval->n_elements())

    for a in range(0, len(Oval)):
        value[a] = np.max(Oval[a])
        wmax = np.argmax(Oval[a])
        location[a] = Opt[a][wmax]
        beg_location[a] = Opt[a][0]
        end_location[a] = Opt[a][np.size(Opt[a]) - 1]

    # ;; result is sorted by order of location
    s_location = np.argsort(location)
    value2 = value[s_location]
    location2 = location[s_location]
    beg_location2 = beg_location[s_location]
    end_location2 = end_location[s_location]

    out = dict(val=value2,
               loc=location2,
               beg_loc=beg_location2,
               end_loc=end_location2)

    return out


def search_mode(hist, pt, val, increm, h, iloc):
    """
    This function look for all the values and points to associate to a
    mode. Backward if increm = -1 and Forward if increm = +1
    :param hist: ensemble des points non scanné pour ceux différent de zero
    :param pt: indices de l'histogramme appartenant au mode
    :param val: valeur de l'histogramme du mode
    :param increm: sens de recherche
    :param h: position dans le mode
    :param iloc: position dans l'histogramme
    :return:
    """

    cond = ((iloc + increm) >= 0) and ((iloc + increm) <= (np.size(hist) - 1))
    while cond:
        if (hist[iloc + increm] > 0.) and (hist[iloc + increm] <= hist[iloc]):
            pt[h] = iloc + increm
            val[h] = hist[iloc + increm]
            h = h + 1
            iloc = iloc + increm
            cond = ((iloc + increm) >= 0) and ((iloc + increm) <= np.size(hist) - 1)
        else:
            cond = 0
    out = dict(pt=pt, val=val, h=h)
    return out


def invert_tab(tab, i):
    """
    ;; invert the order of the i first values of the tab
    :param tab:
    :param i:
    :return:
    """
    # n = np.size(tab)
    # tab_tmp = np.zeros(n)#replicate(0., n )
    tab_tmp = np.copy(tab)
    tab_tmp[:] = 0
    for r in range(0, i):
        tab_tmp[r] = tab[i - 1 - r]

    return tab_tmp
