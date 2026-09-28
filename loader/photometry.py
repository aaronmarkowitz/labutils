"""Aperture photometry of a trapped particle's scattered light (numpy only).

Brightness is reported as background-subtracted counts per second of exposure,
so values taken at different exposures compare directly, and normalised by the
illuminating laser power when it is known (B = counts/s/mW-equivalent).

The background is the median of an annulus around the aperture, measured on the
same frames, so no empty-trap frame is needed (the currently trapped particle
can be measured). The noise floor is measured by placing the same aperture on
source-free positions: that is the per-frame detection limit a dimmer particle
has to beat.
"""
import math

import numpy as np


def box_smooth(img, r):
    """Mean over a (2r+1)^2 box, same shape, edges handled by clipping."""
    if r <= 0:
        return img.astype(float)
    a = np.pad(img.astype(float), r + 1, mode='edge')
    c = a.cumsum(0).cumsum(1)
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    h, w = img.shape
    return s[:h, :w] / (k * k)


def _radius_grid(shape, cx, cy):
    yy, xx = np.indices(shape)
    return np.hypot(xx - cx, yy - cy)


def _crop(img, cx, cy, r):
    """View of img covering a radius-r disc about (cx, cy), plus its origin."""
    h, w = img.shape
    x0 = max(int(math.floor(cx - r)) - 1, 0)
    y0 = max(int(math.floor(cy - r)) - 1, 0)
    x1 = min(int(math.ceil(cx + r)) + 2, w)
    y1 = min(int(math.ceil(cy + r)) + 2, h)
    return img[y0:y1, x0:x1], x0, y0


def find_peak(img, smooth_r=3, exclude=None):
    """(x, y) of the brightest smoothed pixel after removing the image median."""
    s = box_smooth(img - np.median(img), smooth_r)
    if exclude is not None:
        s = np.where(exclude, -np.inf, s)
    iy, ix = np.unravel_index(np.argmax(s), s.shape)
    return float(ix), float(iy), float(s[iy, ix])


def find_blobs(excess_smoothed, thresh, min_sep=15, max_blobs=10):
    """Greedy local maxima above `thresh` separated by >= min_sep px."""
    s = excess_smoothed.copy()
    out = []
    for _ in range(max_blobs):
        iy, ix = np.unravel_index(np.argmax(s), s.shape)
        v = s[iy, ix]
        if not v > thresh:
            break
        out.append((float(ix), float(iy), float(v)))
        r = _radius_grid(s.shape, ix, iy)
        s[r < min_sep] = -np.inf
    return out


def centroid_and_fwhm(img, cx, cy, r):
    """Intensity-weighted centroid and a moment-based FWHM (px) inside radius r
    of a background-subtracted image."""
    img, ox, oy = _crop(img, cx, cy, r)
    rr = _radius_grid(img.shape, cx - ox, cy - oy)
    m = (rr <= r)
    w = np.clip(img[m], 0, None)
    tot = w.sum()
    if tot <= 0:
        return cx, cy, float('nan')
    yy, xx = np.nonzero(m)
    yy = yy + oy
    xx = xx + ox
    x0 = (w * xx).sum() / tot
    y0 = (w * yy).sum() / tot
    var = (w * ((xx - x0) ** 2 + (yy - y0) ** 2)).sum() / tot / 2.0
    return float(x0), float(y0), float(2.3548 * math.sqrt(max(var, 0)))


def aperture_sum(img, cx, cy, r_ap, r_in, r_out):
    """(net_sum, bkg_per_px, n_ap) with annulus-MEAN background.

    Not the median: on 8-bit data the median steps in whole counts, and one count
    times ~10^4 aperture pixels is a spurious 10^4-count jump per frame."""
    img, ox, oy = _crop(img, cx, cy, r_out)
    rr = _radius_grid(img.shape, cx - ox, cy - oy)
    ap = rr <= r_ap
    ann = (rr >= r_in) & (rr <= r_out)
    bkg = float(img[ann].mean()) if ann.any() else 0.0
    n = int(ap.sum())
    return float(img[ap].sum() - bkg * n), bkg, n


DEFAULT_R_AP = 80
"""Fixed aperture radius (px). The x-cam PSF is ~50 px FWHM with wide wings (net
flux still grows past r=110 on 2026-09-28), so a growth-curve radius would differ
between particles. Every particle is unresolved, so the SAME fixed aperture
captures the same fraction of each one's light and ratios stay valid."""


def sky_noise(stack, r_ap, r_in, r_out, avoid_xy, avoid_r, n_pos=12, rng=0):
    """(temporal, spatial, n_positions) 1-sigma of the aperture sum on source-free
    positions, counts/frame. Spatial = rms of the MEAN-frame aperture sums (the
    fixed-pattern floor, relevant with no empty-trap reference).

    Temporal (the detector's floor):

    Each position's mean is removed first, so fixed-pattern background cancels as
    it does for a baseline-subtracting detector; what remains is the per-frame
    noise a new particle's flux must beat."""
    stack = np.asarray(stack, dtype=float)
    n, h, w = stack.shape
    gen = np.random.default_rng(rng)
    pad = r_out + 2
    sig = []
    spatial = []
    tries = 0
    while len(sig) < n_pos and tries < 200 * n_pos and w > 2 * pad and h > 2 * pad:
        tries += 1
        x = gen.uniform(pad, w - pad)
        y = gen.uniform(pad, h - pad)
        if avoid_xy is not None and math.hypot(x - avoid_xy[0], y - avoid_xy[1]) < avoid_r + r_out:
            continue
        sums = np.array([aperture_sum(f, x, y, r_ap, r_in, r_out)[0] for f in stack])
        sig.append(sums.std(ddof=1) if len(sums) > 1 else float('nan'))
        spatial.append(aperture_sum(stack.mean(0), x, y, r_ap, r_in, r_out)[0])
    temporal = float(np.nanmedian(sig)) if sig else float('nan')
    spatial_sigma = float(np.sqrt(np.mean(np.square(spatial)))) if spatial else float('nan')
    return temporal, spatial_sigma, len(sig)


def measure_stack(stack, exposure_s, saturation_level, center=None, r_ap=DEFAULT_R_AP,
                  smooth_r=3, search_mask=None, n_noise=12, rng=0):
    """Photometry of a (n, h, w) stack of raw frames of one particle.

    Returns counts_per_s (net aperture flux / exposure, mean over frames), the
    per-frame flux scatter (real: the particle moves and, if a cluster, rotates),
    the sky noise floor (temporal, same aperture), saturation, centroid motion,
    and FWHM. The aperture is centred on the time-averaged centroid.
    """
    stack = np.asarray(stack, dtype=float)
    mean = stack.mean(0)
    if center is None:
        excl = None if search_mask is None else ~search_mask
        cx, cy, _ = find_peak(mean, smooth_r, exclude=excl)
    else:
        cx, cy = center
    h, w = mean.shape
    r_ap = int(r_ap)
    r_in, r_out = r_ap + 10, r_ap + 25
    b0 = aperture_sum(mean, cx, cy, r_ap, r_in, r_out)[1]
    cx, cy, fwhm = centroid_and_fwhm(mean - b0, cx, cy, r_ap)
    per_frame, cents, fwhms = [], [], []
    for f in stack:
        net_f, bk_f, _ = aperture_sum(f, cx, cy, r_ap, r_in, r_out)
        per_frame.append(net_f)
        sub, ox, oy = _crop(f, cx, cy, r_ap + 2)
        x0, y0, fw = centroid_and_fwhm(box_smooth(sub - bk_f, 1), cx - ox, cy - oy, r_ap)
        x0, y0 = x0 + ox, y0 + oy
        cents.append((x0, y0))
        fwhms.append(fw)
    per_frame = np.array(per_frame)
    cents = np.array(cents)
    net, bkg, n_ap = aperture_sum(mean, cx, cy, r_ap, r_in, r_out)
    crop, ox, oy = _crop(mean, cx, cy, r_ap)
    rr = _radius_grid(crop.shape, cx - ox, cy - oy)
    peak_raw = float(stack[:, oy:oy + crop.shape[0], ox:ox + crop.shape[1]][:, rr <= r_ap].max())
    sky_raw_max = float(np.percentile(stack, 99.9))
    sigma, sigma_sp, n_pos = sky_noise(stack, r_ap, r_in, r_out, (cx, cy), r_ap, n_pos=n_noise, rng=rng)
    clipped = r_out > min(cx, cy, w - 1 - cx, h - 1 - cy)
    return {
        'center_xy': (float(cx), float(cy)), 'fwhm_px': fwhm,
        'fwhm_frame_median_px': float(np.nanmedian(fwhms)),
        'centroid_std_xy_px': (float(cents[:, 0].std()), float(cents[:, 1].std())),
        'r_ap': r_ap, 'r_in': int(r_in), 'r_out': int(r_out), 'n_ap_px': n_ap,
        'aperture_clipped_by_edge': bool(clipped), 'bkg_per_px': bkg,
        'net_counts': net, 'counts_per_s': net / exposure_s,
        'frame_flux_rel_std': float(per_frame.std(ddof=1) / abs(net)) if len(per_frame) > 1 and net else float('nan'),
        'noise_floor_counts': sigma, 'noise_floor_counts_per_s': sigma / exposure_s,
        'noise_positions': n_pos, 'noise_floor_spatial_counts': sigma_sp,
        'snr_frame': net / sigma if sigma > 0 else float('inf'),
        'peak_raw': peak_raw, 'frame_p999_raw': sky_raw_max,
        'saturation_level': int(saturation_level),
        'saturated': bool(peak_raw >= saturation_level),
        'exposure_s': exposure_s, 'n_frames': int(len(stack)),
    }


def difference_photometry(stack, baseline, exposure_s, saturation_level, center=None,
                          r_ap=DEFAULT_R_AP, search_r=80, **kw):
    """measure_stack on (frames - empty-trap baseline mean + its median level).

    The fixed-pattern background that limits annulus photometry cancels; the
    annulus step stays and absorbs a uniform level drift since the baseline.
    Extra keys: 'significance' (net / noise of the mean difference frame),
    'noise_floor_mean_counts', 'annulus_residual_counts' (the annulus level
    relative to the baseline's, times the aperture area) and 'baseline_stale'
    (that residual > 5 x the noise). baseline_stale is a DRIFT INDICATOR, not a
    validity flag: a uniform level drift (0.09 count/px seen over a minute) trips
    it, yet the annulus removes exactly that. Don't gate decisions on it.
    Saturation is judged on the raw frames."""
    stack = np.asarray(stack, dtype=float)
    base = np.asarray(baseline, dtype=float)
    if base.shape[1:] != stack.shape[1:]:
        raise ValueError(f"baseline frame shape {base.shape[1:]} != {stack.shape[1:]}")
    bm = base.mean(0)
    level = float(np.median(bm))
    diff = stack - bm + level
    if center is not None and search_r:
        # The detector reports the ring PSF's bright side, up to ~50 px from the
        # flux centre; re-centre on the aperture-scale flux maximum nearby.
        yy, xx = np.indices(bm.shape)
        near = np.hypot(xx - center[0], yy - center[1]) <= search_r
        cx, cy, _ = find_peak(diff.mean(0), smooth_r=max(1, int(r_ap) // 2), exclude=~near)
        center = (cx, cy)
    m = measure_stack(diff, exposure_s, saturation_level, center=center, r_ap=r_ap, **kw)
    cx, cy = m['center_xy']
    crop, ox, oy = _crop(stack.mean(0), cx, cy, m['r_ap'])
    rr = _radius_grid(crop.shape, cx - ox, cy - oy)
    raw = stack[:, oy:oy + crop.shape[0], ox:ox + crop.shape[1]][:, rr <= m['r_ap']]
    m['peak_raw'] = float(raw.max())
    m['saturated'] = bool(m['peak_raw'] >= saturation_level)
    n, nb = len(stack), len(base)
    floor = m['noise_floor_counts'] * math.sqrt(1.0 / n + 1.0 / nb)
    if math.isfinite(m['noise_floor_spatial_counts']):
        floor = max(floor, m['noise_floor_spatial_counts'])
    resid = (m['bkg_per_px'] - level) * m['n_ap_px']
    m.update({'method': 'difference', 'baseline_level': level, 'n_baseline': nb,
              'noise_floor_mean_counts': floor,
              # no source-free positions -> no floor -> NaN, never a pass
              'significance': (m['net_counts'] / floor if math.isfinite(floor) and floor > 0
                               else float('nan')),
              'annulus_residual_counts': float(resid),
              'baseline_stale': bool(abs(resid) > 5 * floor)})
    return m


def exposure_ladder(grab, start_us, min_us=100, max_us=500000, n=10,
                    target_frac=0.6, max_steps=8, **measure_kw):
    """Find the longest unsaturated exposure and measure there.

    grab(exposure_us, n) -> (stack, meta) with meta['exposure_us'] (actual) and
    meta['saturation_level']. Steps x0.5 while saturated, x2 while the peak is
    below target_frac of full scale (and still unsaturated). Returns (best, rungs)
    where rungs lists every measurement (for a linearity check: counts_per_s at
    adjacent unsaturated rungs should agree).
    """
    rungs = []
    exp = float(start_us)
    center = measure_kw.pop('center', None)
    best = None
    for _ in range(max_steps):
        stack, meta = grab(exp, n)
        res = measure_stack(stack, meta['exposure_us'] * 1e-6, meta['saturation_level'],
                            center=center, **measure_kw)
        center = res['center_xy']
        res['exposure_us'] = meta['exposure_us']
        rungs.append(res)
        if res['saturated']:
            if exp <= min_us:
                break
            exp = max(min_us, exp / 2)
            continue
        if best is None or res['exposure_us'] > best['exposure_us']:
            best = res
        headroom = res['peak_raw'] / res['saturation_level']
        if headroom < target_frac / 2 and exp < max_us:
            nxt = min(max_us, exp * 2)
            if any(r['saturated'] and r['exposure_us'] <= nxt for r in rungs):
                break
            exp = nxt
            continue
        break
    unsat = sorted([r for r in rungs if not r['saturated']], key=lambda r: r['exposure_us'])
    lin = None
    if len(unsat) >= 2:
        a, b = unsat[-2], unsat[-1]
        lin = b['counts_per_s'] / a['counts_per_s'] if a['counts_per_s'] else None
    if best is not None:
        best['linearity_ratio'] = lin
    return best, rungs
