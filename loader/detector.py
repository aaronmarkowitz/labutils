"""Newly-trapped-particle detector for the loader (numpy only).

Per frame: subtract the empty-trap baseline, box-smooth at about the PSF scale,
and take the maximum inside the trap region. That max is compared with a
threshold calibrated on the baseline frames themselves (leave-one-out), so the
threshold already includes the look-elsewhere effect of searching the region.

A particle falling through the beam also lights up the frame. It is rejected by
demanding BOUNDEDNESS, not stillness: a trapped particle wanders (the 10 pg
cluster moved +-18 px rms in x on 2026-09-28, and thermal rms grows as 1/sqrt(m),
so a nanoparticle wanders more), while a falling one keeps going. So the
detections in the window must stay inside the trap region, with small net
displacement from first to last.

Two stages:
  candidate  -- persistent for persist_s: stop the dropper now (cheap to be wrong)
  confirmed  -- persistent for confirm_s: treat as trapped and measure it
"""
import math
from collections import deque

import numpy as np

from photometry import box_smooth


class TrapDetector:
    def __init__(self, baseline, region_center=None, region_radius=None, smooth_r=12,
                 k_sigma=6.0, persist_s=0.5, confirm_s=3.0, frac_above=0.7,
                 max_net_drift_px=60.0, max_speed_px_s=200.0, gap_reset_frames=4,
                 destripe=True, exclude=None):
        base = np.asarray(baseline, dtype=float)
        if base.ndim != 3 or len(base) < 5:
            raise ValueError("baseline must be a stack of >= 5 frames")
        self.shape = base.shape[1:]
        h, w = self.shape
        self.center = region_center if region_center is not None else (w / 2.0, h / 2.0)
        self.radius = region_radius if region_radius is not None else math.hypot(w, h)
        yy, xx = np.indices(self.shape)
        self.smooth_r = int(smooth_r)
        # Search only where the smoothing box is fully inside the frame: edge
        # pixels average fewer samples, are noisier, and otherwise set the null max.
        b = self.smooth_r
        self.mask = ((np.hypot(xx - self.center[0], yy - self.center[1]) <= self.radius)
                     & (xx >= b) & (xx < w - b) & (yy >= b) & (yy < h - b))
        if exclude is not None:
            # e.g. the x cam's dark-frame corner glow, which otherwise sets the
            # full-frame null max (97 % of frames, 2026-09-29)
            self.mask &= ~np.asarray(exclude, bool)
        if not self.mask.any():
            raise ValueError("trap region has no pixels a full smoothing box away from the edge")
        self.destripe = bool(destripe)
        self.mean = base.mean(0)
        n = len(base)
        total = base.sum(0)
        null = []
        for f in base:
            loo = (total - f) / (n - 1)
            s, _, _ = self._stat(f, loo)
            null.append(s)
        null = np.array(null)
        # Leave-one-out residual variance is n/(n-1) times the true one; rescale.
        self.null_mean = float(null.mean())
        self.null_std = float(null.std(ddof=1) * math.sqrt((n - 1) / n)) or 1e-9
        self.k_sigma = float(k_sigma)
        self.threshold = self.null_mean + self.k_sigma * self.null_std
        self.persist_s = float(persist_s)
        self.confirm_s = float(confirm_s)
        self.frac_above = float(frac_above)
        self.max_net_drift_px = float(max_net_drift_px)
        self.max_speed_px_s = float(max_speed_px_s)
        self.gap_reset_frames = int(gap_reset_frames)
        self.reset()

    def reset(self):
        self.episode = deque()       # (t, above, x, y, stat) since episode start
        self.below_run = 0
        self.state = 'empty'

    def _stat(self, frame, ref):
        d = frame - ref
        if self.destripe:
            d = d - row_offsets(d)[:, None]
        s = box_smooth(d, self.smooth_r)
        # Remove any uniform level change (laser power drift) before the max.
        s = s - np.median(s[self.mask])
        s = np.where(self.mask, s, -np.inf)
        iy, ix = np.unravel_index(np.argmax(s), s.shape)
        return float(s[iy, ix]), float(ix), float(iy)

    def score(self, frame):
        """(sigma_score, x, y) of the brightest in-region excess in one frame."""
        s, x, y = self._stat(np.asarray(frame, dtype=float), self.mean)
        return (s - self.null_mean) / self.null_std, x, y

    def update(self, frame, t):
        """Feed one frame; returns dict(state, score, x, y, duration_s)."""
        z, x, y = self.score(frame)
        above = z >= self.k_sigma
        if above:
            self.below_run = 0
            if not self.episode:
                self.episode.append((t, True, x, y, z))
            else:
                self.episode.append((t, True, x, y, z))
        elif self.episode:
            self.below_run += 1
            self.episode.append((t, False, x, y, z))
            if self.below_run >= self.gap_reset_frames:
                self.reset()
        dur = self.episode[-1][0] - self.episode[0][0] if self.episode else 0.0
        state = 'empty'
        if self.episode:
            state = 'transient'
            if self._persistent(self.persist_s):
                state = 'candidate'
            if self._persistent(self.confirm_s):
                state = 'confirmed'
        self.state = state
        return {'state': state, 'score': z, 'x': x, 'y': y, 'duration_s': dur,
                'threshold_sigma': self.k_sigma}

    def _persistent(self, need_s):
        ep = self.episode
        if ep[-1][0] - ep[0][0] < need_s:
            return False
        ups = [e for e in ep if e[1]]
        if len(ups) < 3 or len(ups) / len(ep) < self.frac_above:
            return False
        # Bounded: every detection inside the region (guaranteed by the mask) and
        # small net drift between the first and last quarter of the episode.
        # Two limits: net drift (catches a slow sinker over confirm_s) and net
        # speed (catches a fast faller within persist_s, before it has drifted far).
        q = max(1, len(ups) // 4)
        t0 = np.mean([e[0] for e in ups[:q]]); t1 = np.mean([e[0] for e in ups[-q:]])
        x0 = np.mean([e[2] for e in ups[:q]]); y0 = np.mean([e[3] for e in ups[:q]])
        x1 = np.mean([e[2] for e in ups[-q:]]); y1 = np.mean([e[3] for e in ups[-q:]])
        d = math.hypot(x1 - x0, y1 - y0)
        speed = d / (t1 - t0) if t1 > t0 else 0.0
        return d <= self.max_net_drift_px and speed <= self.max_speed_px_s

    def describe(self):
        return {'threshold_sigma': self.k_sigma, 'null_mean': self.null_mean,
                'null_std': self.null_std, 'region_center': self.center,
                'region_radius': self.radius, 'smooth_r': self.smooth_r,
                'destripe': self.destripe,
                'persist_s': self.persist_s, 'confirm_s': self.confirm_s}


def row_offsets(d, n_chunks=16):
    """Robust per-row level of a difference frame: the median over column chunks
    of the chunk means. The IDS sensor has row-correlated read noise (row-mean
    std ~0.08 counts vs 0.018 expected for white noise, 2026-09-28 dark frames),
    which dominates a large smoothing box. A particle covering fewer than half
    the chunks of a row does not bias the median."""
    h, w = d.shape
    n = max(1, min(n_chunks, w // 8))
    edges = np.linspace(0, w, n + 1).astype(int)
    chunk = np.stack([d[:, a:b].mean(1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    return np.median(chunk, axis=1)


def occupancy(stack, center, r_ap, saturation_level, exposure_s, k_sigma=5.0, **kw):
    """Is there a particle at `center` in these frames? Uses aperture photometry
    against the annulus, so it needs no empty-trap reference. Returns
    (occupied: bool, photometry dict)."""
    from photometry import measure_stack
    m = measure_stack(stack, exposure_s, saturation_level, center=center, r_ap=r_ap, **kw)
    # Spatial noise (aperture on the MEAN frame at empty positions): with no
    # empty-trap reference, fixed-pattern background is the floor, not shot noise.
    if not m['noise_positions'] or not math.isfinite(m['noise_floor_spatial_counts']):
        raise ValueError("no source-free aperture positions fit in the frame; "
                         "use a larger ROI or a smaller r_ap")
    return bool(m['net_counts'] >= k_sigma * m['noise_floor_spatial_counts']), m
