"""Synthetic and real-data frame scenes for testing the detector without a dropper.

`real_scene(...)` builds an "empty trap" scene from a particle-free crop of real
frames (real noise, real fixed pattern) and a particle cut-out from the same
frames (real PSF, real motion), so a dimmer particle can be injected at any
brightness scale. `synthetic_*` make fully synthetic scenes for unit tests.
"""
import math

import numpy as np

from photometry import aperture_sum


def gaussian_blob(shape, x, y, sigma, total):
    yy, xx = np.indices(shape)
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return g * (total / g.sum())


def synthetic_background(n, shape=(300, 300), level=100.0, read_sigma=3.0, pattern=5.0, rng=0):
    gen = np.random.default_rng(rng)
    fixed = gen.normal(0, pattern, shape)
    return level + fixed + gen.normal(0, read_sigma, (n,) + shape)


def bounded_walk(n, center, rms_px, rng=1, corr=0.8):
    """AR(1) positions around center with the given stationary rms (trapped)."""
    gen = np.random.default_rng(rng)
    xy = np.zeros((n, 2))
    s = rms_px * math.sqrt(1 - corr ** 2)
    for i in range(1, n):
        xy[i] = corr * xy[i - 1] + gen.normal(0, s, 2)
    return xy + np.asarray(center, dtype=float)


def falling_track(n, start, velocity_px_per_frame):
    t = np.arange(n)[:, None]
    return np.asarray(start, dtype=float) + t * np.asarray(velocity_px_per_frame, dtype=float)


def inject(frames, track, sigma, total, start=0):
    """Add a Gaussian particle following `track` into frames[start:start+len(track)]."""
    out = np.array(frames, dtype=float, copy=True)
    for i, (x, y) in enumerate(track):
        j = start + i
        if j >= len(out):
            break
        if -3 * sigma < x < out.shape[2] + 3 * sigma and -3 * sigma < y < out.shape[1] + 3 * sigma:
            out[j] += gaussian_blob(out.shape[1:], x, y, sigma, total)
    return out


def requantize(frame, saturation_level=255, rng=None):
    """Integer camera counts from (integer scene + injected float signal),
    by unbiased stochastic rounding: floor(x + U[0,1)). The real camera adds the
    signal before quantization and its read noise dithers sub-count signals;
    plain rounding would instead delete any injected signal < 0.5 count/px
    (a 1%-of-reference particle is < 0.25 count/px at 50 ms, lights off).
    An integer frame comes back unchanged."""
    rng = np.random.default_rng() if rng is None else rng
    f = np.asarray(frame, dtype=float)
    return np.clip(np.floor(f + rng.random(f.shape)), 0, saturation_level)


def real_scene(stack, particle_xy, scene_xy, scene_half=150, cut_half=110, r_ap=80):
    """Split real frames into (scene, excess): `scene` is a particle-free crop
    centred at scene_xy (scene_half: int, or (half_w, half_h) for a strip ROI);
    `excess` is the particle cut-out (annulus-subtracted,
    per frame) centred at particle_xy, same size as the scene crop.
    Returns (scene, excess, excess_total_counts) -- excess_total is the mean
    net aperture flux of the cut-out, so a scale s injects s * that flux."""
    stack = np.asarray(stack, dtype=float)
    hx, hy = (scene_half, scene_half) if np.isscalar(scene_half) else map(int, scene_half)
    sx, sy = int(scene_xy[0]), int(scene_xy[1])
    scene = stack[:, sy - hy:sy + hy, sx - hx:sx + hx]
    if scene.shape[1:] != (2 * hy, 2 * hx):
        raise ValueError(f"scene crop {scene.shape[1:]} runs off the frame {stack.shape[1:]}")
    px, py = int(round(particle_xy[0])), int(round(particle_xy[1]))
    cut = stack[:, py - cut_half:py + cut_half, px - cut_half:px + cut_half]
    exc = np.zeros_like(scene)
    ox, oy = hx - cut_half, hy - cut_half
    for i, c in enumerate(cut):
        _, bk, _ = aperture_sum(c, cut_half, cut_half, r_ap, r_ap + 10, r_ap + 25)
        e = c - bk
        rr = np.hypot(*np.meshgrid(np.arange(2 * cut_half) - cut_half,
                                   np.arange(2 * cut_half) - cut_half))
        e[rr > r_ap + 25] = 0.0
        exc[i, oy:oy + 2 * cut_half, ox:ox + 2 * cut_half] = e
    tot = float(np.mean([aperture_sum(exc[i], hx, hy, r_ap, r_ap + 10, r_ap + 25)[0]
                         for i in range(len(exc))]))
    return scene, exc, tot
