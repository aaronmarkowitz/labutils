import numpy as np
import pytest

import synth
from detector import TrapDetector, occupancy

FPS = 13.0
SIGMA_PSF = 20.0


def _run(frames, det):
    states = []
    for i, f in enumerate(frames):
        states.append(det.update(f, i / FPS)['state'])
    return states


def _baseline_and_live(n_live=120, rng=0):
    frames = synth.synthetic_background(30 + n_live, rng=rng)
    return frames[:30], frames[30:]


def _noise_sigma_for_psf(det):
    return det.null_std


def test_bounded_wandering_particle_is_confirmed():
    base, live = _baseline_and_live()
    det = TrapDetector(base, region_center=(150, 150), region_radius=120)
    track = synth.bounded_walk(80, (150, 150), rms_px=25)
    live = synth.inject(live, track, SIGMA_PSF, total=4000.0, start=20)
    st = _run(live, det)
    assert 'candidate' in st and 'confirmed' in st
    assert st.index('candidate') < st.index('confirmed')
    assert all(s in ('empty',) for s in st[:20])


@pytest.mark.control
def test_control_falling_particle_never_becomes_candidate():
    base, live = _baseline_and_live()
    det = TrapDetector(base, region_center=(150, 150), region_radius=120)
    # 30 px/frame = 390 px/s ~ 200 um/s at 0.52 um/px (fast-ish fall, low pressure).
    track = synth.falling_track(40, (150, -20), (0.5, 30.0))
    live = synth.inject(live, track, SIGMA_PSF, total=20000.0, start=10)
    st = _run(live, det)
    assert 'transient' in st          # it IS seen ...
    assert 'candidate' not in st      # ... but not accepted as trapped


@pytest.mark.control
def test_control_slow_drift_through_region_rejected_at_confirm():
    """A slowly sinking particle (high pressure) may pass the 0.5 s candidate stage,
    but its net drift over confirm_s must stop it being confirmed."""
    base, live = _baseline_and_live(n_live=200)
    det = TrapDetector(base, region_center=(150, 150), region_radius=140, max_net_drift_px=40)
    track = synth.falling_track(120, (150, 30), (0.0, 2.0))
    live = synth.inject(live, track, SIGMA_PSF, total=6000.0, start=10)
    assert 'confirmed' not in _run(live, det)


@pytest.mark.control
def test_control_no_particle_no_false_alarm():
    base, live = _baseline_and_live(n_live=600, rng=3)
    det = TrapDetector(base, region_center=(150, 150), region_radius=120)
    st = _run(live, det)
    assert 'candidate' not in st
    assert st.count('transient') < 10


def test_uniform_brightness_step_is_not_a_particle():
    base, live = _baseline_and_live()
    det = TrapDetector(base, region_center=(150, 150), region_radius=120)
    st = _run(live + 15.0, det)   # laser power jump: whole frame brighter
    assert 'candidate' not in st


def test_occupancy_sees_injected_and_not_empty():
    frames = synth.synthetic_background(40, shape=(500, 500), rng=5)
    ok_empty, m0 = occupancy(frames, (250, 250), 60, 255, 0.05)
    track = synth.bounded_walk(40, (250, 250), rms_px=5)
    full = synth.inject(frames, track, SIGMA_PSF, total=10000.0)
    ok_full, m1 = occupancy(full, (250, 250), 60, 255, 0.05)
    assert not ok_empty and ok_full
    assert abs(m1["net_counts"] - 10000.0) < 3 * m1["noise_floor_spatial_counts"]


@pytest.mark.control
def test_control_occupancy_refuses_when_noise_unmeasurable():
    frames = synth.synthetic_background(10, shape=(150, 150), rng=5)
    with pytest.raises(ValueError):
        occupancy(frames, (75, 75), 60, 255, 0.05)
