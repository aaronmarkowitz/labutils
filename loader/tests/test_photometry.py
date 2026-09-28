import numpy as np
import pytest

import photometry as P
import synth


def test_aperture_recovers_injected_flux_and_scales_with_exposure():
    frames = synth.synthetic_background(20, rng=2)
    track = synth.bounded_walk(20, (150, 150), rms_px=3)
    full = synth.inject(frames, track, 12.0, total=8000.0)
    m = P.measure_stack(full, 0.05, 255, center=(150, 150), r_ap=40)
    assert m['net_counts'] == pytest.approx(8000.0, rel=0.05)
    assert m['counts_per_s'] == pytest.approx(8000.0 / 0.05, rel=0.05)
    assert not m['saturated']


def test_annulus_mean_not_median_on_8bit():
    """Quantised background: median steps by a whole count, the mean does not."""
    rng = np.random.default_rng(0)
    frames = np.round(100.4 + rng.normal(0, 0.3, (30, 200, 200)))
    m = P.measure_stack(frames, 0.05, 255, center=(100, 100), r_ap=40)
    # true net is 0; a median background (100) would give +0.4*5000 = +2000
    assert abs(m['net_counts']) < 200


def test_exposure_ladder_backs_off_saturation():
    base = synth.synthetic_background(1, rng=0)[0] * 0 + 10.0

    def grab(exp_us, n):
        s = exp_us / 1e4
        img = base * s + synth.gaussian_blob(base.shape, 150, 150, 6, 30000.0 * s)
        stack = np.clip(np.repeat(img[None], n, 0), 0, 255)
        return stack, {'exposure_us': exp_us, 'saturation_level': 255}

    best, rungs = P.exposure_ladder(grab, 80000, n=3, r_ap=40, center=(150, 150))
    assert rungs[0]['saturated']
    assert best is not None and not best['saturated']
    assert best['counts_per_s'] == pytest.approx(3.0e6, rel=0.05)


@pytest.mark.control
def test_control_saturation_flag_trips():
    img = np.full((3, 100, 100), 250.0)
    img[:, 50, 50] = 255
    m = P.measure_stack(img, 0.01, 255, center=(50, 50), r_ap=10, n_noise=2)
    assert m['saturated']
