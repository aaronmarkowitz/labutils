"""Tests for the 2026-09-28 new-configuration changes: stochastic requantization,
difference photometry, the detector edge mask, the 1 um decision rule and --roi."""
import math
import types

import numpy as np
import pytest

import detector as D
import load_particle as lp
import photometry as P
import size_model as sm
import synth


# ------------------------------------------------------------------ requantize

def test_requantize_unbiased_and_integer_frames_unchanged():
    rng = np.random.default_rng(0)
    q = np.full((200, 200), 5.0)
    assert np.array_equal(synth.requantize(q, rng=rng), q)
    out = synth.requantize(q + 0.2, rng=rng)
    assert out.mean() == pytest.approx(5.2, abs=0.01)


@pytest.mark.control
def test_control_plain_rounding_deletes_subcount_signal():
    """Why requantize exists: np.round(q + 0.2) == q, the injected signal is gone."""
    q = np.full((50, 50), 5.0)
    assert np.round(q + 0.2).mean() == 5.0


# -------------------------------------------------------- difference photometry

def _patterned_scene(n, rng):
    """Background with fixed structure (a ramp + blotches) + read noise, quantised."""
    h, w = 300, 900
    yy, xx = np.mgrid[:h, :w]
    pattern = 4.0 + 0.004 * xx + 1.5 * np.exp(-((xx - 450) ** 2 + (yy - 140) ** 2) / (2 * 60 ** 2))
    return np.floor(pattern + rng.normal(0, 0.45, (n, h, w)) + rng.random((n, h, w)))


def test_difference_photometry_recovers_faint_flux_on_fixed_pattern():
    rng = np.random.default_rng(1)
    base = _patterned_scene(30, rng)
    live = _patterned_scene(40, rng)
    blob = synth.gaussian_blob(live.shape[1:], 450, 140, 20.0, total=600.0)
    fr = synth.requantize(live + blob, rng=rng)
    d = P.difference_photometry(fr, base, 0.05, 255, center=(450, 140), r_ap=80)
    assert d['net_counts'] == pytest.approx(600.0, abs=120)
    assert d['significance'] > 5
    assert not d['baseline_stale']


@pytest.mark.control
def test_control_annulus_photometry_biased_by_fixed_pattern():
    """The same injection measured by the annulus alone is off by the pattern."""
    rng = np.random.default_rng(1)
    _patterned_scene(30, rng)
    live = _patterned_scene(40, rng)
    blob = synth.gaussian_blob(live.shape[1:], 450, 140, 20.0, total=600.0)
    fr = synth.requantize(live + blob, rng=rng)
    a = P.measure_stack(fr, 0.05, 255, center=(450, 140), r_ap=80)
    assert abs(a['net_counts'] - 600.0) > 500


@pytest.mark.control
def test_control_no_noise_floor_gives_nan_significance_not_pass():
    rng = np.random.default_rng(2)
    base = _patterned_scene(10, rng)[:, :, 300:600]      # too narrow for any noise aperture
    fr = _patterned_scene(10, rng)[:, :, 300:600]
    d = P.difference_photometry(fr, base, 0.05, 255, center=(150, 140), r_ap=80)
    assert math.isnan(d['significance'])


# ------------------------------------------------------------- detector edges

@pytest.mark.control
def test_control_edge_pixels_excluded_from_search():
    """A large smoothing box is noisier at the frame edge; the search must not go there."""
    rng = np.random.default_rng(3)
    base = rng.normal(5, 0.5, (30, 120, 200))
    d = D.TrapDetector(base, smooth_r=25)
    ys, xs = np.nonzero(d.mask)
    assert xs.min() >= 25 and xs.max() < 175 and ys.min() >= 25 and ys.max() < 95
    with pytest.raises(ValueError):
        D.TrapDetector(base[:, :40, :40], smooth_r=25)


# --------------------------------------------------------------- decision rule

def _args(**kw):
    a = dict(accept_max_nm=1000.0, k_occupancy=5.0)
    a.update(kw)
    return types.SimpleNamespace(**a)


def _entry(d_min_nm, d_max_nm, sig=50.0):
    return {'size_estimate': {'d_min_m': d_min_nm * 1e-9, 'd_max_m': d_max_nm * 1e-9,
                              'd_central_m': d_min_nm * 1e-9},
            'photometry': {'saturated': False}, 'significance': sig}


def test_decide_accepts_all_sub_um_rejects_all_over_um():
    assert lp.decide(_entry(240, 900), _args())[0] == 'accept'
    assert lp.decide(_entry(1100, 3000), _args())[0] == 'reject'
    assert lp.decide(_entry(500, 1500), _args())[0] == 'ambiguous'


@pytest.mark.control
def test_control_decide_insignificant_or_unknown_flux_never_accepts():
    assert lp.decide(_entry(150, 300, sig=3.0), _args())[0] == 'no-particle'
    assert lp.decide(_entry(150, 300, sig=float('nan')), _args())[0] == 'ambiguous'


def test_spread_curve_identity_at_zero_and_fills_dips():
    d = np.geomspace(0.9e-6, 2e-6, 300)
    C = sm.signal_curve(d, pol_deg=0)
    assert np.array_equal(sm.spread_curve(C, d, 0.0), C)
    Cs = sm.spread_curve(C, d, 0.1)
    inner = slice(60, 240)                  # away from the averaging edge
    assert (Cs[inner].min() / np.median(Cs[inner])) > (C[inner].min() / np.median(C[inner]))


# ------------------------------------------------------------------------ ROI

def test_fixed_roi_converts_trap_to_roi_pixels(monkeypatch):
    # The default became --roi auto on 2026-09-29 (test_alignment.py); an explicit
    # strip keeps the 2026-09-28 behaviour.
    seen = {}
    monkeypatch.setattr(lp, 'cmd_ledger', lambda a: seen.setdefault('a', a) and 0)
    monkeypatch.setattr(lp, 'cmd_measure', lambda a: seen.setdefault('a', a) and 0)
    lp.main(['measure', '--roi'] + [str(v) for v in lp.DEFAULT_ROI])
    a = seen['a']
    assert a.roi == lp.DEFAULT_ROI
    assert a.trap_xy == (lp.TRAP_XY[0] - lp.DEFAULT_ROI[0], lp.TRAP_XY[1] - lp.DEFAULT_ROI[1])
    seen.clear()
    lp.main(['measure', '--roi', '0', '0', '0', '0'])
    assert seen['a'].roi is None and seen['a'].trap_xy == lp.TRAP_XY


@pytest.mark.control
def test_control_reference_with_other_aperture_is_not_used():
    import ledger
    rows = [{'id': 'a', 'brightness': 1.0, 'epstein_mass_pg': 10, 'r_ap': 80, 'camera_serial': 's'}]
    assert len(ledger.references(rows, camera_serial='s', r_ap=80)) == 1
    assert ledger.references(rows, camera_serial='s', r_ap=50) == []


def _offset_case(search_r):
    rng = np.random.default_rng(4)
    base = _patterned_scene(30, rng)
    live = _patterned_scene(40, rng)
    blob = synth.gaussian_blob(live.shape[1:], 450, 140, 20.0, total=900.0)
    fr = synth.requantize(live + blob, rng=rng)
    # start 50 px off, as the detector reports the ring PSF's bright side
    return P.difference_photometry(fr, base, 0.05, 255, center=(500, 140), r_ap=50,
                                   search_r=search_r)


def test_difference_photometry_recentres_from_detector_offset():
    d = _offset_case(80)
    assert abs(d['center_xy'][0] - 450) < 10
    assert d['net_counts'] == pytest.approx(900.0, abs=150)


@pytest.mark.control
def test_control_fixed_offset_aperture_loses_flux():
    assert _offset_case(0)['net_counts'] < 600
