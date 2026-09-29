"""--roi auto (find the trap, don't assume it), re-referencing after a realignment,
circular polarisation, and the x-laser preflight (2026-09-29)."""
import types

import numpy as np
import pytest

import detector as D
import ledger
import load_particle as lp
import size_model as sm


def _args(argv):
    a = lp.build_parser().parse_args(argv)
    a.trap_xy = tuple(a.trap_xy) if a.trap_xy is not None else None
    return lp.resolve_geometry(a)


# ------------------------------------------------------------------ geometry

def test_roi_auto_is_default_live_and_fixed_for_recordings():
    a = _args(['run'])
    assert a.auto_roi and a.roi is None and a.trap_xy is None
    r = _args(['run', '--replay', 'x.npy'])
    assert not r.auto_roi and r.roi == lp.DEFAULT_ROI
    f = _args(['run', '--roi', '0', '500', '1280', '300'])
    assert not f.auto_roi and f.trap_xy == (lp.TRAP_XY[0], lp.TRAP_XY[1] - 500)


def test_strip_roi_centred_clamped_even():
    assert lp.strip_roi(645.7, 1280, 1024) == (0, 496, 1280, 300)
    assert lp.strip_roi(20, 1280, 1024)[1] == 0
    assert lp.strip_roi(1010, 1280, 1024) == (0, 724, 1280, 300)
    for y in np.linspace(0, 1023, 50):
        x0, y0, w, h = lp.strip_roi(y, 1280, 1024)
        assert y0 % 2 == 0 and 0 <= y0 and y0 + h <= 1024
        assert y0 <= y < y0 + h or y < 150 or y > 874


class _FakeSrc:
    def __init__(self):
        self.roi = (0, 0, 1280, 1024)
        self.calls = []

    def set_roi(self, roi):
        self.calls.append(roi)
        self.roi = roi or (0, 0, 1280, 1024)
        return self.roi


def _blob_stack(y, x=700, n=20, h=1024, w=1280, amp=3.0, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    img = 3.5 + (amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 20 ** 2)) if amp else 0)
    return rng.poisson(img, (n, h, w)).astype(float)


def test_full_frame_locate_then_strip_follows_the_particle():
    """A realignment moved the trap to y=250 (was ~640): the brightest object is
    found on the full frame and the strip + cropped baseline follow it."""
    for y_true in (250, 640, 900):
        st = _blob_stack(y_true)
        occ, m = D.occupancy(st, None, 80, 255, 0.05, k_sigma=5)
        assert occ and abs(m['center_xy'][1] - y_true) < 10
        a = types.SimpleNamespace(auto_roi=True, roi=None, trap_xy=None, strip_h=300)
        src = _FakeSrc()
        base = st[:3]
        where, cropped = lp.narrow_to_strip(src, m['center_xy'], base, a, lambda *k, **kw: None)
        assert src.calls[-1] == a.roi and a.roi[3] == 300
        assert cropped.shape == (3, 300, 1280)
        yi, xi = int(where[1]), int(where[0])
        assert cropped[0, yi, xi] == base[0, yi + a.roi[1], xi]      # same pixels
        assert lp.to_full(where, a.roi) == pytest.approx(m['center_xy'])


@pytest.mark.control
def test_control_empty_full_frame_is_not_occupied():
    occ, m = D.occupancy(_blob_stack(500, amp=0), None, 80, 255, 0.05, k_sigma=5)
    assert not occ


def test_full_frame_resets_geometry():
    a = types.SimpleNamespace(auto_roi=True, roi=(0, 496, 1280, 300), trap_xy=(700, 150))
    src = _FakeSrc()
    lp.full_frame(src, a, lambda *k, **kw: None)
    assert src.calls == [None] and a.roi is None and a.trap_xy is None


def test_ref_full_xy_old_and_new_rows():
    assert lp.ref_full_xy({'photometry': {'center_xy': [765, 637]}}) == (765, 637)
    assert lp.ref_full_xy({'roi': [0, 500, 1280, 300],
                           'photometry': {'center_xy': [786, 145.7]}}) == (786, 645.7)
    assert lp.ref_full_xy({'trap_full_xy': [1, 2]}) == (1, 2)


# ------------------------------------------------------------- re-reference

def _row(i, B, **kw):
    return dict(id=i, brightness=B, camera_serial='s', x_laser_on=False, r_ap=80, **kw)


def test_superseded_reference_is_excluded():
    rows = [_row('old', 2.4e5, epstein_mass_pg=10.0, superseded_by='new'),
            _row('new', 1.2e5, epstein_mass_pg=10.0, reference_from='old')]
    refs = ledger.references(rows, camera_serial='s', x_laser_on=False, r_ap=80)
    assert [r[2] for r in refs] == ['new']


@pytest.mark.control
def test_control_without_supersede_both_epochs_mix():
    """Why superseded_by exists: the same particle before/after a realignment
    matches on camera, x-laser state and r_ap, so both would set the scale."""
    rows = [_row('old', 2.4e5, epstein_mass_pg=10.0),
            _row('new', 1.2e5, epstein_mass_pg=10.0, reference_from='old')]
    assert len(ledger.references(rows, camera_serial='s', x_laser_on=False, r_ap=80)) == 2


def test_set_field_marks_superseded(tmp_path):
    p = str(tmp_path / 'l.jsonl')
    ledger.append(_row('old', 2.4e5, epstein_mass_pg=10.0), p)
    ledger.set_field('old', 'superseded_by', 'new', p)
    assert ledger.load(p)[0]['superseded_by'] == 'new'


def test_decide_flags_stale_calibration():
    a = types.SimpleNamespace(k_occupancy=5.0, accept_max_nm=1000.0)
    e = {'significance': 50.0, 'photometry': {'saturated': False},
         'size_estimate': {'d_min_m': 200e-9, 'd_max_m': 400e-9, 'd_central_m': 300e-9},
         'alignment_dy_px': 120.0}
    v, why = lp.decide(e, a)
    assert v == 'accept' and 'STALE' in why
    e['alignment_dy_px'] = 8.0
    assert 'STALE' not in lp.decide(e, a)[1]


# ---------------------------------------------------------------- polarisation

def test_pol_default_is_circular_and_numbers_still_parse():
    assert _args(['run']).pol_deg == ['circular']
    assert _args(['run', '--pol-deg', '0', '90']).pol_deg == [0.0, 90.0]


def test_circular_estimate_between_linear_states():
    refs = [(1.0, 1.758e-6)]
    B = sm.predicted_ratio(300e-9, 1.758e-6, pol_deg='circular')
    est = sm.estimate_diameter(B, refs, pols=('circular',), rel_tol=0.05)
    assert est['d_min_m'] <= 300e-9 <= est['d_max_m']


# ------------------------------------------------------------- x-laser preflight

@pytest.mark.control
@pytest.mark.parametrize('pox,ok,expect', [(5000.0, False, 2), (None, False, 2)])
def test_control_run_refuses_with_x_laser_on_or_unknown(monkeypatch, tmp_path, pox, ok, expect):
    monkeypatch.setattr(lp, 'caget_all', lambda *a, **k: {'pox_inmon': pox})
    monkeypatch.setattr(lp, 'make_source', lambda a: pytest.fail('opened the camera'))
    a = _args(['run', '--out-dir', str(tmp_path)])
    assert lp.cmd_run(a) == expect


# ------------------------------------------------ dark full frame: glow + gradient

def _glow_stack(n=30, seed=1, blob=0.0):
    """Empty dark frame like the IDS x cam's (2026-09-29): left-right gradient plus a
    top-right corner glow, Poisson noise; optional particle at (784, 645)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:1024, :1280].astype(float)
    img = (3.5 + 0.6 * xx / 1280
           + 0.5 * np.exp(-(((xx - 1280) / 220) ** 2 + (yy / 220) ** 2)))   # as measured
    if blob:
        img = img + blob * np.exp(-((xx - 784) ** 2 + (yy - 645) ** 2) / (2 * 20 ** 2))
    return rng.poisson(img, (n, 1024, 1280)).astype(float)


def _gate_args():
    return types.SimpleNamespace(roi=None, trap_xy=None, r_ap=80, k_gate=6.0, k_occupancy=5.0,
                                 region_px=600.0)


_SRC = types.SimpleNamespace(saturation_level=255, exposure_us=50000)


@pytest.mark.control
def test_control_plain_gate_reads_the_corner_glow_as_a_particle():
    occ, m = D.occupancy(_glow_stack(), None, 80, 255, 0.05, k_sigma=5)
    assert occ and m['center_xy'][0] > 1000 and m['center_xy'][1] < 300


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_highpass_gate_empty_is_empty_and_finds_a_faint_particle(seed):
    occ, m = lp.gate_occupancy(_glow_stack(seed=seed), None, _gate_args(), _SRC)
    assert not occ
    occ, m = lp.gate_occupancy(_glow_stack(seed=seed, blob=1.0), None, _gate_args(), _SRC)
    assert occ and abs(m['center_xy'][0] - 784) < 15 and abs(m['center_xy'][1] - 645) < 15


def test_glow_exclude_follows_the_roi():
    assert lp.glow_exclude((1024, 1280), None)[:300, 1000:].all()
    assert lp.glow_exclude((300, 1280), (0, 496, 1280, 300)) is None      # strip below the glow
    m = lp.glow_exclude((300, 1280), (0, 100, 1280, 300))
    assert m[:200, 1000:].all() and not m[200:].any()
