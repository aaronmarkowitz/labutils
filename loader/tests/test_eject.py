"""Eject plan + execution order, with mocked EPICS (nothing reaches hardware)."""
import numpy as np
import pytest

import eject as E

C = [0.0199022, -0.170742, -0.492907, 0.973573]     # live spinC column, 2026-09-29
S = [-1.346, -0.789, 1.591, 0.773]                  # live spinS column
FWD = np.array([[0.0314, -0.1312, -0.4221, 0.7898],  # measured A_field (x, y rows)
                [0.2424, 0.1399, -0.2918, -0.1281]])


def _state(**over):
    acts = np.zeros((8, 8))
    acts[:4, 5] = C
    acts[:4, 6] = S
    st = {'acts': acts, 'col_inmon': np.zeros(8), 'poles_gain': np.ones(4),
          'poles_outmon': np.zeros(4),
          'spin': {'FREQ': 5000.0, 'COSGAIN': 0.0, 'SINGAIN': 0.0, 'CLKGAIN': 0.0, 'TRAMP': 5.0},
          'pll': {'FILT_GAIN': 0.0, 'INT_GAIN': 0.0}}
    st.update(over)
    return st


def test_amplitude_fills_binding_dac_and_field_is_uniform():
    p = E.plan_eject(_state(), forward=FWD)
    assert p['problems'] == []
    assert p['binding_electrode'] == 'E3'
    assert max(p['peak_counts_quadrature']) == pytest.approx(E.DAC_SAFE_COUNTS)
    assert p['cosgain_counts'] == pytest.approx(19212, abs=5)
    assert p['field_uniformity']['min_over_max'] > 0.99
    assert any('guard band' in n for n in p['notes'])


@pytest.mark.control
def test_control_nonuniform_field_is_a_problem():
    bad = _state()
    bad['acts'][2, 6] *= 0.5                   # corrupt one spinS coefficient
    p = E.plan_eject(bad, forward=FWD)
    assert any('varies with angle' in m for m in p['problems'])


@pytest.mark.control
def test_control_preconditions_block():
    st = _state()
    st['acts'][6, 5] = 1.0                     # spinC leaking into DAC6 -> LOC loopback
    st['col_inmon'][0] = 500.0                 # XCTL feedback active
    st['pll']['INT_GAIN'] = 1.0
    p = E.plan_eject(st, forward=FWD, other_procs=['measure_actuator_gain.py'])
    txt = ' '.join(p['problems'])
    for key in ('ACTS_7_6', 'column 1', 'PLL', 'running'):
        assert key in txt


@pytest.mark.control
def test_control_execute_refuses_without_allow_and_with_problems():
    p = E.plan_eject(_state(), forward=FWD)
    with pytest.raises(E.EjectDisabled):
        E.execute(p, put=lambda *a: pytest.fail('wrote EPICS'), sleep=lambda s: None)
    p['problems'] = ['x']
    with pytest.raises(E.EjectDisabled):
        E.execute(p, put=lambda *a: pytest.fail('wrote EPICS'), sleep=lambda s: None, allow=True)


def _run(abort_at=None, occupied_seq=None):
    log = []
    state = {'gain': 0.0}

    def put(ch, v):
        if ch.endswith('_FREQ') and state['gain'] and v > 1000:
            raise AssertionError('restored kHz FREQ with the gain up')
        if ch.endswith('_COSGAIN'):
            state['gain'] = v
        log.append((ch.split('_')[-1], v))

    n = {'k': 0}

    def check_abort():
        n['k'] += 1
        if abort_at is not None and n['k'] == abort_at:
            raise KeyboardInterrupt

    occ = iter(occupied_seq or [])
    p = E.plan_eject(_state(), forward=FWD, sweep_s=1.0, n_sweeps=3)
    try:
        out = E.execute(p, put=put, sleep=lambda s: None, check_abort=check_abort,
                        occupied=lambda: next(occ), allow=True, log=lambda *a: None)
    except KeyboardInterrupt:
        out = 'aborted'
    return out, log


@pytest.mark.control
def test_control_ctrl_c_mid_sweep_zeroes_gain_before_restoring_freq():
    out, log = _run(abort_at=5)
    assert out == 'aborted'
    names = [k for k, _ in log]
    i0 = max(i for i, (k, v) in enumerate(log) if k == 'COSGAIN' and v == 0.0)
    assert ('FREQ', 5000.0) in log[i0:] and ('TRAMP', 5.0) in log[i0:]
    assert names.index('COSGAIN') > names.index('FREQ')      # gain came up after FREQ = f_start


def test_sweep_stops_when_trap_empties_and_always_restores():
    out, log = _run(occupied_seq=[True, False])
    assert out == 'ejected'
    freqs = [v for k, v in log if k == 'FREQ']
    assert freqs[0] == pytest.approx(1.0) and max(f for f in freqs if f < 1000) == pytest.approx(60.0)
    assert log[-2:] == [('FREQ', 5000.0), ('TRAMP', 5.0)]


# ------------------------------------------------------------------ gaps + dark run

@pytest.mark.control
def test_control_spin_already_driving_is_a_problem():
    st = _state()
    st['spin']['COSGAIN'] = 100.0
    assert any('already driving' in m for m in E.plan_eject(st, forward=FWD)['problems'])


def test_quadrature_loss_note_names_e3():
    notes = E.plan_eject(_state(), forward=FWD)['notes']
    assert any('E3' in n and '32767' in n for n in notes)


class FakeCDS:
    """SW2 writes XOR-toggle the output bit, like cdsFilt. `stuck` cells ignore it;
    `gain_sticks` makes COSGAIN read back nonzero after safe-off."""

    def __init__(self, stuck=(), gain_sticks=False):
        self.v = {f'{c}_SW2R': 1536.0 for c in E.SPIN_CELLS}
        self.v.update({f'{E.SPIN}_COSGAIN': 0.0, f'{E.SPIN}_SINGAIN': 0.0})
        self.stuck, self.gain_sticks, self.log = set(stuck), gain_sticks, []

    def get(self, ch):
        return self.v[ch]

    def put(self, ch, val):
        self.log.append((ch, val))
        if ch.endswith('_SW2'):
            cell = ch[:-4]
            if cell not in self.stuck:
                self.v[f'{cell}_SW2R'] = float(int(self.v[f'{cell}_SW2R']) ^ int(val))
        elif ch.endswith('_COSGAIN'):
            self.v[ch] = 5.0 if (self.gain_sticks and val == 0) else val
            self.v[f'{E.SPIN}_SINGAIN'] = self.v[ch]           # ezca slaving

    def outputs(self):
        return E.spin_outputs(self.get)


def _dark(cds, **kw):
    p = E.plan_eject(_state(), forward=FWD, sweep_s=1.0, n_sweeps=1)
    return E.dark_run(p, get=cds.get, put=cds.put, sleep=lambda s: None, log=lambda *a: None, **kw)


def test_dark_run_outputs_off_for_every_gain_write_then_restored():
    cds = FakeCDS()
    seen = []
    orig = cds.put

    def put(ch, v):
        if ch.endswith('_COSGAIN') and v:
            seen.append(any(cds.outputs().values()))
        orig(ch, v)
    cds.put = put
    out, _ = _dark(cds)
    assert out == 'still-trapped' and seen == [False]          # gain up only with all 8 OFF
    assert all(cds.outputs().values())                          # restored ON
    i_last_gain = max(i for i, (c, _) in enumerate(cds.log) if c.endswith('_COSGAIN'))
    assert all(c.endswith('_SW2') for c, _ in cds.log[i_last_gain + 1:] if not c.endswith(('_FREQ', '_TRAMP')))


@pytest.mark.control
def test_control_dark_run_refuses_when_a_switch_does_not_take():
    cds = FakeCDS(stuck={E.SPIN_CELLS[3]})
    with pytest.raises(E.EjectDisabled, match='still ON'):
        _dark(cds)
    assert not any(c.endswith('_COSGAIN') for c, _ in cds.log)  # never drove


@pytest.mark.control
def test_control_dark_run_leaves_outputs_off_if_gain_not_zero():
    cds = FakeCDS(gain_sticks=True)
    _dark(cds)
    assert not any(cds.outputs().values())


def test_analyze_dark_quadrature_and_poles():
    fs = {c: 2048.0 for c in E.DARK_RECORD}
    t = np.arange(0, 4, 1 / 2048.0)
    ph = 2 * np.pi * 7.0 * t
    rec = {E.DARK_RECORD[0]: 19212 * np.cos(ph), E.DARK_RECORD[1]: 19212 * np.sin(ph)}
    rec.update({c: np.zeros_like(t) for c in E.DARK_RECORD[2:]})
    a = E.analyze_dark(rec, fs)
    b = a['blocks'][1]
    assert b['amp_cos'] == pytest.approx(19212, rel=1e-3) and b['f_hz'] == pytest.approx(7.0)
    assert abs(b['quad_err_deg']) < 0.5
    assert max(a['poles_max_excursion_counts'].values()) == 0


@pytest.mark.control
def test_control_analyze_dark_sees_lost_quadrature():
    fs = {c: 2048.0 for c in E.DARK_RECORD}
    t = np.arange(0, 2, 1 / 2048.0)
    ph = 2 * np.pi * 7.0 * t
    rec = {E.DARK_RECORD[0]: np.cos(ph), E.DARK_RECORD[1]: np.sin(ph + np.radians(30))}
    assert E.analyze_dark(rec, fs)['blocks'][0]['quad_err_deg'] == pytest.approx(30, abs=1)
