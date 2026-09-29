"""load_particle.py eject: camera in the main thread, execute() in a worker.
Fake camera + fake EPICS; nothing reaches hardware."""
import time

import numpy as np
import pytest

import eject as E
import load_particle as lp
import sources
from test_eject import FWD, _state


def _src(fail_after=None):
    rng = np.random.default_rng(0)
    h, w = 300, 900
    yy, xx = np.mgrid[:h, :w]
    scene = rng.poisson(3.0, (40, h, w)).astype(float)
    blob = 40.0 * np.exp(-((xx - 450) ** 2 + (yy - 150) ** 2) / (2 * 15 ** 2))
    s = sources.ReplaySource(scene, np.repeat(blob[None], 40, 0), 50000, 255, realtime=False,
                             occupied_at_start=True, schedule=(1.0,))
    s.roi = (0, 0, w, h)
    if fail_after is not None:
        n = {'k': 0}
        nxt = s._next

        def _next():
            n['k'] += 1
            if n['k'] > fail_after:
                raise TimeoutError('camera died')
            return nxt()
        s._next = _next
    return s


def _setup(monkeypatch, tmp_path, src, eject_on_gain=True):
    plan = E.plan_eject(_state(), forward=FWD, sweep_s=0.3, n_sweeps=2)
    log = []

    def put(ch, v):
        log.append((ch.split('_')[-1], v))
        if ch.endswith('_COSGAIN') and v and eject_on_gain:
            src.on_eject()

    real = E.execute
    monkeypatch.setattr(E, 'execute', lambda p, **kw: real(p, put=put, sleep=lambda s: time.sleep(min(s, 0.005)),
                                                          **kw))
    monkeypatch.setattr(lp, 'live_eject_plan', lambda a: plan)
    monkeypatch.setattr(lp, 'make_source', lambda a: src)
    monkeypatch.setattr(lp, 'caget_all', lambda *a, **k: {'cos': 0.0, 'sin': 0.0})
    a = lp.build_parser().parse_args(['eject', '--go', '--no-preview', '--r-ap', '50',
                                      '--n-baseline', '10', '--out-dir', str(tmp_path)])
    a.trap_xy = tuple(a.trap_xy)
    lp.resolve_geometry(a)
    return a, log


def test_eject_command_ejects_and_verifies_on_camera(monkeypatch, tmp_path, capsys):
    src = _src()
    a, log = _setup(monkeypatch, tmp_path, src)
    assert lp.cmd_eject(a) == 0
    out = capsys.readouterr().out
    assert 'EJECTED' in out and "execute -> ejected" in out
    i0 = max(i for i, (k, v) in enumerate(log) if k == 'COSGAIN' and v == 0)
    assert ('FREQ', 5000.0) in log[i0:]
    assert (tmp_path / 'eject_timeseries.npz').exists()


@pytest.mark.control
def test_control_camera_failure_stops_the_drive(monkeypatch, tmp_path):
    src = _src(fail_after=25)            # dies during the sweep (baseline uses 10)
    a, log = _setup(monkeypatch, tmp_path, src, eject_on_gain=False)
    with pytest.raises(TimeoutError):
        lp.cmd_eject(a)
    assert log[-3:] == [('COSGAIN', 0.0), ('FREQ', 5000.0), ('TRAMP', 5.0)]
    n_freq = sum(1 for k, v in log if k == 'FREQ' and v < 1000)
    assert n_freq < 2 * len(E.sweep_frequencies(1, 60, 0.3, 0.1)) + 1     # did not finish both sweeps


@pytest.mark.control
def test_control_refuses_without_a_particle(monkeypatch, tmp_path):
    src = _src()
    src._present = None
    a, log = _setup(monkeypatch, tmp_path, src)
    assert lp.cmd_eject(a) == 2 and log == []


@pytest.mark.control
def test_control_sigterm_mid_sweep_runs_safe_off(monkeypatch, tmp_path):
    import os
    import signal
    import threading
    src = _src()
    a, log = _setup(monkeypatch, tmp_path, src, eject_on_gain=False)
    plan = E.plan_eject(_state(), forward=FWD, sweep_s=5.0, n_sweeps=3)    # long enough to interrupt
    monkeypatch.setattr(lp, 'live_eject_plan', lambda a: plan)
    old = signal.getsignal(signal.SIGTERM)
    try:
        threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
        lp.cmd_eject(a)
    finally:
        signal.signal(signal.SIGTERM, old)
    assert log[-3:] == [('COSGAIN', 0.0), ('FREQ', 5000.0), ('TRAMP', 5.0)]
    assert sum(1 for k, v in log if k == 'FREQ' and v < 1000) < 3 * plan['n_steps']
