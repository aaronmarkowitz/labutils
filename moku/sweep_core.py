"""Importable core of the Moku piezo-dropper sweep (`sweep.py`).

`sweep.py` used to hold the conf parsing, ramp planning and drive loop at module
level, so nothing could reuse them without connecting to the instrument at
import. They live here now; `sweep.py` is a thin interactive CLI over them, and
`dropper_server.py` (the particle loader's dropper child process) drives the
same loop with a stop event so a detected particle cancels the chirp at once
instead of after the current `sweep duration`.

Only `load_conf` needs PyYAML; everything else is stdlib, so the planning and the
drive loop are testable against `fake_moku.FakeWaveformGenerator` anywhere.
"""
import math
import os
import time
from datetime import datetime

import sweep_limits

DATA_ROOT = os.path.expanduser("~/Dropbox/Microspheres/MAST-QG/worker1/data")

DEFAULTS = {
    'channel': 1,
    'amp': 1,
    'end amp': 2,
    'base frequency': 1,
    'stop frequency': 100,
    'sweep duration': 3,
    'amplitude increment': 1.01,
    'no_pulses': 10,
    'dc offset': 0,
}


def format_ip(addr):
    # IPv6 addresses need brackets in URLs; IPv4 addresses must not have them
    return '[' + addr + ']' if ':' in addr else addr


def load_conf(path):
    """Read a conf_sweep.yaml file into a dict (raw keys as written)."""
    import yaml
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def plan_ramp(loaded_conf):
    """Turn a raw conf dict into the concrete ramp the drive loop will emit.

    Same arithmetic as the original `sweep.py`: when `amplitude increment` is
    given, `no_pulses` is derived from amp/end amp; otherwise the increment is
    derived from `no_pulses`. `max_time` bounds the loop by elapsed time, so the
    loop emits `n_emitted` = one more pulse than `no_pulses` (see
    `sweep_limits.n_pulses_emitted`).
    """
    c = dict(DEFAULTS)
    c.update(loaded_conf or {})
    amp = c['amp']
    end_amp = c['end amp']
    T = c['sweep duration']
    amp_incr = c['amplitude increment']
    if amp_incr == 'None':
        amp_incr = None
    no_pulses = c['no_pulses']
    if amp_incr is not None:
        no_pulses = int(math.ceil(math.log(end_amp / amp) / math.log(amp_incr))) + 1
        max_time = T * no_pulses
    else:
        max_time = no_pulses * T
        amp_incr = (end_amp / amp) ** (1 / (no_pulses - 1))
    dc_offset = float(c.get('dc offset') or 0)
    n_emitted = sweep_limits.n_pulses_emitted(T, max_time)
    amplitudes = sweep_limits.pulse_amplitudes(amp, amp_incr, n_emitted)
    return {
        'amp': amp,
        'channel': int(c['channel']),
        'end amp': end_amp,
        'ip_address': str(c.get('ip_address') or '').strip('[]'),
        'base frequency': float(c['base frequency']),
        'stop frequency': float(c['stop frequency']),
        'sweep duration': T,
        'amplitude increment': amp_incr,
        'no_pulses': no_pulses,
        'dc offset': dc_offset,
        # Derived, not part of the saved conf:
        'max_time': max_time,
        'n_emitted': n_emitted,
        'amplitudes': amplitudes,
        'headroom': sweep_limits.check_headroom(amplitudes, dc_offset),
    }


SAVED_KEYS = ('amp', 'channel', 'end amp', 'ip_address', 'base frequency',
              'stop frequency', 'sweep duration', 'amplitude increment',
              'no_pulses', 'dc offset')


def saved_conf(plan):
    """The conf dict written to the dated data directory (same keys as before)."""
    return {k: plan[k] for k in SAVED_KEYS}


def save_conf_copy(plan, data_root=DATA_ROOT, suffix="_conf_sweep.yaml"):
    """Write the run's conf to `data_root/YYMMDD/<timestamp><suffix>`; return path."""
    import yaml
    data_dir = os.path.join(data_root, datetime.now().strftime('%y%m%d'))
    os.makedirs(data_dir, exist_ok=True)
    file_name = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss") + suffix
    file_path = os.path.join(data_dir, file_name)
    with open(file_path, 'w') as file:
        yaml.dump(saved_conf(plan), file)
    return file_path


def run_sweep(wg, plan, stop_event=None, on_pulse=None, lock=None):
    """Emit the amplitude ramp. Returns the number of pulses started.

    Each pulse is a linear frequency sweep base->stop over `sweep duration`.
    With `stop_event`, the inter-pulse wait is `stop_event.wait(T)`, so a stop
    request returns within one instrument call instead of up to T seconds later;
    the caller then turns the output off (`output_off`). `lock` serialises
    instrument calls against a concurrent `output_off` from another thread.
    `on_pulse(k, amplitude, time_elapsed)` is called after each pulse is armed.
    """
    ch = plan['channel']
    T = plan['sweep duration']
    time_elapsed = 0
    c_amp = plan['amp']
    k = 0
    while time_elapsed <= plan['max_time']:
        if stop_event is not None and stop_event.is_set():
            break
        _call(lock, wg.generate_waveform, channel=ch, type='Sine', amplitude=c_amp,
              frequency=plan['base frequency'], offset=plan['dc offset'])
        _call(lock, wg.set_sweep_mode, channel=ch, source='Internal',
              stop_frequency=plan['stop frequency'], sweep_time=T, trigger_level=0)
        if on_pulse is not None:
            on_pulse(k, c_amp, time_elapsed)
        k += 1
        if stop_event is not None:
            if stop_event.wait(T):
                break
        else:
            time.sleep(T)
        time_elapsed += T
        c_amp = c_amp * plan['amplitude increment']
    return k


def output_off(wg, channel, lock=None):
    """Stop driving the piezo now. `set_defaults()` alone lags (the sweep keeps
    running for a while), so switch the channel's waveform Off first."""
    _call(lock, wg.generate_waveform, channel=channel, type='Off')


def _call(lock, fn, **kw):
    if lock is None:
        return fn(**kw)
    with lock:
        return fn(**kw)
