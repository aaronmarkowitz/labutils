"""Electrical eject: a rotating E-field swept 1 -> 60 Hz -- PLAN + DARK RUN ONLY.

Alma's spec (2026-09-29): sweep 1-60 Hz at max drive, E rotating in the x-y
plane, normalised in magnitude vs angle through the measured actuator gains,
<= 32000 counts on any one DAC. NOT to be run until she says so: `execute`
raises EjectDisabled unless called with allow=True, and no CLI path passes it.

How the drive is made (read from y1dmd.mdl and the live ACTS, 2026-09-29):
  PARTICLE_SPIN is a cdsOsc: SIN -> ACTS column 7 (spinS), COS -> column 6
  (spinC). The model slaves SPIN_SINGAIN to SPIN_COSGAIN (ezca), so one caput
  of COSGAIN = A sets equal quadrature amplitudes. upload_actuation_matrix.py
  wrote columns 6/7 as field-normalised unit vectors (+x, -y), so electrode e
  sees A*(C_e cos wt + S_e sin wt) and |E| = A (anchored field unit) at every
  angle -- `field_uniformity` re-checks this against the measured forward
  matrix. The sweep steps SPIN_FREQ (log in f).

DAC budget: POLES_E* are pass-through, so DAC_e = POLES_GAIN_e * ACTS row e.
With SIN/COS in quadrature, peak_e = A*hypot(C_e, S_e); E3 binds today
(1.666 -> A ~ 19200, Alma's "~20000"). If quadrature were lost (unverified
oscillator behaviour on a FREQ change) the worst case is A*(|C_e|+|S_e|) --
reported, and to be checked by the dark run before any real eject.

Guard band: the sweep crosses 10-20 Hz by Alma's explicit spec. The only
consumer of that band is measure_actuator_gain.py's guard monitor (it must
not be running); the eject's loss check is the x-cam occupancy test.

Ordering (safety): TRAMP short -> FREQ = f_start -> wait > TRAMP -> ramp
COSGAIN up -> sweep. On exit/abort ALWAYS COSGAIN -> 0, wait > TRAMP, and only
then restore FREQ/TRAMP: if FREQ ramps over TRAMP on this RCG version,
restoring 5000 Hz with the gain up would sweep the electrodes through kHz.

Dark run (`--dark-run`, needs Alma's go; writes CDS but drives no electrode):
the 8 spin cells ACTS_{1..4}_{6,7} get their OUTPUT switch turned off and read
back (refuses unless all 8 read off), then the real execute() sequence runs
while NDS2 records the oscillator (ACTS_1_6_IN1 = COS, ACTS_1_7_IN1 = SIN) and
the electrode drives (POLES_E*_OUT_DQ, which must stay static). It answers what
the plan cannot: SIN/COS quadrature and amplitude through FREQ steps, whether
SINGAIN really follows COSGAIN, and whether FREQ ramps over TRAMP. The outputs
are restored only after COSGAIN (and SINGAIN) read back 0; if they do not, the
cells are left OFF (the safe state) and it says so.
"""
import math
import os
import subprocess
import threading
import time

import numpy as np

GUARD_BAND_HZ = (10.0, 20.0)   # trap-loss guard band (CLAUDE.md); waived for eject
DAC_SAFE_COUNTS = 32000
PREFIX = 'Y1:DMD'
SPIN = f'{PREFIX}-PARTICLE_SPIN'
COL_SPINC, COL_SPINS = 6, 7
ELECTRODE_ROWS = (1, 2, 3, 4)
NON_ELECTRODE_ROWS = (5, 6, 7, 8)   # LASER_CTLX, LASER_CTLZ, DAC6, DAC7 (DAC6/7 -> LOC/LOS inputs)
N_COLS = 8
SPIN_CELLS = tuple(f'{PREFIX}-ACTS_{r}_{c}' for r in ELECTRODE_ROWS for c in (COL_SPINC, COL_SPINS))
SW2_OUTPUT_ON = 1024          # cdsFilt SW2 bit 10 (measure_actuator_gain_config.yml)
DARK_RECORD = ((f'{PREFIX}-ACTS_1_6_IN1', f'{PREFIX}-ACTS_1_7_IN1')
               + tuple(f'{PREFIX}-POLES_E{e}_OUT_DQ' for e in ELECTRODE_ROWS))
ABORT_FILE = '/tmp/abort_eject'
WAIVER = ("guard band 10-20 Hz crossed deliberately (Alma, 2026-09-29): the eject aims at trap "
          "loss and its loss check is the x-cam; measure_actuator_gain.py's guard monitor must not run")


class EjectDisabled(RuntimeError):
    pass


def caget(ch):
    out = subprocess.run(['caget', '-t', ch], capture_output=True, text=True, timeout=10)
    if out.returncode != 0:
        raise RuntimeError(f"caget {ch} failed: {out.stderr.strip()}")
    return float(out.stdout.strip())


def caput(ch, value):
    out = subprocess.run(['caput', ch, str(value)], capture_output=True, text=True, timeout=10)
    if out.returncode != 0:
        raise RuntimeError(f"caput {ch} failed: {out.stderr.strip()}")


def read_state(get=caget):
    """Everything the plan needs, read-only."""
    acts = np.array([[get(f'{PREFIX}-ACTS_{r}_{c}_GAIN') for c in range(1, N_COLS + 1)]
                     for r in range(1, 9)])
    col_in = np.array([get(f'{PREFIX}-ACTS_1_{c}_INMON') for c in range(1, N_COLS + 1)])
    poles_gain = np.array([get(f'{PREFIX}-POLES_E{e}_GAIN') for e in ELECTRODE_ROWS])
    poles_out = np.array([get(f'{PREFIX}-POLES_E{e}_OUTMON') for e in ELECTRODE_ROWS])
    spin = {k: get(f'{SPIN}_{k}') for k in ('FREQ', 'COSGAIN', 'SINGAIN', 'CLKGAIN', 'TRAMP')}
    pll = {k: get(f'{PREFIX}-PARTICLE_PLL_{k}') for k in
           ('FILT_GAIN', 'INT_GAIN', 'VCO_CLKGAIN', 'VCO_COSGAIN', 'VCO_SINGAIN')}
    return {'acts': acts, 'col_inmon': col_in, 'poles_gain': poles_gain,
            'poles_outmon': poles_out, 'spin': spin, 'pll': pll}


def load_forward_matrix(config=None):
    """Measured field-normalised forward matrix A_field (dofs x electrodes) via
    upload_actuation_matrix.assemble. Returns None if it cannot be loaded here
    (e.g. the thorcam env has no yaml/h5py)."""
    import contextlib
    import io
    import os
    import sys
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts', 'dipole')
    try:
        sys.path.insert(0, d)
        import yaml
        import upload_actuation_matrix as U
        cfg = yaml.safe_load(open(config or os.path.join(d, 'upload_actuation_matrix_config.yml')))
        with contextlib.redirect_stdout(io.StringIO()):
            r = U.assemble(cfg)
        return np.asarray(r['A_field'])
    except Exception as e:                                  # noqa: BLE001 -- reported, not fatal
        return f"unavailable: {e!r}"


def field_uniformity(forward, C, S, n=361):
    """|E(phi)| over one rotation for electrode vectors C (cos) and S (sin):
    returns (min/max ratio, mean |E| per unit A, rotation degrees per cycle)."""
    ph = np.linspace(0, 2 * np.pi, n)
    E = np.array([forward @ (C * math.cos(p) + S * math.sin(p)) for p in ph])
    m = np.hypot(E[:, 0], E[:, 1])
    ang = np.unwrap(np.arctan2(E[:, 1], E[:, 0]))
    return float(m.min() / m.max()), float(m.mean()), float(np.degrees(ang[-1] - ang[0]))


def sweep_frequencies(f_start, f_stop, sweep_s, step_s):
    n = max(2, int(round(sweep_s / step_s)) + 1)
    return list(np.geomspace(f_start, f_stop, n))


def plan_eject(state, forward=None, f_start=1.0, f_stop=60.0, sweep_s=30.0, step_s=0.1,
               amp_counts=20000.0, dac_safe=DAC_SAFE_COUNTS, n_sweeps=3, tramp_s=2.0,
               inmon_tol=1.0, uniformity_min=0.95, other_procs=None):
    """The drive that would be applied, the budget behind its amplitude, and
    every precondition violation (`problems` non-empty => must not run)."""
    problems, notes = [], []
    acts = np.asarray(state['acts'], float)
    C = acts[[r - 1 for r in ELECTRODE_ROWS], COL_SPINC - 1]
    S = acts[[r - 1 for r in ELECTRODE_ROWS], COL_SPINS - 1]
    if f_start < GUARD_BAND_HZ[1] and f_stop > GUARD_BAND_HZ[0]:
        notes.append(WAIVER)
    for r in NON_ELECTRODE_ROWS:
        for c in (COL_SPINC, COL_SPINS):
            if abs(acts[r - 1, c - 1]) > 0:
                problems.append(f"ACTS_{r}_{c} = {acts[r - 1, c - 1]:g}: the spin drive would leave "
                                f"the electrodes (row {r} is a laser or DAC6/7->LOC/LOS loopback row)")
    others = [c for c in range(1, N_COLS + 1) if c not in (COL_SPINC, COL_SPINS)]
    for c in others:
        v = float(state['col_inmon'][c - 1])
        if abs(v) > inmon_tol:
            problems.append(f"ACTS column {c} input is {v:g} counts (feedback/LO active): it shares the "
                            f"electrode DACs; turn it off or re-budget")
    for k, v in state['pll'].items():
        if v:
            problems.append(f"PARTICLE_PLL_{k} = {v:g}: the PLL demodulates spinS; zero it first")
    if state['spin']['COSGAIN'] or state['spin']['SINGAIN']:
        problems.append(f"PARTICLE_SPIN already driving (COSGAIN {state['spin']['COSGAIN']:g}); "
                        f"refusing to take it over")
    if other_procs:
        problems.append(f"running: {other_procs} (its guard monitor watches 10-20 Hz)")
    pg = np.asarray(state['poles_gain'], float)
    static = np.abs(np.asarray(state['poles_outmon'], float))
    per_A_quad = np.abs(pg) * np.hypot(C, S)
    per_A_worst = np.abs(pg) * (np.abs(C) + np.abs(S))
    with np.errstate(divide='ignore'):
        a_lim = np.where(per_A_quad > 0, (dac_safe - static) / per_A_quad, np.inf)
    A = float(min(amp_counts, a_lim.min()))
    bind = int(np.argmin(a_lim)) + 1
    if not A > 0:
        problems.append("no DAC headroom for any drive")
    peak_quad = static + A * per_A_quad
    peak_worst = static + A * per_A_worst
    if peak_worst.max() > 32767:
        notes.append(f"if SIN/COS lost quadrature, E{int(np.argmax(peak_worst)) + 1} would reach "
                     f"{peak_worst.max():.0f} counts (DAC clips at 32767) -- the dark run checks this")
    uni = None
    if isinstance(forward, np.ndarray):
        ratio, mean_per_A, rot = field_uniformity(forward, C, S)
        uni = {'min_over_max': ratio, 'field_per_count': mean_per_A, 'rotation_deg_per_cycle': rot}
        if ratio < uniformity_min:
            problems.append(f"|E| varies with angle (min/max {ratio:.3f} < {uniformity_min}): ACTS "
                            f"spin columns do not match the measured actuator gains")
    else:
        notes.append(f"field-uniformity check skipped (forward matrix {forward})")
    freqs = sweep_frequencies(f_start, f_stop, sweep_s, step_s)
    return {'drive': 'PARTICLE_SPIN rotating field', 'cosgain_counts': A,
            'binding_electrode': f'E{bind}', 'electrode_C': C.tolist(), 'electrode_S': S.tolist(),
            'peak_counts_quadrature': peak_quad.tolist(), 'peak_counts_if_quadrature_lost': peak_worst.tolist(),
            'static_counts': static.tolist(), 'f_start_hz': f_start, 'f_stop_hz': f_stop,
            'sweep_s': sweep_s, 'step_s': step_s, 'n_steps': len(freqs), 'n_sweeps': n_sweeps,
            'tramp_s': tramp_s, 'field_uniformity': uni, 'spin_snapshot': dict(state['spin']),
            'stop_when': 'x-cam occupancy reports the trap empty (checked after each sweep)',
            'problems': problems, 'notes': notes}


def _safe_off(put, sleep, plan):
    """Gain to zero FIRST, wait out the ramp, then restore frequency and ramp time."""
    snap = plan['spin_snapshot']
    put(f'{SPIN}_COSGAIN', 0.0)
    sleep(plan['tramp_s'] + 1.0)
    put(f'{SPIN}_FREQ', snap['FREQ'])
    put(f'{SPIN}_TRAMP', snap['TRAMP'])


def execute(plan, put=caput, sleep=time.sleep, occupied=None, check_abort=None, allow=False,
            log=print):
    """Run the eject. Refuses unless allow=True (no CLI passes it until Alma
    approves). Returns 'ejected' | 'still-trapped'. Always ends in _safe_off."""
    if not allow:
        raise EjectDisabled("electrical eject is disabled until Alma says so (2026-09-29); "
                            "planned drive: " + repr({k: plan[k] for k in
                                                      ('cosgain_counts', 'f_start_hz', 'f_stop_hz',
                                                       'sweep_s', 'n_sweeps')}))
    if plan['problems']:
        raise EjectDisabled("preconditions failed: " + '; '.join(plan['problems']))
    freqs = sweep_frequencies(plan['f_start_hz'], plan['f_stop_hz'], plan['sweep_s'], plan['step_s'])
    try:
        put(f'{SPIN}_TRAMP', plan['tramp_s'])
        put(f'{SPIN}_FREQ', freqs[0])
        sleep(plan['tramp_s'] + 1.0)
        put(f'{SPIN}_COSGAIN', plan['cosgain_counts'])
        sleep(plan['tramp_s'] + 0.5)
        for k in range(plan['n_sweeps']):
            log(f"eject sweep {k + 1}/{plan['n_sweeps']}: {freqs[0]:.1f} -> {freqs[-1]:.1f} Hz")
            for f in freqs:
                if check_abort:
                    check_abort()
                put(f'{SPIN}_FREQ', f)
                sleep(plan['step_s'])
            if occupied is not None and not occupied():
                return 'ejected'
        return 'still-trapped'
    finally:
        _safe_off(put, sleep, plan)


def spin_outputs(get=caget):
    """{cell: OUTPUT switch on?} for the 8 spin cells (SW2R bit 10)."""
    return {c: bool(int(float(get(f'{c}_SW2R'))) & SW2_OUTPUT_ON) for c in SPIN_CELLS}


def _set_outputs(want, get, put):
    """SW2 writes XOR-toggle the bit, so toggle only the cells not already in
    the wanted state (want: {cell: bool})."""
    now = spin_outputs(get)
    for c, on in want.items():
        if now[c] != on:
            put(f'{c}_SW2', SW2_OUTPUT_ON)


class NDSRecorder:
    """Stream channels over NDS2 in a thread (test points need a live stream).
    Fast channels are block-averaged to <= out_fs (the spin tone is <= 60 Hz)."""

    def __init__(self, channels=DARK_RECORD, server='192.168.1.11', port=8088, out_fs=2048):
        self.channels, self.server, self.port, self.out_fs = list(channels), server, port, out_fs
        self.data = {c: [] for c in self.channels}
        self.fs, self.error, self.gps0 = {}, None, None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            import nds2
            conn = nds2.connection(self.server, self.port)
            for bufs in conn.iterate(self.channels):
                for b in bufs:
                    x = np.asarray(b.data, float)
                    fs = b.channel.sample_rate
                    k = max(1, int(fs // self.out_fs))
                    self.fs[b.channel.name] = fs / k
                    self.data[b.channel.name].append(x[:len(x) // k * k].reshape(-1, k).mean(1))
                    if self.gps0 is None:
                        self.gps0 = b.gps_seconds
                self._ready.set()
                if self._stop.is_set():
                    return
        except Exception as e:      # surfaced in the report; the run itself is unaffected
            self.error = repr(e)
            self._ready.set()

    def start(self, timeout=30.0):
        self._th.start()
        self._ready.wait(timeout)
        if self.error:
            raise RuntimeError(f"NDS2 recorder failed to start: {self.error}")

    def stop(self):
        self._stop.set()
        self._th.join(10)
        return {c: np.concatenate(v) if v else np.zeros(0) for c, v in self.data.items()}


def analyze_dark(rec, fs, block_s=1.0):
    """Per block: COS/SIN amplitude (peak), tone frequency, quadrature error, and
    the max excursion of each electrode drive from its pre-run level."""
    c = rec[DARK_RECORD[0]]
    sn = rec[DARK_RECORD[1]]
    f = fs[DARK_RECORD[0]]
    n = int(block_s * f)
    rows = []
    for i in range(0, min(len(c), len(sn)) - n + 1, n):
        a, b = c[i:i + n], sn[i:i + n]
        amp_c, amp_s = math.sqrt(2) * a.std(), math.sqrt(2) * b.std()
        spec = np.abs(np.fft.rfft((a - a.mean()) * np.hanning(n)))
        f0 = float(np.fft.rfftfreq(n, 1 / f)[int(np.argmax(spec[1:])) + 1])
        # quadrature: correlation of normalised COS and SIN is cos(phase diff)
        rho = float(np.mean((a - a.mean()) * (b - b.mean())) / (a.std() * b.std() + 1e-30))
        rows.append({'t_s': i / f, 'amp_cos': amp_c, 'amp_sin': amp_s, 'f_hz': f0,
                     'quad_err_deg': math.degrees(math.asin(max(-1, min(1, rho))))})
    poles = {}
    for ch in DARK_RECORD[2:]:
        x = rec.get(ch)
        if x is not None and len(x):
            m = int(fs[ch] * 2) or 1           # first 2 s: before the gain ramps
            poles[ch] = float(np.max(np.abs(x - np.median(x[:m]))))
    return {'blocks': rows, 'poles_max_excursion_counts': poles}


def dark_run(plan, get=caget, put=caput, sleep=time.sleep, recorder=None, check_abort=None,
             log=print):
    """Run the real execute() with the 8 spin cells' outputs OFF (see module doc).
    Returns (execute result, recording dict or None)."""
    if plan['problems']:
        raise EjectDisabled("preconditions failed: " + '; '.join(plan['problems']))
    snap = spin_outputs(get)
    log(f"spin-cell outputs before: {snap}")
    rec = None
    result = None
    try:
        _set_outputs({c: False for c in SPIN_CELLS}, get, put)
        sleep(0.5)
        now = spin_outputs(get)
        if any(now.values()):
            raise EjectDisabled(f"spin cells still ON after switching off: "
                                f"{[c for c, v in now.items() if v]} -- NOT driving")
        log("all 8 spin-cell outputs read OFF")
        if recorder is not None:
            recorder.start()
            sleep(2.0)                       # pre-run level for the POLES excursion check
        result = execute(plan, put=put, sleep=sleep, occupied=None, check_abort=check_abort,
                         allow=True, log=log)
    finally:
        if recorder is not None and recorder._th.is_alive():
            sleep(1.0)
            rec = recorder.stop()
            recorder.captured = rec          # survives an abort that skips the return
        gains = [float(get(f'{SPIN}_COSGAIN')), float(get(f'{SPIN}_SINGAIN'))]
        if any(gains):
            log(f"!! SPIN gains read {gains} after safe-off: spin-cell outputs LEFT OFF (safe). "
                f"Zero PARTICLE_SPIN_COSGAIN, then restore the outputs by hand: {snap}")
        else:
            _set_outputs(snap, get, put)
            back = spin_outputs(get)
            log(f"spin-cell outputs restored: {'OK' if back == snap else 'MISMATCH ' + repr(back)}")
    return result, rec


def _abort_file():
    if os.path.exists(ABORT_FILE):
        raise KeyboardInterrupt(ABORT_FILE)


def running_guarded_scripts():
    out = subprocess.run(['pgrep', '-af', 'measure_actuator_gain'], capture_output=True, text=True)
    return [l for l in out.stdout.splitlines() if 'pgrep' not in l] or None


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Plan (read-only) the rotating-field eject sweep; "
                                             "--dark-run runs it with the electrode cells switched off.")
    ap.add_argument('--f-start', type=float, default=1.0)
    ap.add_argument('--f-stop', type=float, default=60.0)
    ap.add_argument('--sweep-s', type=float, default=30.0)
    ap.add_argument('--amp-counts', type=float, default=20000.0)
    ap.add_argument('--n-sweeps', type=int, default=None, help='default 3 (1 with --dark-run)')
    ap.add_argument('--dark-run', action='store_true',
                    help='WRITES CDS: spin-cell outputs off, run the sweep, record over NDS2, restore '
                         '(one sweep unless --n-sweeps is given). Needs Alma\'s go.')
    ap.add_argument('--yes', action='store_true', help='skip the typed confirmation of --dark-run')
    ap.add_argument('--json', action='store_true', help='print only the plan as JSON (read-only)')
    ap.add_argument('--out', default=os.path.expanduser(
        '~/Dropbox/Microspheres/MAST-QG/worker1/data/' + time.strftime('%y%m%d') + '/eject_dark'))
    args = ap.parse_args(argv)
    n_sweeps = args.n_sweeps or (1 if args.dark_run else 3)
    plan = plan_eject(read_state(), forward=load_forward_matrix(), f_start=args.f_start,
                      f_stop=args.f_stop, sweep_s=args.sweep_s, amp_counts=args.amp_counts,
                      n_sweeps=n_sweeps, other_procs=running_guarded_scripts())
    if args.json:
        print(json.dumps(plan, default=float))
        return 0
    print(json.dumps(plan, indent=1, default=float))
    if not args.dark_run:
        print("\nPLAN ONLY -- nothing was written to EPICS.")
        return 1 if plan['problems'] else 0
    if plan['problems']:
        print("\nDARK RUN REFUSED: preconditions failed.")
        return 1
    print(f"\nDARK RUN: will switch OFF the outputs of {len(SPIN_CELLS)} spin cells, run "
          f"{n_sweeps} sweep(s) {args.f_start}->{args.f_stop} Hz at COSGAIN {plan['cosgain_counts']:.0f}, "
          f"record {list(DARK_RECORD)}, then restore. Abort: Ctrl-C or touch {ABORT_FILE}.")
    if not args.yes and input("type DARK to proceed: ").strip() != 'DARK':
        print("not confirmed; nothing written.")
        return 1
    if os.path.exists(ABORT_FILE):
        os.remove(ABORT_FILE)
    os.makedirs(args.out, exist_ok=True)
    rec_obj = NDSRecorder()
    t0 = time.strftime('%H%M%S')
    err = None
    try:
        result, rec = dark_run(plan, recorder=rec_obj, check_abort=_abort_file)
    except KeyboardInterrupt:           # Ctrl-C or the abort file: switches already handled
        result, rec = 'aborted', getattr(rec_obj, 'captured', None)
    except Exception as e:              # e.g. recorder refused to start: nothing was driven
        result, rec, err = 'refused', getattr(rec_obj, 'captured', None), repr(e)
    report = {'result': result, 'error': err, 'plan': plan, 'recorder_error': rec_obj.error,
              'gps0': rec_obj.gps0}
    if rec:
        np.savez_compressed(os.path.join(args.out, f'dark_{t0}.npz'),
                            **{k.replace(':', '_'): v for k, v in rec.items()},
                            fs=json.dumps(rec_obj.fs), gps0=rec_obj.gps0 or 0)
        report['analysis'] = analyze_dark(rec, rec_obj.fs)
    path = os.path.join(args.out, f'dark_{t0}.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=1, default=float)
    print(f"dark run: {result}; report {path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
