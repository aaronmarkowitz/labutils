#!/var/lib/cds-conda/base/envs/cds-testing/bin/python3
# Run with: /var/lib/cds-conda/base/envs/cds-testing/bin/python3 measure_actuator_gain.py <config.yml>
"""Measure the flat actuator gain of the four electrodes to each particle DOF.

This is the actuation counterpart to the SENSE pipeline (sensor diagonalization
-> upload_sense_matrix.py). It drives a simultaneous multi-tone "SineResponse"
measurement with the CLI tool ``diag`` (the headless form of diaggui), parses the
result with the ``dttxml`` package, fits the mechanical plant resonance per DOF,
and divides it out to recover a complex K×4 relative coupling matrix
(rows = active DOFs, cols = electrode E1..E4; K derived from step 01 HDF5 or config).

METHOD (see README.md for the full rationale and pitfalls)
  * All four electrodes (POLES_E1..E4) are driven simultaneously, each at several
    DISTINCT tone frequencies clustered around the X/Y/Z resonances (dense near
    X~39 / Y~54 Hz, sparse near Z~5 Hz). Frequency-division multiplexing lets one
    measurement separate every electrode->DOF coupling: each tone frequency is
    assigned to exactly one electrode, so the transfer coefficient at that
    frequency (response / excitation) is that electrode's coupling into each DOF.
  * Because all electrodes drive the SAME mechanical mode for a given DOF, the
    common plant H_d(f) cancels in the per-DOF comparison. We fit H_d (Lorentzian
    f0, Q) jointly with the four per-electrode complex gains and report the gains.
    The reported gain is RELATIVE (the common per-DOF plant scale is absorbed),
    i.e. proportional to counts->Newtons. Actuator gains are assumed
    frequency-flat at these low frequencies; a future version could generalize to
    an arbitrary (frequency-dependent) actuation matrix.
  * Per-electrode tones are Schroeder-phased to minimize crest factor so the
    summed DAC waveform uses actuator range efficiently.

SAFETY
  * No excitation tone may land in the 10-20 Hz guard band (z-mode harmonics live
    there and dominate trap loss; a tone there would also pollute the guard).
  * A background NDS2 monitor watches the 10-20 Hz band-limited RMS of
    PARTICLE_X/Y/Z; if it exceeds ``factor`` x its starting baseline the run
    aborts and the excitation is ramped to zero.
  * Manual abort: SIGINT (Ctrl-C) / SIGTERM, OR an MEDM button / abort_actuator_gain.sh
    that touches the sentinel file (polled every loop).
  * POLES_E* are assumed OPEN LOOP. All POLES_E settings are snapshotted and
    restored; each module's input switch is turned OFF for the measurement so only
    the _EXC test point drives the DAC.

Usage:
    measure_actuator_gain.py <config.yml> [--dry-run] [--premeasure-only] [--label NAME]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

# Heavy hardware/analysis deps. Available in the cds-testing conda env (see shebang).
# Imported at module top so tests under that interpreter can import this module.
import h5py
import nds2
import dttxml
from scipy import signal as sp_signal
from scipy import optimize as sp_optimize

# Plotting module (optional — gracefully absent if matplotlib not installed).
# Same directory as this script; sys.path entry added for direct-script invocation.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from plot_actuator_gain import plot_measurement as _plot_measurement
except ImportError:
    _plot_measurement = None

GPS_OFFSET = 315964818  # GPS = Unix - GPS_OFFSET
DAC_FULL_SCALE_COUNTS = 32768  # 16-bit signed DAC

# Post-connect command sequence handed to `diag -l -f <cmdfile>`. The {gen} and
# {result} placeholders are filled per run. THIS IS THE FIRST THING TO VERIFY on
# the loopback (see README "Verify-first"); it is centralized here so it is a
# one-line fix if the live kernel wants different verbs.
DIAG_COMMAND_SEQUENCE = [
    "restore {gen}",
    "run -w",          # run and WAIT for completion (verbs verified via `diag -l` help)
    "save {result}",
    "quit",
]
# NOTE: diag's `help` prints `restore 'filename'` but the quotes are placeholder
# notation, NOT literal — pass the path unquoted (paths here have no spaces).


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Tone:
    """One excitation tone: a frequency on a specific electrode for a specific DOF."""
    freq: float          # Hz (bin-snapped, guard-clean, globally distinct)
    electrode: str       # "E1".."E4"
    dof: str             # intended DOF "x"/"y"/"z" (for bookkeeping/plant fit grouping)
    amp_counts: float = 0.0
    phase_rad: float = 0.0
    channel: str = ""    # optional explicit excitation channel (else derived from electrode)


@dataclass
class DofFit:
    dof: str
    f0: float
    Q: float
    gains: np.ndarray           # complex, length = n_electrodes (one per electrode)
    fit_plant: bool
    residual_norm: float
    per_electrode_coherence: dict = field(default_factory=dict)


class AbortRequested(Exception):
    """Raised when the guard monitor, a signal, or the sentinel file requests abort."""


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def validate_config(cfg: dict) -> None:
    required_top = ["prefix", "electrodes", "dofs", "frequency_plan", "amplitude",
                    "trim", "diag", "guard_monitor", "safety", "abort", "output_root"]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    n_elec = len(cfg["electrodes"])
    for dof, d in cfg["dofs"].items():
        for k in ["channel", "f0", "Q", "n_tones", "tone_spacing_hz"]:
            if k not in d:
                raise ValueError(f"dofs.{dof} missing key '{k}'")
        if d["n_tones"] < n_elec:
            raise ValueError(
                f"dofs.{dof}.n_tones ({d['n_tones']}) must be >= number of "
                f"electrodes ({n_elec}) so every electrode is driven near every "
                f"resonance.")
    lo, hi = cfg["frequency_plan"]["guard_band_hz"]
    if lo >= hi:
        raise ValueError("frequency_plan.guard_band_hz must be [low, high] with low < high")


def resolve_active_dofs(cfg: dict, step01_h5_path: Path | None = None) -> list[str]:
    """Determine the active DOF list for the measurement.

    If *step01_h5_path* points to an existing step 01 results HDF5, reads its
    ``dofs`` attribute (a JSON-encoded list like ``["x", "y"]``).  Otherwise
    falls back to the keys defined in ``cfg["dofs"]``.
    """
    if step01_h5_path is not None and step01_h5_path.exists():
        with h5py.File(step01_h5_path, "r") as f:
            dofs = json.loads(f.attrs["dofs"])
        for d in dofs:
            if d not in cfg["dofs"]:
                raise ValueError(
                    f"Step 01 HDF5 lists DOF '{d}' but it is not defined in "
                    f"cfg['dofs'] (available: {list(cfg['dofs'].keys())}). "
                    f"Add a config section for it.")
        return dofs
    return list(cfg["dofs"].keys())


# --------------------------------------------------------------------------- #
# Frequency plan + Schroeder phasing  (pure functions, unit-tested)
# --------------------------------------------------------------------------- #
def _snap_to_bin(freq: float, bin_hz: float) -> float:
    return round(freq / bin_hz) * bin_hz


def generate_frequency_plan(cfg: dict, bin_hz: float,
                            active_dofs: list[str] | None = None) -> list[Tone]:
    """Build the list of excitation tones.

    Per DOF: place ``n_tones`` frequencies symmetrically around the seed f0 with
    the configured spacing, snap to FFT-bin multiples, push any tone out of the
    guard band, and assign tones to electrodes round-robin (so every electrode is
    driven near every resonance, and dense DOFs give some electrodes a 2nd tone to
    pin the Lorentzian). Finally enforce global distinctness (>= min_bin_separation
    bins apart) across ALL tones.

    *active_dofs*: subset of ``cfg["dofs"]`` keys to generate tones for. If None,
    generates for all configured DOFs (backward-compatible default).

    Raises ValueError if a DOF's tones cannot be placed outside the guard band.
    """
    fp = cfg["frequency_plan"]
    guard_lo, guard_hi = fp["guard_band_hz"]
    min_sep_bins = int(fp.get("min_bin_separation", 1))
    snap = fp.get("fft_bin_snap", True)
    electrodes = list(cfg["electrodes"])
    min_sep_hz = max(min_sep_bins * bin_hz, bin_hz)

    def in_guard(f):
        return guard_lo <= f <= guard_hi

    def push_out_of_guard(f):
        if not in_guard(f):
            return f
        # move to nearest guard edge minus/plus one bin, staying on the side it came from
        below = guard_lo - min_sep_hz
        above = guard_hi + min_sep_hz
        return below if abs(f - guard_lo) <= abs(f - guard_hi) else above

    dof_items = {d: cfg["dofs"][d] for d in active_dofs} if active_dofs else cfg["dofs"]
    tones: list[Tone] = []
    for dof, d in dof_items.items():
        n = int(d["n_tones"])
        spacing = float(d["tone_spacing_hz"])
        f0 = float(d["f0"])
        offsets = (np.arange(n) - (n - 1) / 2.0) * spacing
        freqs = []
        for j, off in enumerate(offsets):
            f = f0 + off
            if snap:
                f = _snap_to_bin(f, bin_hz)
            f = push_out_of_guard(f)
            if snap:
                f = _snap_to_bin(f, bin_hz)
            if in_guard(f):
                raise ValueError(
                    f"dofs.{dof}: tone {f:.3f} Hz cannot be placed outside guard "
                    f"band {guard_lo}-{guard_hi} Hz; adjust f0/spacing/n_tones.")
            freqs.append(f)
        # assign electrodes round-robin across the (sorted) cluster
        order = np.argsort(freqs)
        for rank, idx in enumerate(order):
            tones.append(Tone(freq=float(freqs[idx]),
                              electrode=electrodes[rank % len(electrodes)],
                              dof=dof))

    # global distinctness: bump colliding tones up by bins until separated
    tones.sort(key=lambda t: t.freq)
    for i in range(1, len(tones)):
        if tones[i].freq - tones[i - 1].freq < min_sep_hz - 1e-9:
            newf = tones[i - 1].freq + min_sep_hz
            if snap:
                newf = _snap_to_bin(newf, bin_hz)
                if newf - tones[i - 1].freq < min_sep_hz - 1e-9:
                    newf += bin_hz
            if guard_lo <= newf <= guard_hi:
                raise ValueError(
                    f"De-collision pushed a tone to {newf:.3f} Hz inside the guard "
                    f"band; reduce tone density or widen spacing.")
            tones[i].freq = float(newf)
    return tones


def schroeder_phases(amplitudes) -> np.ndarray:
    """Schroeder phases (radians) that minimize the crest factor of a multitone.

    phi_k = -2*pi * sum_{j<k} (k-j) * P_j ,  P_j = a_j^2 / sum(a^2)   (0-indexed)
    Must be recomputed whenever the amplitudes change.
    """
    a = np.asarray(amplitudes, dtype=float)
    if a.size == 0:
        return np.zeros(0)
    total = np.sum(a ** 2)
    if total <= 0:
        return np.zeros(a.size)
    P = a ** 2 / total
    phases = np.zeros(a.size)
    for k in range(a.size):
        phases[k] = -2.0 * np.pi * sum((k - j) * P[j] for j in range(k))
    return np.mod(phases, 2.0 * np.pi)


def crest_factor(amps, freqs, phases, fs=2048.0, dur=2.0) -> float:
    """Peak/RMS of the synthesized multitone (for tests and logging)."""
    t = np.arange(int(fs * dur)) / fs
    x = np.zeros_like(t)
    for a, f, p in zip(amps, freqs, phases):
        x += a * np.cos(2.0 * np.pi * f * t + p)
    rms = np.sqrt(np.mean(x ** 2))
    return float(np.max(np.abs(x)) / rms) if rms > 0 else float("inf")


def assign_schroeder_phases(tones: list[Tone]) -> None:
    """Set tone.phase_rad per electrode from its tones' amplitudes (in place)."""
    by_elec: dict[str, list[Tone]] = {}
    for t in tones:
        by_elec.setdefault(t.electrode, []).append(t)
    for elec, ts in by_elec.items():
        amps = [t.amp_counts for t in ts]
        ph = schroeder_phases(amps)
        for t, p in zip(ts, ph):
            t.phase_rad = float(p)


# --------------------------------------------------------------------------- #
# EPICS helpers (subprocess caget/caput, mirroring particle_lo_scan / upload_sense)
# --------------------------------------------------------------------------- #
def caget_t(pv: str) -> str:
    return subprocess.check_output(["caget", "-t", pv], text=True, timeout=10).strip()


def caput(pv, value) -> None:
    subprocess.run(["caput", pv, str(value)], check=True, capture_output=True, timeout=10)


def caget_batch(pvs: list[str]) -> dict[str, float]:
    """Batch caget; returns {pv: float or None}."""
    out: dict[str, float] = {}
    if not pvs:
        return out
    res = subprocess.run(["caget"] + pvs, capture_output=True, text=True, timeout=30)
    for line in res.stdout.strip().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                out[parts[0]] = None
    return out


# --------------------------------------------------------------------------- #
# POLES snapshot / restore (XOR-toggle switch convention from upload_sense_matrix)
# --------------------------------------------------------------------------- #
def _poles_base(cfg, elec):
    return f"{cfg['prefix']}-POLES_{elec}"


def snapshot_poles(cfg: dict) -> dict:
    """Batch-read GAIN/OFFSET/TRAMP/SW1R/SW2R for every electrode module."""
    pvs = []
    for e in cfg["electrodes"]:
        b = _poles_base(cfg, e)
        pvs += [f"{b}_GAIN", f"{b}_OFFSET", f"{b}_TRAMP", f"{b}_SW1R", f"{b}_SW2R"]
    raw = caget_batch(pvs)
    snap = {}
    sw1_bit = cfg["safety"]["sw1_input_on_bit"]
    for e in cfg["electrodes"]:
        b = _poles_base(cfg, e)
        sw1r = raw.get(f"{b}_SW1R")
        snap[e] = {
            "gain": raw.get(f"{b}_GAIN"),
            "offset": raw.get(f"{b}_OFFSET"),
            "tramp": raw.get(f"{b}_TRAMP"),
            "sw1r": sw1r,
            "sw2r": raw.get(f"{b}_SW2R"),
            "input_on": (sw1r is not None) and bool(int(sw1r) & sw1_bit),
        }
    return snap


def setup_poles_for_measurement(cfg: dict, snap: dict, dry_run: bool) -> None:
    """Set GAIN=1 and turn OFF each module input switch.

    GAIN=1 is required: EXC goes through GAIN, so a leftover GAIN=0 (e.g. from
    a previous aborted run) would silently block all excitation. Input switch OFF
    blocks the normal POLES input path while EXC still passes through GAIN to the
    output.
    """
    sw1_bit = cfg["safety"]["sw1_input_on_bit"]
    for e in cfg["electrodes"]:
        b = _poles_base(cfg, e)
        if dry_run:
            on_off = "ON->OFF" if snap[e]["input_on"] else "OFF->OFF"
            print(f"  DRY-RUN: {b}: GAIN=1 (was {snap[e]['gain']}), input {on_off}")
            continue
        caput(f"{b}_GAIN", 1)
        if snap[e]["input_on"]:
            caput(f"{b}_SW1", sw1_bit)


def disable_poles_inputs(cfg: dict, snap: dict, dry_run: bool) -> None:
    """Turn OFF each module input switch (only toggle if currently on)."""
    sw1_bit = cfg["safety"]["sw1_input_on_bit"]
    for e in cfg["electrodes"]:
        if snap[e]["input_on"]:
            b = _poles_base(cfg, e)
            if dry_run:
                print(f"  DRY-RUN: caput {b}_SW1 {sw1_bit}  (turn input OFF)")
            else:
                caput(f"{b}_SW1", sw1_bit)


def restore_poles(cfg: dict, snap: dict, dry_run: bool) -> None:
    """Restore GAIN/OFFSET/TRAMP and re-enable any input switch that was on."""
    sw1_bit = cfg["safety"]["sw1_input_on_bit"]
    tramp = cfg["safety"].get("restore_tramp_s", 2.0)
    for e in cfg["electrodes"]:
        b = _poles_base(cfg, e)
        s = snap[e]
        if dry_run:
            print(f"  DRY-RUN: restore {b}: GAIN={s['gain']} OFFSET={s['offset']} "
                  f"TRAMP={s['tramp']} input_on={s['input_on']}")
            continue
        if s["tramp"] is not None:
            caput(f"{b}_TRAMP", tramp)
        if s["offset"] is not None:
            caput(f"{b}_OFFSET", s["offset"])
        if s["gain"] is not None:
            caput(f"{b}_GAIN", s["gain"])
        # re-enable input switch if it was on and is now off
        cur = caget_batch([f"{b}_SW1R"]).get(f"{b}_SW1R")
        cur_on = (cur is not None) and bool(int(cur) & sw1_bit)
        if s["input_on"] and not cur_on:
            caput(f"{b}_SW1", sw1_bit)
        if s["tramp"] is not None:
            caput(f"{b}_TRAMP", s["tramp"])


# --------------------------------------------------------------------------- #
# ACTS snapshot / setup / restore
# --------------------------------------------------------------------------- #
def snapshot_acts(cfg: dict, excl: list[str]) -> dict:
    """Snapshot GAIN/OFFSET/TRAMP/SW1R/SW2R for ACTS elements from their EXC channels.

    excl: list of _EXC channel names, e.g. ['Y1:DMD-ACTS_8_1_EXC', ...]
    Returns dict keyed by EXC channel name.
    """
    pvs = []
    for exc in excl:
        base = exc[:-4]  # strip _EXC
        pvs += [f"{base}_GAIN", f"{base}_OFFSET", f"{base}_TRAMP",
                f"{base}_SW1R", f"{base}_SW2R"]
    raw = caget_batch(pvs)
    snap = {}
    sw1_bit = cfg["safety"]["sw1_input_on_bit"]
    sw2_bit = cfg["safety"]["sw2_output_on_bit"]
    for exc in excl:
        base = exc[:-4]
        sw1r = raw.get(f"{base}_SW1R")
        sw2r = raw.get(f"{base}_SW2R")
        snap[exc] = {
            "gain": raw.get(f"{base}_GAIN"),
            "offset": raw.get(f"{base}_OFFSET"),
            "tramp": raw.get(f"{base}_TRAMP"),
            "sw1r": sw1r,
            "sw2r": sw2r,
            "input_on": (sw1r is not None) and bool(int(sw1r) & sw1_bit),
            "output_on": (sw2r is not None) and bool(int(sw2r) & sw2_bit),
        }
    return snap


def setup_acts_for_measurement(cfg: dict, excl: list[str], snap: dict,
                                dry_run: bool) -> None:
    """Prepare ACTS elements: GAIN=1, input switch OFF, output switch ON.

    EXC goes through GAIN (confirmed), so GAIN=1 is required. Input switch OFF
    blocks normal ACTS inputs while EXC still bypasses it. Output switch ON
    ensures the EXC contribution passes to the row sum (output switch gates EXC).
    """
    sw1_bit = cfg["safety"]["sw1_input_on_bit"]
    sw2_bit = cfg["safety"]["sw2_output_on_bit"]
    for exc in excl:
        base = exc[:-4]
        s = snap[exc]
        if dry_run:
            print(f"  DRY-RUN: {base}: GAIN=1 (was {s['gain']}), "
                  f"input {'OFF->OFF' if not s['input_on'] else 'ON->OFF'}, "
                  f"output {'ON->ON' if s['output_on'] else 'OFF->ON'}")
            continue
        if s["gain"] != 1.0:
            caput(f"{base}_GAIN", 1)
        if s["input_on"]:
            caput(f"{base}_SW1", sw1_bit)      # XOR-toggle → turn input OFF
        if not s["output_on"]:
            caput(f"{base}_SW2", sw2_bit)      # XOR-toggle → turn output ON


def restore_acts(cfg: dict, excl: list[str], snap: dict, dry_run: bool) -> None:
    """Restore ACTS elements to their pre-measurement state (idempotent)."""
    sw1_bit = cfg["safety"]["sw1_input_on_bit"]
    sw2_bit = cfg["safety"]["sw2_output_on_bit"]
    tramp = cfg["safety"].get("restore_tramp_s", 2.0)
    for exc in excl:
        base = exc[:-4]
        s = snap[exc]
        if dry_run:
            print(f"  DRY-RUN: restore {base}: GAIN={s['gain']} "
                  f"input_on={s['input_on']} output_on={s['output_on']}")
            continue
        if s["tramp"] is not None:
            caput(f"{base}_TRAMP", tramp)
        if s["offset"] is not None:
            caput(f"{base}_OFFSET", s["offset"])
        if s["gain"] is not None:
            caput(f"{base}_GAIN", s["gain"])
        cur = caget_batch([f"{base}_SW1R"]).get(f"{base}_SW1R")
        cur_in = (cur is not None) and bool(int(cur) & sw1_bit)
        if s["input_on"] and not cur_in:
            caput(f"{base}_SW1", sw1_bit)      # XOR-toggle → turn back ON
        cur2 = caget_batch([f"{base}_SW2R"]).get(f"{base}_SW2R")
        cur_out = (cur2 is not None) and bool(int(cur2) & sw2_bit)
        if not s["output_on"] and cur_out:
            caput(f"{base}_SW2", sw2_bit)      # XOR-toggle → turn back OFF
        if s["tramp"] is not None:
            caput(f"{base}_TRAMP", s["tramp"])


_MAX_AWG_SLOTS = 9  # MAX_NUM_AWG from /usr/include/gds/dtt/awgtype.h


def assign_acts_channels(tones: list[Tone], cfg: dict) -> None:
    """Assign one distinct ACTS EXC channel per tone (in place).

    Groups tones by electrode, looks up ACTS row from cfg["acts"]["electrode_row"],
    assigns columns 1..N sequentially. Sets tone.channel so that each tone has its
    own distinct physical excitation channel — this makes sizeA == sizeExc in diag's
    SineResponse, enabling its native per-tone coefficient extraction.

    Raises ValueError if:
    - more than 8 tones per electrode (only 8 ACTS columns per row), OR
    - total distinct EXC channels > MAX_NUM_AWG (9): each channel consumes one AWG
      slot, so the distinct-channel approach is limited to 9 tones total.
      Use POLES_EXC (acts.enabled=false) for higher tone counts.
    """
    prefix = cfg["prefix"]
    elec_row = cfg["acts"]["electrode_row"]
    by_elec: dict[str, list[Tone]] = {}
    for t in tones:
        by_elec.setdefault(t.electrode, []).append(t)
    # Check AWG slot limit BEFORE assigning channels so the error fires before
    # any EPICS writes.
    n_distinct = sum(len(ts) for ts in by_elec.values())
    if n_distinct > _MAX_AWG_SLOTS:
        raise ValueError(
            f"acts.enabled=true requires one AWG slot per tone; {n_distinct} tones "
            f"would exceed MAX_NUM_AWG={_MAX_AWG_SLOTS}. Reduce n_tones across all "
            f"DOFs to <= {_MAX_AWG_SLOTS}, or set acts.enabled=false to use the "
            f"POLES_EXC path (4 AWG slots regardless of tone count).")
    for elec, ts in by_elec.items():
        if elec not in elec_row:
            raise ValueError(f"Electrode '{elec}' not in acts.electrode_row config")
        row = int(elec_row[elec])
        if len(ts) > 8:
            raise ValueError(
                f"Electrode {elec} (ACTS row {row}) has {len(ts)} tones but ACTS "
                f"only has 8 columns per row; reduce n_tones for this DOF")
        for col, t in enumerate(ts, start=1):
            t.channel = f"{prefix}-ACTS_{row}_{col}_EXC"


# --------------------------------------------------------------------------- #
# Trap-loss guard monitor (NDS2 stream + 10-20 Hz band RMS)
# --------------------------------------------------------------------------- #
def band_rms(x: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    """RMS of x after a band-pass filter. Pure function (unit-tested)."""
    lo, hi = band
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.999)
    sos = sp_signal.butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    y = sp_signal.sosfiltfilt(sos, x)
    return float(np.sqrt(np.mean(y ** 2)))


class GuardMonitor(threading.Thread):
    """Streams PARTICLE_X/Y/Z, measures a 10-20 Hz band-RMS baseline, and sets the
    abort event if any DOF's band-RMS exceeds factor x baseline."""

    def __init__(self, cfg: dict, abort_event: threading.Event):
        super().__init__(daemon=True)
        gm = cfg["guard_monitor"]
        self.server = gm["nds2_server"]
        self.port = int(gm["nds2_port"])
        self.channels = list(gm["channels"])
        self.band = tuple(gm["band_hz"])
        self.factor = float(gm["factor"])
        self.baseline_seconds = int(gm["baseline_seconds"])
        self.abort_event = abort_event
        self.baseline_ready = threading.Event()
        self.baseline: dict[str, float] = {}
        self.last: dict[str, float] = {}
        self.error: str | None = None
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            conn = nds2.connection(self.server, self.port)
            baseline_acc = {ch: [] for ch in self.channels}
            n_base = 0
            for bufs in conn.iterate(self.channels):
                if self._stop_event.is_set():
                    return
                for buf in bufs:
                    fs = buf.channel.sample_rate
                    rms = band_rms(np.asarray(buf.data, dtype=np.float64), fs, self.band)
                    ch = buf.channel.name
                    self.last[ch] = rms
                    if not self.baseline_ready.is_set():
                        baseline_acc[ch].append(rms)
                if not self.baseline_ready.is_set():
                    n_base += 1
                    if n_base >= self.baseline_seconds:
                        for ch in self.channels:
                            vals = baseline_acc[ch]
                            self.baseline[ch] = float(np.median(vals)) if vals else 0.0
                        self.baseline_ready.set()
                else:
                    for ch in self.channels:
                        base = self.baseline.get(ch, 0.0)
                        if base > 0 and self.last.get(ch, 0.0) > self.factor * base:
                            self.abort_event.set()
                            return
        except Exception as e:  # noqa: BLE001 - surface any NDS2 error to main
            self.error = repr(e)
            # Do not abort on monitor failure by itself; main decides. Record it.


# --------------------------------------------------------------------------- #
# diag XML generation + invocation
# --------------------------------------------------------------------------- #
def build_sine_response_xml(cfg: dict, tones: list[Tone], meas: dict,
                            meas_channels: list[str] | None = None,
                            active_dofs: list[str] | None = None) -> str:
    """Generate a diaggui SineResponse measurement XML from scratch.

    Convention (verified against a real comb result): the Stimulus* arrays are
    index-aligned (row i = StimulusChannel[i] at StimulusFrequency[i] / amp[i] /
    phase[i]); driving one electrode at several tones means repeating that
    electrode's channel across several rows. So #stimulus rows == #tones.

    Each tone's excitation channel is ``tone.channel`` if set, else the derived
    ``{prefix}-POLES_{electrode}_EXC``. ``meas_channels`` overrides the default
    PARTICLE_{DOF}_IN1 readbacks (used by the loopback self-test).
    """
    prefix = cfg["prefix"]
    rate = cfg["measurement_channel_rate"]
    if meas_channels is None:
        dofs = active_dofs or list(cfg["dofs"].keys())
        meas_channels = [cfg["dofs"][d]["channel"] for d in dofs]

    # stimulus rows in a stable order (sorted by frequency)
    rows = sorted(tones, key=lambda t: t.freq)

    def P(name, typ, val, extra=""):
        return f'    <Param Name="{name}" Type="{typ}"{extra}>{val}</Param>'

    lines = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<LIGO_LW Name="Diagnostics Test">')
    lines.append('  <LIGO_LW Name="Header" Type="Global">')
    lines.append(P("Flag", "string", "TestParameters"))
    lines.append(P("Creator", "string", "measure_actuator_gain.py"))
    lines.append(P("TestType", "string", "SineResponse"))
    lines.append('  </LIGO_LW>')
    # Def block is required: without it diaggui defaults NoStimulus=true ->
    # "Unable to turn on excitations; Unable to start test".
    lines.append('  <LIGO_LW Name="Def" Type="Defaults">')
    lines.append(P("Flag", "string", "TestParameters"))
    lines.append(P("AllowCancel", "boolean", "true"))
    lines.append(P("NoStimulus", "boolean", "false"))
    lines.append(P("NoAnalysis", "boolean", "false"))
    lines.append(P("KeepTraces", "int", "100"))
    lines.append(P("SiteDefault", "byte", "."))
    lines.append(P("SiteForce", "byte", " "))
    lines.append(P("IfoDefault", "byte", " "))
    lines.append(P("IfoForce", "byte", " "))
    lines.append('  </LIGO_LW>')
    lines.append('  <LIGO_LW Name="Sync" Type="Synchronization">')
    lines.append(P("Flag", "string", "TestParameters"))
    lines.append(P("Type", "int", "0"))
    lines.append('    <Time Name="Start" Type="GPS">0</Time>')
    lines.append(P("Wait", "double", "-0", ' Unit="s"'))
    lines.append(P("Repeat", "int", "1"))
    lines.append(P("RepeatRate", "double", "0", ' Unit="s"'))
    lines.append(P("SlowDown", "double", "0", ' Unit="s"'))
    lines.append('  </LIGO_LW>')
    lines.append('  <LIGO_LW Name="Test" Type="TestParameter">')
    lines.append(P("Flag", "string", "TestParameters"))
    lines.append(P("Subtype", "string", "SineResponse"))
    # MeasurementTime Dim2 = [min_time_seconds, n_cycles] (verified against
    # gds SweptSine.hh: SweptSine(... double cycles, double mintime ...)). The
    # measurement runs for max(min_time, n_cycles/freq). We set min_time long enough
    # to cover our NDS2 capture window. SettlingTime is RELATIVE (a fraction of the
    # measurement time), per SweptSine.hh.
    lines.append(P("MeasurementTime", "double",
                   f"{meas['min_time_s']}\t{meas['cycles']}", ' Unit="s" Dim="2"'))
    lines.append(P("SettlingTime", "double", meas.get("settling_frac", 0.1)))
    lines.append(P("RampDown", "double", cfg["diag"]["rampdown_s"]))
    lines.append(P("RampUp", "double", cfg["diag"]["rampup_s"]))
    lines.append(P("AverageType", "int", cfg["diag"]["average_type"]))
    lines.append(P("Averages", "int", meas.get("averages", 1)))
    for i, t in enumerate(rows):
        lines.append(P(f"StimulusFrequency[{i}]", "double", f"{t.freq:.6f}", ' Unit="Hz"'))
    for i, t in enumerate(rows):
        lines.append(P(f"StimulusAmplitude[{i}]", "double", f"{t.amp_counts:.6f}"))
    for i, t in enumerate(rows):
        lines.append(P(f"StimulusOffset[{i}]", "double", "0"))
    for i, t in enumerate(rows):
        lines.append(P(f"StimulusPhase[{i}]", "double", f"{t.phase_rad:.12f}"))
    lines.append(P("HarmonicOrder", "int", "1"))
    lines.append(P("Window", "int", cfg["diag"]["window"]))
    lines.append(P("FFTResult", "boolean", "false"))  # required by diag SineResponse setup
    for i, ch in enumerate(meas_channels):
        lines.append(P(f"MeasurementChannelRate[{i}]", "int", rate))
        lines.append(P(f"MeasurementChannel[{i}]", "string", ch, ' Unit="channel"'))
        lines.append(P(f"MeasurementActive[{i}]", "boolean", "true"))
    for i, t in enumerate(rows):
        ch = t.channel or f"{prefix}-POLES_{t.electrode}_EXC"
        lines.append(P(f"StimulusChannel[{i}]", "string", ch, ' Unit="channel"'))
        lines.append(P(f"StimulusActive[{i}]", "boolean", "true"))
        # StimulusReadback non-empty -> isReadback=true: diag calls sineAnalyze on
        # the actual excitation channel at each tone's frequency.
        # NOTE: this only fully fixes the normalization when sizeExc==sizeA (one
        # distinct channel per tone). With multiple tones per electrode (the real
        # measurement), sizeA=4 but sizeExc=~20; diag's normalization loop still
        # indexes out of the excitation zone for tones 4+. The NDS2 workaround
        # (inject_and_capture + compute_tfs) is the robust path for both cases.
        # Keeping readback set is still correct and harmless. (sineresponse.cc.)
        lines.append(P(f"StimulusReadback[{i}]", "string", ch, ' Unit="channel"'))
    lines.append('  </LIGO_LW>')
    lines.append('</LIGO_LW>')
    return "\n".join(lines) + "\n"


def run_diag_measurement(gen_xml: Path, result_xml: Path, timeout_s: int) -> Path:
    """Drive diag headless: diag -l -f <cmdfile>. Centralized, single-purpose.

    The exact post-connect verbs are DIAG_COMMAND_SEQUENCE (see module top); this
    is the first thing to confirm on the loopback.
    """
    cmds = "\n".join(c.format(gen=gen_xml, result=result_xml)
                     for c in DIAG_COMMAND_SEQUENCE) + "\n"
    cmdfile = result_xml.with_suffix(".cmd")
    cmdfile.write_text(cmds)
    proc = subprocess.run(["diag", "-l", "-f", str(cmdfile)],
                          capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"diag failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    if not result_xml.exists():
        raise RuntimeError(f"diag produced no result file at {result_xml}\n"
                           f"stdout tail:\n{proc.stdout[-1000:]}")
    return result_xml


# --------------------------------------------------------------------------- #
# Inject (via diag) + capture raw (via NDS2) + compute TFs ourselves
# --------------------------------------------------------------------------- #
# We do NOT use diag's SineResponse coefficient output: in dtt 4.1.5 its per-tone
# excitation normalization (sineAnalyze on the drive channel) is unreliable for a
# multi-tone comb. Validated alternative: diag injects the comb correctly (raw
# excitation = commanded amplitude at every tone), we capture the raw channels over
# NDS2 and compute the transfer functions + coherence ourselves. On the ACTS_8_8 ->
# LOS_IN1 loopback this gives 0.500 flat at every tone.

def _exc_channel(cfg: dict, tone: Tone) -> str:
    return tone.channel or f"{cfg['prefix']}-POLES_{tone.electrode}_EXC"


def inject_and_capture(cfg: dict, tones: list[Tone], meas: dict, run_dir: Path,
                       label: str, abort_event: threading.Event, sentinel: Path,
                       active_dofs: list[str] | None = None):
    """Inject the comb with diag and capture raw PARTICLE + excitation channels.

    diag runs in a background thread (its injection is correct and its result XML is
    saved for diaggui/archiving); the main thread captures the raw channels over
    NDS2 during the steady injection window.

    Returns (captured: {channel: np.ndarray}, fs: float, result_xml: Path).
    """
    timeout_s = max(float(cfg["diag"]["diag_timeout_s"]), meas["min_time_s"] + 60)

    gen_xml = run_dir / f"gen_{label}.xml"
    gen_xml.write_text(build_sine_response_xml(cfg, tones, meas,
                                               active_dofs=active_dofs))
    result_xml = run_dir / f"result_{label}.xml"

    diag_err: dict = {}

    def _run():
        try:
            run_diag_measurement(gen_xml, result_xml, timeout_s)
        except Exception as e:  # noqa: BLE001 - surfaced to caller after join
            diag_err["e"] = e

    th = threading.Thread(target=_run, daemon=True)
    th.start()

    an = cfg["analysis"]
    waited = 0.0
    while waited < float(an["warmup_s"]):        # let settling + ramp-up pass
        _check_aborts(abort_event, sentinel)
        time.sleep(0.5)
        waited += 0.5

    dofs = active_dofs or list(cfg["dofs"].keys())
    chans = [cfg["dofs"][d]["channel"] for d in dofs]
    chans += [_exc_channel(cfg, t) for t in tones]
    chans = sorted(set(chans))            # dedup for the NDS2 request
    conn = nds2.connection(an["nds2_server"], int(an["nds2_port"]))
    acc = {c: [] for c in chans}
    fs = None
    n_strides = int(round(float(meas["capture_s"])))
    n = 0
    for bufs in conn.iterate(chans):
        _check_aborts(abort_event, sentinel)
        for b in bufs:
            acc[b.channel.name].append(np.asarray(b.data, dtype=np.float64))
            fs = b.channel.sample_rate
        n += 1
        if n >= n_strides:
            break

    th.join(timeout=timeout_s)
    if diag_err:
        raise diag_err["e"]
    captured = {c: np.concatenate(acc[c]) for c in chans if acc[c]}
    return captured, fs, result_xml


def compute_tfs(captured: dict, fs: float, tones: list[Tone], cfg: dict,
                segment_s: float | None = None,
                active_dofs: list[str] | None = None) -> list[dict]:
    """Per-tone transfer coefficients + coherence from raw captured data.

    For a tone driven on excitation channel E at frequency f, and DOF channel R:
        TF  = S_ER(f) / S_EE(f)                       (H1 estimator; response/drive)
        coh = |S_ER(f)|^2 / (S_EE(f) * S_RR(f))
    via Welch cross/auto spectra (segment = analysis.segment_s). Spectra are cached
    per channel/pair so each is computed once.

    segment_s overrides cfg["analysis"]["segment_s"] when supplied (used by the trim
    loop, which may increase segment_s beyond the config default to improve coherence).
    """
    dofs = active_dofs or list(cfg["dofs"].keys())
    dof_ch = {d: cfg["dofs"][d]["channel"] for d in dofs}
    seg = segment_s if segment_s is not None else float(cfg["analysis"]["segment_s"])
    nperseg = max(256, int(seg * fs))
    exc_names = sorted({_exc_channel(cfg, t) for t in tones})

    f_arr = None
    Pee, Prr, Per = {}, {}, {}
    for ec in exc_names:
        f_arr, Pee[ec] = sp_signal.csd(captured[ec], captured[ec], fs=fs, nperseg=nperseg)
    for d, ch in dof_ch.items():
        _, Prr[d] = sp_signal.csd(captured[ch], captured[ch], fs=fs, nperseg=nperseg)
    for ec in exc_names:
        for d, ch in dof_ch.items():
            _, Per[(ec, d)] = sp_signal.csd(captured[ec], captured[ch], fs=fs, nperseg=nperseg)

    records = []
    for t in tones:
        ec = _exc_channel(cfg, t)
        i = int(np.argmin(np.abs(f_arr - t.freq)))
        rec = {"electrode": t.electrode, "dof_intended": t.dof,
               "freq": t.freq, "tf": {}, "coh": {}}
        for d in dofs:
            denom = Pee[ec][i] if abs(Pee[ec][i]) > 0 else 1.0
            rec["tf"][d] = complex(Per[(ec, d)][i] / denom)
            coh = (abs(Per[(ec, d)][i]) ** 2) / (abs(Pee[ec][i]) * abs(Prr[d][i]) + 1e-30)
            rec["coh"][d] = float(np.clip(coh, 0.0, 1.0))
        records.append(rec)
    return records


# --------------------------------------------------------------------------- #
# Plant fit + gain extraction
# --------------------------------------------------------------------------- #
def plant_lorentzian(f, f0, Q):
    """Peak-magnitude-normalized complex Lorentzian (|H(f0)| = 1)."""
    f = np.asarray(f, dtype=float)
    h_raw = 1.0 / (f0 ** 2 - f ** 2 + 1j * f * f0 / Q)
    peak = Q / (f0 ** 2)            # = |h_raw(f0)|
    return h_raw / peak


def _coh_weight(coh):
    """Coherence-based weight: sqrt(coh / (1 - coh))."""
    c = np.clip(coh, 0, 0.999999)
    return np.sqrt(c / (1.0 - c + 1e-6))


def fit_dof(records: list[dict], dof: str, electrodes: list[str],
            f0_seed: float, Q_seed: float, fit_plant: bool,
            fit_strategy: str = "joint") -> DofFit:
    """Fit the shared plant (optional) + per-electrode complex gains for one DOF.

    Model: TF_meas(electrode e at freq f, measured in this DOF) = G[e] * H(f; f0, Q).

    fit_strategy (used only when fit_plant=True):
      "joint"          - complex fit of f0, Q, G using ALL tones (original algorithm)
      "dof_filtered"   - same complex fit, but only tones with dof_intended == dof
      "mag_then_linear"- magnitude-only fit for f0/Q on dof-intended tones, then
                         |H|^2-weighted linear solve for complex G
    """
    VALID_STRATEGIES = ("joint", "dof_filtered", "mag_then_linear")
    if fit_strategy not in VALID_STRATEGIES:
        raise ValueError(f"fit_strategy must be one of {VALID_STRATEGIES}, got {fit_strategy!r}")

    n_e = len(electrodes)
    e_index = {e: i for i, e in enumerate(electrodes)}

    # Select records based on strategy
    if fit_plant and fit_strategy in ("dof_filtered", "mag_then_linear"):
        fit_records = [r for r in records if r["dof_intended"] == dof]
    else:
        fit_records = list(records)

    freqs = np.array([r["freq"] for r in fit_records], dtype=float)
    tf = np.array([r["tf"][dof] for r in fit_records], dtype=complex)
    coh = np.array([r["coh"][dof] for r in fit_records], dtype=float)
    e_idx = np.array([e_index[r["electrode"]] for r in fit_records], dtype=int)
    w = _coh_weight(coh)

    if fit_plant and fit_strategy == "mag_then_linear":
        # --- Step 1: magnitude-only fit for f0, Q, |G| per electrode ---
        # Jointly fit the plant shape and per-electrode gain magnitudes.
        # Immune to phase corruption from cross-coupling.
        H_seed = plant_lorentzian(freqs, f0_seed, Q_seed)
        G0_mag = np.ones(n_e)
        for i in range(n_e):
            mask = e_idx == i
            if np.any(mask):
                sel = np.where(mask)[0][np.argmax(np.abs(tf[mask]))]
                denom = abs(H_seed[sel]) if abs(H_seed[sel]) > 1e-12 else 1.0
                G0_mag[i] = abs(tf[sel]) / denom

        p0_mag = np.concatenate([[f0_seed, Q_seed], G0_mag])
        lb_mag = np.concatenate([[f0_seed * 0.8, Q_seed * 0.2], np.zeros(n_e)])
        ub_mag = np.concatenate([[f0_seed * 1.2, Q_seed * 2.0], np.inf * np.ones(n_e)])

        def mag_residuals(p):
            f0, Q = p[0], p[1]
            G_mag = p[2:2 + n_e]
            H = plant_lorentzian(freqs, f0, Q)
            return w * (G_mag[e_idx] * np.abs(H) - np.abs(tf))

        sol_mag = sp_optimize.least_squares(
            mag_residuals, p0_mag, bounds=(lb_mag, ub_mag), method="trf")
        f0_fit, Q_fit = float(sol_mag.x[0]), float(sol_mag.x[1])

        # --- Step 2: complex G via weighted linear solve ---
        # Phase from linear solve: G_phase = Σ(w²·conj(H)·TF) / Σ(w²·|H|²)
        # Magnitude from step 1 (immune to phase corruption): |G| = sol_mag.x[2+i]
        # Combining them preserves the robust magnitude while extracting phase.
        H_fit = plant_lorentzian(freqs, f0_fit, Q_fit)
        G_fit = np.zeros(n_e, dtype=complex)
        for i in range(n_e):
            mask = e_idx == i
            if np.any(mask):
                Hi = H_fit[mask]
                TFi = tf[mask]
                wi = w[mask]
                denom = np.sum(wi ** 2 * np.abs(Hi) ** 2)
                G_phase = (np.sum(wi ** 2 * np.conj(Hi) * TFi) / denom
                           if denom > 0 else 1.0)
                G_fit[i] = sol_mag.x[2 + i] * np.exp(1j * np.angle(G_phase))

        model = G_fit[e_idx] * H_fit
        residual = w * (model - tf)
        res_norm = float(np.linalg.norm(np.concatenate([residual.real, residual.imag])))

    elif fit_plant:
        # --- "joint" or "dof_filtered": complex fit for f0, Q, G ---
        def unpack(p):
            return p[0], p[1], p[2:2 + 2 * n_e:2] + 1j * p[3:2 + 2 * n_e:2]

        def residuals(p):
            f0, Q, G = unpack(p)
            model = G[e_idx] * plant_lorentzian(freqs, f0, Q)
            r = w * (model - tf)
            return np.concatenate([r.real, r.imag])

        G0 = np.ones(n_e, dtype=complex)
        H_seed = plant_lorentzian(freqs, f0_seed, Q_seed)
        for i in range(n_e):
            mask = e_idx == i
            if np.any(mask):
                j = np.argmax(coh[mask])
                sel = np.where(mask)[0][j]
                denom = H_seed[sel] if abs(H_seed[sel]) > 1e-12 else 1.0
                G0[i] = tf[sel] / denom
        g0 = np.empty(2 * n_e)
        g0[0::2] = G0.real
        g0[1::2] = G0.imag

        p0 = np.concatenate([[f0_seed, Q_seed], g0])
        lb = np.concatenate([[f0_seed * 0.8, Q_seed * 0.2], -np.inf * np.ones(2 * n_e)])
        ub = np.concatenate([[f0_seed * 1.2, Q_seed * 5.0], np.inf * np.ones(2 * n_e)])
        sol = sp_optimize.least_squares(residuals, p0, bounds=(lb, ub), method="trf")
        f0_fit, Q_fit, G_fit = unpack(sol.x)
        res_norm = float(np.linalg.norm(sol.fun))

    else:
        # --- fit_plant=False: G-only at fixed f0/Q (unchanged) ---
        def residuals_g(p):
            G = p[0::2] + 1j * p[1::2]
            model = G[e_idx] * plant_lorentzian(freqs, f0_seed, Q_seed)
            r = w * (model - tf)
            return np.concatenate([r.real, r.imag])

        G0 = np.ones(n_e, dtype=complex)
        H_seed = plant_lorentzian(freqs, f0_seed, Q_seed)
        for i in range(n_e):
            mask = e_idx == i
            if np.any(mask):
                j = np.argmax(coh[mask])
                sel = np.where(mask)[0][j]
                denom = H_seed[sel] if abs(H_seed[sel]) > 1e-12 else 1.0
                G0[i] = tf[sel] / denom
        g0 = np.empty(2 * n_e)
        g0[0::2] = G0.real
        g0[1::2] = G0.imag

        sol = sp_optimize.least_squares(residuals_g, g0, method="lm")
        G_fit = sol.x[0::2] + 1j * sol.x[1::2]
        f0_fit, Q_fit = f0_seed, Q_seed
        res_norm = float(np.linalg.norm(sol.fun))

    # per-electrode coherence from ALL records (for reporting)
    all_coh = np.array([r["coh"][dof] for r in records], dtype=float)
    all_e_idx = np.array([e_index[r["electrode"]] for r in records], dtype=int)
    per_e_coh = {}
    for i, e in enumerate(electrodes):
        mask = all_e_idx == i
        per_e_coh[e] = float(np.max(all_coh[mask])) if np.any(mask) else 0.0
    return DofFit(dof=dof, f0=float(f0_fit), Q=float(Q_fit), gains=G_fit,
                  fit_plant=fit_plant, residual_norm=res_norm,
                  per_electrode_coherence=per_e_coh)


def assemble_gain_matrix(dof_fits: dict, electrodes: list[str],
                         active_dofs: list[str] | None = None) -> np.ndarray:
    """Stack per-DOF gains into a complex (K, n_electrodes) matrix."""
    dofs = active_dofs or list(dof_fits.keys())
    return np.array([dof_fits[d].gains for d in dofs], dtype=complex)


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def write_hdf5(run_dir: Path, gain_matrix: np.ndarray, dof_fits: dict,
               records: list[dict], cfg_text: str, electrodes: list[str],
               active_dofs: list[str] | None = None) -> Path:
    dofs = active_dofs or list(dof_fits.keys())
    path = run_dir / "actuator_gain_results.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("gain_matrix_real", data=gain_matrix.real)
        f.create_dataset("gain_matrix_imag", data=gain_matrix.imag)
        f.create_dataset("tone_freqs", data=np.array([r["freq"] for r in records]))
        for d in dofs:
            f.create_dataset(f"tf_{d}_real",
                             data=np.array([r["tf"][d].real for r in records]))
            f.create_dataset(f"tf_{d}_imag",
                             data=np.array([r["tf"][d].imag for r in records]))
            f.create_dataset(f"coherence_{d}",
                             data=np.array([r["coh"][d] for r in records]))
        for d in dofs:
            f.attrs[f"peak_frequency_hz_{d}"] = dof_fits[d].f0
            f.attrs[f"Q_{d}"] = dof_fits[d].Q
            f.attrs[f"fit_plant_{d}"] = dof_fits[d].fit_plant
            f.attrs[f"residual_norm_{d}"] = dof_fits[d].residual_norm
        f.attrs["electrodes"] = json.dumps(electrodes)
        f.attrs["tone_electrode"] = json.dumps([r["electrode"] for r in records])
        f.attrs["dof_order"] = json.dumps(dofs)
        f.attrs["channel_names"] = json.dumps([r["electrode"] for r in records])
        f.attrs["params_yaml"] = cfg_text
    return path


def write_report(run_dir: Path, gain_matrix: np.ndarray, dof_fits: dict,
                 electrodes: list[str], active_dofs: list[str] | None = None) -> Path:
    dofs = active_dofs or list(dof_fits.keys())
    path = run_dir / "actuator_gain_report.txt"
    lines = []
    lines.append("Actuator gain measurement report")
    lines.append("=" * 60)
    for d in dofs:
        fit = dof_fits[d]
        lines.append(f"\nDOF {d.upper()}: f0={fit.f0:.4f} Hz  Q={fit.Q:.2f}  "
                     f"(plant {'fit' if fit.fit_plant else 'fixed'})  "
                     f"residual={fit.residual_norm:.3g}")
        for i, e in enumerate(electrodes):
            g = fit.gains[i]
            lines.append(f"    {e}: |G|={abs(g):.4g}  phase={np.degrees(np.angle(g)):+7.1f} deg"
                         f"   coh_max={fit.per_electrode_coherence.get(e, 0):.3f}")
    lines.append("\nGain matrix |G| (rows " + "/".join(d.upper() for d in dofs)
                 + ", cols " + ",".join(electrodes) + "):")
    for d_i, d in enumerate(dofs):
        lines.append("  " + "  ".join(f"{abs(gain_matrix[d_i, j]):10.4g}"
                                       for j in range(len(electrodes))))
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(text)
    return path


def save_raw_capture(run_dir: Path, captured: dict, fs: float,
                     tones: list[Tone], cfg_text: str) -> Path:
    """Save the raw NDS2 capture + tone metadata so `--analyze` can recompute."""
    path = run_dir / "raw_capture.h5"
    with h5py.File(path, "w") as f:
        cmap = {}
        for ch, arr in captured.items():
            key = ch.replace(":", "_")
            cmap[key] = ch
            f.create_dataset(key, data=arr.astype(np.float32), compression="gzip")
        f.attrs["channel_map"] = json.dumps(cmap)
        f.attrs["fs"] = fs
        f.attrs["tone_freqs"] = json.dumps([t.freq for t in tones])
        f.attrs["tone_electrodes"] = json.dumps([t.electrode for t in tones])
        f.attrs["tone_dofs"] = json.dumps([t.dof for t in tones])
        f.attrs["tone_channels"] = json.dumps([t.channel for t in tones])
        f.attrs["params_yaml"] = cfg_text
    return path


def load_raw_capture(path: Path):
    """Inverse of save_raw_capture. Returns (captured, fs, tones, stored_cfg)."""
    with h5py.File(path, "r") as f:
        cmap = json.loads(f.attrs["channel_map"])
        captured = {orig: f[key][:].astype(np.float64) for key, orig in cmap.items()}
        fs = float(f.attrs["fs"])
        freqs = json.loads(f.attrs["tone_freqs"])
        elecs = json.loads(f.attrs["tone_electrodes"])
        dofs = json.loads(f.attrs["tone_dofs"])
        chans = json.loads(f.attrs["tone_channels"])
        stored_cfg = yaml.safe_load(f.attrs["params_yaml"])
    tones = [Tone(freq=fr, electrode=e, dof=d, channel=c)
             for fr, e, d, c in zip(freqs, elecs, dofs, chans)]
    return captured, fs, tones, stored_cfg


def analyze_and_write(cfg: dict, records: list[dict], run_dir: Path,
                      cfg_text: str, electrodes: list[str],
                      plots_dir: Path | None = None, is_trim: bool = False,
                      active_dofs: list[str] | None = None):
    """Fit the plant + gains, assemble the matrix, write HDF5 + report + plots.

    plots_dir: override plot output directory (default: run_dir/plots/ for the final
    measurement, run_dir/plots/trim_{label}/ for trim steps when is_trim=True).
    is_trim: if True, pass is_trim=True to the plotter (skip gain matrix figure,
    full scatter opacity regardless of coherence).
    Shared by the live measurement and the offline `--analyze` mode.
    """
    dofs = active_dofs or list(cfg["dofs"].keys())
    dof_fits = {}
    for d in dofs:
        dd = cfg["dofs"][d]
        dof_fits[d] = fit_dof(records, d, electrodes, float(dd["f0"]),
                              float(dd["Q"]), bool(dd.get("fit_plant", True)),
                              str(dd.get("fit_strategy", "joint")))
    gain_matrix = assemble_gain_matrix(dof_fits, electrodes, dofs)
    h5 = write_hdf5(run_dir, gain_matrix, dof_fits, records, cfg_text, electrodes, dofs)
    write_report(run_dir, gain_matrix, dof_fits, electrodes, dofs)
    print(f"Saved: {h5}")

    # Bode plots
    if _plot_measurement is not None:
        pd = plots_dir if plots_dir is not None else (run_dir / "plots")
        try:
            written = _plot_measurement(records, dof_fits, electrodes, cfg, pd,
                                        is_trim=is_trim)
            for p in written:
                print(f"  plot: {p}")
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: plotting failed ({e}); data saved successfully.")

    return gain_matrix, dof_fits


# --------------------------------------------------------------------------- #
# Trim loop helpers
# --------------------------------------------------------------------------- #
def worst_primary_coherence(records: list[dict], active_dofs: list[str]) -> float:
    """Minimum coherence over primary DOFs on each tone's intended DOF.

    Z is excluded from the coherence target because its resonance (~5 Hz) gives
    far fewer cycles per measurement window, resulting in structurally lower
    coherence. If Z is the only active DOF, it is used.
    """
    primary = [d for d in active_dofs if d != "z"] or active_dofs
    vals = [r["coh"][r["dof_intended"]] for r in records
            if r["dof_intended"] in primary]
    return min(vals) if vals else 1.0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _check_aborts(abort_event: threading.Event, sentinel: Path):
    if abort_event.is_set():
        raise AbortRequested("trap-loss guard or signal")
    if sentinel.exists():
        raise AbortRequested(f"sentinel file present: {sentinel}")


def _resolve_capture_s(acfg: dict, key_n: str, key_s: str) -> float:
    """Resolve NDS2 capture duration from either an averages count or an explicit seconds value.

    Set exactly one of ``key_n`` (integer number of Welch averages) or ``key_s`` (explicit
    seconds) in ``acfg``; setting both raises ValueError. ``key_n`` is multiplied by
    ``acfg["segment_s"]`` to yield the window length.
    """
    has_n = key_n in acfg
    has_s = key_s in acfg
    if has_n and has_s:
        raise ValueError(
            f"analysis config: set exactly one of '{key_n}' or '{key_s}', not both")
    if has_n:
        return float(acfg[key_n]) * float(acfg["segment_s"])
    if has_s:
        return float(acfg[key_s])
    raise ValueError(
        f"analysis config: missing '{key_n}' or '{key_s}' — set one of them")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=None,
                    help="Path to config YAML (optional with --analyze; "
                         "defaults to config embedded in capture file)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build the plan/XML and print planned actions; touch no hardware")
    ap.add_argument("--premeasure-only", action="store_true",
                    help="Run only the short pre-measurement, then stop")
    ap.add_argument("--emit-xml", action="store_true",
                    help="Write the comb diag XML and exit (for manual diaggui use); no hardware")
    ap.add_argument("--diaggui", action="store_true",
                    help="Write the comb XML and launch diaggui on it to watch live; no analysis")
    ap.add_argument("--analyze", metavar="RAW_CAPTURE_H5", default=None,
                    help="Recompute the gain matrix offline from a saved raw_capture.h5; no hardware")
    ap.add_argument("--reanalysis-label", default=None,
                    help="Suffix for the reanalysis output dir, e.g. 'joint' -> reanalysis_joint/")
    ap.add_argument("--step01-h5", metavar="PATH", default=None,
                    help="Step 01 HDF5 to derive the active DOF list from (reads its 'dofs' attr)")
    ap.add_argument("--label", default=None, help="Override run_label")
    args = ap.parse_args()

    if args.analyze and args.config is None:
        import tempfile
        with h5py.File(args.analyze, "r") as _f:
            _yaml_text = str(_f.attrs["params_yaml"])
        _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
        _tmp.write(_yaml_text)
        _tmp.flush()
        args.config = _tmp.name
        print("--analyze: using config embedded in capture file")
    elif not args.analyze and args.config is None:
        ap.error("config argument is required for live measurement")

    cfg = load_config(Path(args.config))
    validate_config(cfg)
    cfg_text = Path(args.config).read_text()
    electrodes = list(cfg["electrodes"])

    step01_path = (Path(args.step01_h5) if args.step01_h5
                   else (Path(os.path.expandvars(cfg["step01_results_h5"]))
                         if cfg.get("step01_results_h5") else None))
    active_dofs = resolve_active_dofs(cfg, step01_path)
    print(f"Active DOFs: {active_dofs}"
          f" (from {'step 01 HDF5' if step01_path else 'config'})")

    # ---- offline re-analysis of a saved raw capture (no hardware) ----
    if args.analyze:
        cap_path = Path(args.analyze)
        captured, fs, tones, _ = load_raw_capture(cap_path)
        records = compute_tfs(captured, fs, tones, cfg, active_dofs=active_dofs)
        reanalysis_name = (f"reanalysis_{args.reanalysis_label}"
                           if args.reanalysis_label else "reanalysis")
        run_dir = cap_path.resolve().parent / reanalysis_name
        run_dir.mkdir(exist_ok=True)
        print(f"Re-analyzing {cap_path} -> {run_dir}")
        plots_dir = _resolve_plots_dir(cfg, run_dir, label=None)
        analyze_and_write(cfg, records, run_dir, cfg_text, electrodes,
                          plots_dir=plots_dir, active_dofs=active_dofs)
        return

    # Tones are snapped to the analysis FFT bin (1 / Welch segment length) so each
    # tone lands cleanly on a bin in our own spectral analysis. segment_s may grow
    # during the trim loop (doubling each step); tones stay on-bin because larger
    # segments produce finer bin grids that are supersets of the original.
    segment_s = float(cfg["analysis"]["segment_s"])
    bin_hz = 1.0 / segment_s
    premeasure_n_avgs = int(_resolve_capture_s(cfg["analysis"],
                                               "premeasure_n_averages",
                                               "premeasure_capture_s") / segment_s)
    measure_n_avgs = int(_resolve_capture_s(cfg["analysis"],
                                            "n_averages",
                                            "measure_capture_s") / segment_s)
    premeasure = dict(cfg["diag"]["premeasure"])
    premeasure["capture_s"] = premeasure_n_avgs * segment_s
    measure = dict(cfg["diag"]["measure"])
    measure["capture_s"] = measure_n_avgs * segment_s

    # Ensure diag's min_time_s covers the full NDS2 capture window. diag drives
    # the AWG excitation for ~min_time_s; if it finishes before our NDS2 capture
    # is done, the excitation drops to zero mid-capture.
    warmup = float(cfg["analysis"]["warmup_s"])
    margin = 5.0  # seconds of extra injection past the capture window end
    for m in (premeasure, measure):
        required = warmup + m["capture_s"] + margin
        if m["min_time_s"] < required:
            m["min_time_s"] = required

    tones = generate_frequency_plan(cfg, bin_hz, active_dofs=active_dofs)
    init_amp = cfg["amplitude"]["initial_amplitude_counts"]
    for t in tones:
        t.amp_counts = float(init_amp)
    if cfg["schroeder"].get("enabled", True):
        assign_schroeder_phases(tones)

    # Assign one distinct ACTS EXC channel per tone when acts mode is enabled.
    # This makes sizeA == sizeExc in diag's SineResponse so its native per-tone
    # coefficient extraction works correctly. Falls back to POLES_E*_EXC otherwise.
    acts_enabled = bool(cfg.get("acts", {}).get("enabled", False))
    if acts_enabled:
        assign_acts_channels(tones, cfg)
    exc_channels = list({t.channel or f"{cfg['prefix']}-POLES_{t.electrode}_EXC"
                         for t in tones})

    sentinel = Path(cfg["abort"]["sentinel_path"])
    run_dir = _make_run_dir(cfg, args.label)
    print(f"Output directory: {run_dir}")
    print(f"Analysis FFT bin: {bin_hz:.4f} Hz   tones: {len(tones)}")
    print(f"Excitation: {'ACTS EXC (one channel/tone)' if acts_enabled else 'POLES EXC'}")
    _print_plan(tones, electrodes)

    if args.dry_run:
        gen = build_sine_response_xml(cfg, tones, premeasure, active_dofs=active_dofs)
        (run_dir / "gen_dryrun.xml").write_text(gen)
        print(f"\nDRY RUN: wrote example XML to {run_dir/'gen_dryrun.xml'}; no hardware touched.")
        return

    if args.emit_xml or args.diaggui:
        xml_path = run_dir / "gen_emit.xml"
        xml_path.write_text(build_sine_response_xml(cfg, tones, measure,
                                                    active_dofs=active_dofs))
        amp = cfg["amplitude"]["initial_amplitude_counts"]
        print(f"\nWrote comb XML to {xml_path}")
        print(f"  (untrimmed; all tones at the initial {amp} counts -- adjust in diaggui as needed)")
        if args.diaggui:
            print("Launching diaggui (its multi-tone coefficient display is unreliable; "
                  "use it to watch the injection, not to read the gains)...")
            subprocess.Popen(["diaggui", str(xml_path)])
        return

    if sentinel.exists():
        print(f"WARNING: sentinel file {sentinel} already exists; remove it before running.")
        sys.exit(1)

    abort_event = threading.Event()

    def _sig_handler(signum, frame):
        print(f"\nSignal {signum} received -> abort.")
        abort_event.set()
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    snap = snapshot_poles(cfg)
    acts_snap = snapshot_acts(cfg, exc_channels) if acts_enabled else {}
    guard = GuardMonitor(cfg, abort_event)
    restored = [False]
    diag_proc_holder: list = []

    def cleanup():
        guard.stop()
        if not restored[0]:
            # Step 1: zero gains immediately (TRAMP=0, instant) so the CDS AWG
            # comb, which continues injecting until MeasurementTime expires after
            # the diag client is killed, cannot reach the DAC output.
            print("Aborting: zeroing POLES gains (TRAMP=0) immediately...")
            for e in cfg["electrodes"]:
                b = _poles_base(cfg, e)
                try:
                    caput(f"{b}_TRAMP", 0)
                    caput(f"{b}_GAIN", 0)
                except Exception:
                    pass
            if acts_enabled:
                for exc in exc_channels:
                    base = exc[:-4]
                    try:
                        caput(f"{base}_TRAMP", 0)
                        caput(f"{base}_GAIN", 0)
                    except Exception:
                        pass
            # Step 2: wait for the AWG to drain before restoring GAIN. Restoring
            # GAIN while the AWG is still outputting the comb would re-enable the
            # excitation at the output.
            print("Waiting for AWG excitation to drain (EXC channels → 0)...")
            drained = _wait_for_awg_drain(cfg, exc_channels)
            if not drained:
                print("WARNING: AWG did not drain within timeout; restoring anyway.")
            else:
                print("AWG drained — restoring POLES state.")
            # Step 3: restore all settings including GAIN (now safe to ramp up).
            try:
                restore_poles(cfg, snap, dry_run=False)
                if acts_enabled:
                    restore_acts(cfg, exc_channels, acts_snap, dry_run=False)
            finally:
                restored[0] = True

    try:
        setup_poles_for_measurement(cfg, snap, dry_run=False)
        if acts_enabled:
            setup_acts_for_measurement(cfg, exc_channels, acts_snap, dry_run=False)
        guard.start()
        print(f"Guard monitor: measuring {guard.baseline_seconds}s baseline "
              f"in {guard.band[0]}-{guard.band[1]} Hz...")
        while not guard.baseline_ready.is_set():
            if guard.error:
                raise RuntimeError(f"Guard monitor failed to start: {guard.error}")
            _check_aborts(abort_event, sentinel)
            time.sleep(0.5)
        print(f"  baseline: {guard.baseline}")

        # ---------------- adaptive trim loop (segment_s-first, then amplitude) ----
        # Each iteration: diag injects the comb, we capture raw NDS2 and compute the
        # transfer functions + coherence ourselves (diag's coefficients are unused).
        # Trim strategy: first double segment_s (longer FFT segments → narrower bins →
        # better SNR per bin → higher true coherence), keeping n_averages fixed so
        # capture_s grows proportionally. Once segment_s hits trim.segment_s_max,
        # raise amplitudes of under-coherent electrodes.
        trim_segment_s = segment_s
        meas = dict(premeasure)
        records = None
        for it in range(cfg["trim"]["max_trim_iters"] + 1):
            _check_aborts(abort_event, sentinel)
            if cfg["schroeder"].get("enabled", True):
                assign_schroeder_phases(tones)
            print(f"[trim {it}] segment_s={trim_segment_s:.2f}s "
                  f"min_time={meas['min_time_s']}s capture={meas['capture_s']:.1f}s "
                  f"max_amp={max(t.amp_counts for t in tones):.0f} cts -> inject+capture...")
            captured, fs, _ = inject_and_capture(cfg, tones, meas, run_dir,
                                                 f"{it:02d}", abort_event, sentinel,
                                                 active_dofs=active_dofs)
            records = compute_tfs(captured, fs, tones, cfg, segment_s=trim_segment_s,
                                  active_dofs=active_dofs)
            # Trim-step plots: fit plant+gains and plot without writing HDF5/report.
            if _plot_measurement is not None:
                trim_plots_dir = _resolve_plots_dir(cfg, run_dir, None) / f"trim_{it:02d}"
                try:
                    trim_dof_fits = {}
                    for d in active_dofs:
                        dd = cfg["dofs"][d]
                        trim_dof_fits[d] = fit_dof(records, d, electrodes,
                                                   float(dd["f0"]), float(dd["Q"]),
                                                   bool(dd.get("fit_plant", True)))
                    _plot_measurement(records, trim_dof_fits, electrodes, cfg,
                                      trim_plots_dir, is_trim=True)
                except Exception as e:  # noqa: BLE001
                    print(f"  WARNING: trim-step plotting failed ({e})")
            worst = worst_primary_coherence(records, active_dofs)
            print(f"  worst primary coherence = {worst:.3f} (target {cfg['trim']['target_coherence']})")
            if worst >= cfg["trim"]["target_coherence"]:
                break
            if it == cfg["trim"]["max_trim_iters"] or args.premeasure_only:
                break
            meas, trim_segment_s, changed_amp = _trim_step(
                cfg, tones, records, meas, trim_segment_s, premeasure_n_avgs,
                active_dofs=active_dofs)
            if changed_amp:
                print("  (raised amplitudes; Schroeder phases will be recomputed)")

        if args.premeasure_only:
            print("Pre-measure only: stopping before the full measurement.")
            return

        # ---------------- final full measurement --------------------------------
        # Use trim_segment_s (which may have grown during trim) for the final
        # measurement. capture_s = measure_n_avgs * trim_segment_s so we keep the
        # configured number of averages at the converged segment length.
        final_segment_s = trim_segment_s
        final_capture_s = measure_n_avgs * final_segment_s
        final_meas = {
            **measure,
            "capture_s":  final_capture_s,
            "min_time_s": max(measure["min_time_s"],
                              warmup + final_capture_s + margin),
        }
        _check_aborts(abort_event, sentinel)
        if cfg["schroeder"].get("enabled", True):
            assign_schroeder_phases(tones)
        print(f"[final] segment_s={final_segment_s:.2f}s "
              f"min_time={final_meas['min_time_s']}s capture={final_meas['capture_s']:.1f}s "
              f"-> inject+capture...")
        captured, fs, _ = inject_and_capture(cfg, tones, final_meas, run_dir,
                                             "final", abort_event, sentinel,
                                             active_dofs=active_dofs)
        _check_aborts(abort_event, sentinel)
        if cfg["analysis"].get("save_raw_capture", True):
            save_raw_capture(run_dir, captured, fs, tones, cfg_text)
        records = compute_tfs(captured, fs, tones, cfg, segment_s=final_segment_s,
                              active_dofs=active_dofs)

        # ---------------- analysis (shared with --analyze) ---------------------
        final_plots_dir = _resolve_plots_dir(cfg, run_dir, label=None)
        analyze_and_write(cfg, records, run_dir, cfg_text, electrodes,
                          plots_dir=final_plots_dir, active_dofs=active_dofs)

    except AbortRequested as e:
        print(f"\n*** ABORT: {e} ***  (excitation will be zeroed, POLES restored)")
        _kill_diag()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        _kill_diag()
    finally:
        abort_event.set()
        _kill_diag()
        guard.join(timeout=5)
        cleanup()
        print(f"Output: {run_dir}")


def _trim_step(cfg, tones, records, meas, segment_s: float, n_averages: int,
               active_dofs: list[str] | None = None):
    """Segment-first then amplitude. Returns (new_meas, new_segment_s, changed_amp).

    Step 1 — segment_s: double the Welch segment length (halves the bin, improves SNR
    per bin, raises true coherence) until trim.segment_s_max is reached. capture_s is
    always kept at n_averages * segment_s so the number of averages stays fixed and the
    coherence estimate reliability doesn't degrade as segments get longer.

    Tones were snapped to multiples of the original (minimum) segment_s bin, so they
    remain on-bin for any doubled segment_s (finer bins → original bins are a subset).

    Step 2 — amplitude: once segment_s is at its cap, raise amplitudes of the
    under-coherent electrodes by amp_step_factor up to max_amplitude_counts.
    """
    target = cfg["trim"]["target_coherence"]
    seg_max = float(cfg["trim"]["segment_s_max"])
    changed_amp = False
    new_meas = dict(meas)

    # 1) segment_s: double while under the cap
    warmup = float(cfg["analysis"]["warmup_s"])
    margin = 5.0
    new_seg = segment_s * 2.0
    if new_seg <= seg_max + 1e-9:
        new_meas["capture_s"] = round(n_averages * new_seg, 4)
        new_meas["min_time_s"] = max(new_meas["min_time_s"],
                                     warmup + new_meas["capture_s"] + margin)
        return new_meas, new_seg, changed_amp

    # 2) amplitude: raise under-coherent primary tones up to the cap
    dofs = active_dofs or list(cfg["dofs"].keys())
    primary = [d for d in dofs if d != "z"] or dofs
    step = cfg["amplitude"]["amp_step_factor"]
    cap = cfg["amplitude"]["max_amplitude_counts"]
    low_tones = set()
    for r in records:
        if r["dof_intended"] in primary and r["coh"][r["dof_intended"]] < target:
            low_tones.add((r["freq"], r["electrode"]))
    for t in tones:
        if (t.freq, t.electrode) in low_tones:
            newamp = min(t.amp_counts * step, cap)
            if newamp > t.amp_counts:
                t.amp_counts = newamp
                changed_amp = True
    return new_meas, segment_s, changed_amp


def _make_run_dir(cfg, label):
    root = Path(cfg["output_root"]).expanduser()
    t = time.localtime()
    date_dir = time.strftime("%y%m%d", t)
    ts = time.strftime("%Y%m%d_%H%M%S", t)
    lbl = label or cfg.get("run_label", "actgain")
    run_dir = root / date_dir / f"{ts}_{lbl}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _resolve_plots_dir(cfg: dict, run_dir: Path, label: str | None) -> Path:
    """Return the plot output directory for the final measurement.

    If cfg["plots_root"] is set (and not None/empty), plots go to
    plots_root/<run_dir.name>/ so they mirror the data layout but live in a
    separate tree. Otherwise, defaults to run_dir/plots/.
    """
    plots_root_raw = cfg.get("plots_root", None)
    if plots_root_raw:
        return Path(str(plots_root_raw)).expanduser() / run_dir.name
    return run_dir / "plots"


def _print_plan(tones, electrodes):
    print("  Tone plan (freq Hz -> electrode, intended DOF):")
    for t in sorted(tones, key=lambda x: x.freq):
        print(f"    {t.freq:8.3f}  {t.electrode}  {t.dof}")


def _kill_diag():
    """Best-effort fast stop of any running diag excitation."""
    subprocess.run(["pkill", "-f", "diag -l -f"], capture_output=True)


def _wait_for_awg_drain(cfg: dict, exc_channels: list[str],
                        timeout_s: float = 90.0, poll_s: float = 1.0) -> bool:
    """Poll NDS2 until every EXC channel RMS drops below 1 count, or timeout.

    Returns True if the AWG drained within timeout_s, False if it timed out.
    Called from cleanup() after GAIN=0 is confirmed written, so the AWG
    output cannot reach the DAC even while we wait. We poll anyway to know
    when it is safe to ramp GAIN back up.

    Uses a fresh NDS2 connection (the main one may have been closed by abort).
    Silently returns False on any NDS2 error so cleanup() always continues.
    """
    an = cfg.get("analysis", {})
    server = an.get("nds2_server", "192.168.1.11")
    port = int(an.get("nds2_port", 8088))
    threshold = 1.0          # counts — AWG output is zero when below this
    deadline = time.monotonic() + timeout_s
    try:
        conn = nds2.connection(server, port)
        for bufs in conn.iterate(exc_channels):
            quiet = all(
                float(np.sqrt(np.mean(np.asarray(b.data, dtype=np.float64) ** 2)))
                < threshold
                for b in bufs
            )
            if quiet:
                return True
            if time.monotonic() > deadline:
                return False
    except Exception:
        pass
    return False


if __name__ == "__main__":
    main()
