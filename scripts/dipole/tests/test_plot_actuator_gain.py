"""Tests for plot_actuator_gain.py — pure logic, no hardware, no display."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Make the dipole scripts directory importable.
_DIPOLE_DIR = Path(__file__).resolve().parents[1]
if str(_DIPOLE_DIR) not in sys.path:
    sys.path.insert(0, str(_DIPOLE_DIR))

import plot_actuator_gain as pag   # noqa: E402
import measure_actuator_gain as mag  # noqa: E402


# --------------------------------------------------------------------------- #
# Helper: build minimal synthetic inputs for plot_measurement
# --------------------------------------------------------------------------- #
def _make_dof_fits(electrodes, f0s=None, Qs=None):
    """Return a dof_fits dict with SimpleNamespace objects."""
    if f0s is None:
        f0s = {"x": 39.0, "y": 54.0, "z": 5.0}
    if Qs is None:
        Qs = {"x": 30.0, "y": 30.0, "z": 10.0}
    n = len(electrodes)
    dof_fits = {}
    for d in ("x", "y", "z"):
        # complex gains: all 1+0j for simplicity
        gains = np.ones(n, dtype=complex)
        dof_fits[d] = SimpleNamespace(
            dof=d,
            f0=f0s[d],
            Q=Qs[d],
            gains=gains,
            fit_plant=True,
            residual_norm=0.0,
            per_electrode_coherence={e: 0.99 for e in electrodes},
        )
    return dof_fits


def _make_records(electrodes, dof_fits):
    """Build synthetic per-tone records for the given electrodes."""
    records = []
    dof_params = {d: (dof_fits[d].f0, dof_fits[d].Q) for d in ("x", "y", "z")}
    for dof, (f0, Q) in dof_params.items():
        n = 4 if dof != "z" else 2
        spacing = 1.5 if dof != "z" else 0.6
        offs = (np.arange(n) - (n - 1) / 2.0) * spacing
        for j, off in enumerate(offs):
            freq = float(f0 + off)
            e = electrodes[j % len(electrodes)]
            ei = electrodes.index(e)
            tf = {}
            coh = {}
            for d2, (f0_2, Q_2) in dof_params.items():
                H = pag._plant_lorentzian(freq, f0_2, Q_2)
                G = dof_fits[d2].gains[ei]
                tf[d2] = complex(G * H)
                coh[d2] = 0.97
            records.append({"electrode": e, "dof_intended": dof,
                             "freq": freq, "tf": tf, "coh": coh})
    return records


def _minimal_cfg(electrodes):
    return {
        "dofs": {
            "x": {"f0": 39.0, "Q": 30.0},
            "y": {"f0": 54.0, "Q": 30.0},
            "z": {"f0": 5.0,  "Q": 10.0},
        },
        "electrodes": electrodes,
    }


# --------------------------------------------------------------------------- #
# Test 1 — expected files are created
# --------------------------------------------------------------------------- #
def test_plot_measurement_creates_expected_files(tmp_path):
    """plot_measurement creates bode_E*.png, bode_combined.png, gain_matrix.png."""
    electrodes = ["E1", "E2", "E3", "E4"]
    dof_fits = _make_dof_fits(electrodes)
    records = _make_records(electrodes, dof_fits)
    cfg = _minimal_cfg(electrodes)

    plots_dir = tmp_path / "plots"
    written = pag.plot_measurement(records, dof_fits, electrodes, cfg, plots_dir,
                                    is_trim=False)

    written_names = {p.name for p in written}
    for e in electrodes:
        assert f"bode_{e}.png" in written_names, f"missing bode_{e}.png"
        assert (plots_dir / f"bode_{e}.png").exists()
    assert "bode_combined.png" in written_names
    assert "gain_matrix.png" in written_names
    assert (plots_dir / "bode_combined.png").exists()
    assert (plots_dir / "gain_matrix.png").exists()


# --------------------------------------------------------------------------- #
# Test 2 — trim plot skips gain matrix
# --------------------------------------------------------------------------- #
def test_trim_plot_skips_gain_matrix(tmp_path):
    """is_trim=True: bode files exist but gain_matrix.png does NOT."""
    electrodes = ["E1", "E2", "E3", "E4"]
    dof_fits = _make_dof_fits(electrodes)
    records = _make_records(electrodes, dof_fits)
    cfg = _minimal_cfg(electrodes)

    plots_dir = tmp_path / "plots"
    written = pag.plot_measurement(records, dof_fits, electrodes, cfg, plots_dir,
                                    is_trim=True)

    written_names = {p.name for p in written}
    for e in electrodes:
        assert f"bode_{e}.png" in written_names, f"missing bode_{e}.png in trim plot"
    assert "bode_combined.png" in written_names
    # gain matrix must be absent for trim plots
    assert "gain_matrix.png" not in written_names
    assert not (plots_dir / "gain_matrix.png").exists()


# --------------------------------------------------------------------------- #
# Test 3 — single electrode
# --------------------------------------------------------------------------- #
def test_plot_handles_single_electrode(tmp_path):
    """Single electrode: no crash, bode_E1.png + bode_combined.png + gain_matrix.png."""
    electrodes = ["E1"]
    dof_fits = _make_dof_fits(electrodes)
    records = _make_records(electrodes, dof_fits)
    cfg = _minimal_cfg(electrodes)

    plots_dir = tmp_path / "plots"
    written = pag.plot_measurement(records, dof_fits, electrodes, cfg, plots_dir,
                                    is_trim=False)

    written_names = {p.name for p in written}
    assert "bode_E1.png" in written_names
    assert "bode_combined.png" in written_names
    assert "gain_matrix.png" in written_names


# --------------------------------------------------------------------------- #
# Test 4 — high tone count pure-logic validation
# --------------------------------------------------------------------------- #
def test_loopback_high_tone_count():
    """Pure-logic test: 30-tone frequency plan.

    (a) All 30 tones are generated.
    (b) No tone lands in guard band [10, 20] Hz.
    (c) All frequencies are globally distinct.
    (d) assign_acts_channels raises ValueError when 12 tones/electrode > 8 ACTS cols.
    """
    import yaml

    cfg_text = """
prefix: "Y1:DMD"
electrodes: ["E1","E2","E3","E4"]
dofs:
  x: {channel: "Y1:DMD-PARTICLE_X_IN1", f0: 39.0, Q: 30.0, n_tones: 12, tone_spacing_hz: 0.5, fit_plant: true}
  y: {channel: "Y1:DMD-PARTICLE_Y_IN1", f0: 54.0, Q: 30.0, n_tones: 12, tone_spacing_hz: 0.5, fit_plant: true}
  z: {channel: "Y1:DMD-PARTICLE_Z_IN1", f0: 5.0,  Q: 10.0, n_tones:  6, tone_spacing_hz: 0.4, fit_plant: false}
measurement_channel_rate: 65536
frequency_plan: {guard_band_hz: [10.0, 20.0], min_bin_separation: 1, fft_bin_snap: true}
amplitude: {max_amplitude_counts: 32000, initial_amplitude_counts: 1000, amp_step_factor: 1.7}
schroeder: {enabled: true}
trim: {target_coherence: 0.9, max_trim_iters: 4}
diag:
  premeasure: {min_time_s: 12.0, cycles: 10, settling_frac: 0.1}
  measure: {min_time_s: 28.0, cycles: 10, settling_frac: 0.1}
  rampup_s: 1.0
  rampdown_s: 1.0
  average_type: 0
  window: 1
  diag_timeout_s: 300
analysis:
  nds2_server: "127.0.0.1"
  nds2_port: 8088
  segment_s: 2.0
  premeasure_capture_s: 6.0
  measure_capture_s: 20.0
  warmup_s: 3.0
guard_monitor:
  nds2_server: "127.0.0.1"
  nds2_port: 8088
  channels: ["Y1:DMD-PARTICLE_X_IN1"]
  band_hz: [10.0, 20.0]
  factor: 2.0
  baseline_seconds: 5
safety: {poles_assume_open_loop: true, sw1_input_on_bit: 4, sw2_output_on_bit: 1024, restore_tramp_s: 2.0}
acts:
  enabled: true
  electrode_row: {E1: 1, E2: 2, E3: 3, E4: 4}
abort: {sentinel_path: "/tmp/abort_test_30tone"}
output_root: "/tmp/actgain_30tone_test"
run_label: "test30"
"""
    cfg = yaml.safe_load(cfg_text)
    bin_hz = 0.5   # 1 / segment_s = 1 / 2.0

    tones = mag.generate_frequency_plan(cfg, bin_hz)

    # (a) total tone count
    assert len(tones) == 30, f"expected 30 tones, got {len(tones)}"

    # (b) no tone in guard band
    guard_lo, guard_hi = 10.0, 20.0
    for t in tones:
        assert not (guard_lo <= t.freq <= guard_hi), \
            f"tone {t.freq:.3f} Hz landed in guard band [{guard_lo}, {guard_hi}]"

    # (c) all frequencies globally distinct
    freqs = [t.freq for t in tones]
    rounded = [round(f, 6) for f in freqs]
    assert len(set(rounded)) == len(rounded), \
        f"duplicate frequencies detected: {sorted(rounded)}"

    # (d) assign_acts_channels raises ValueError because 12 tones/electrode > 8 cols
    # Electrode E1 gets 3 tones from x + 3 from y + ... depending on round-robin,
    # but with 12 x-tones / 4 electrodes = 3 per electrode from x alone,
    # and 12 y-tones / 4 electrodes = 3 per electrode from y alone,
    # and 6 z-tones / 4 electrodes = ceil(6/4)=2 per electrode from z:
    # total per electrode = 3+3+2 = 8 (borderline) or 3+3+1 = 7 (for some).
    # Rewrite the config with larger n_tones to guarantee overflow:
    cfg["dofs"]["x"]["n_tones"] = 16  # 16/4 = 4 from x alone
    cfg["dofs"]["y"]["n_tones"] = 16  # 4 from y -> 4+4+2 = 10 per electrode > 8
    cfg["dofs"]["z"]["n_tones"] = 8
    tones_overflow = mag.generate_frequency_plan(cfg, bin_hz)
    with pytest.raises(ValueError, match="MAX_NUM_AWG|only has 8 columns"):
        mag.assign_acts_channels(tones_overflow, cfg)
