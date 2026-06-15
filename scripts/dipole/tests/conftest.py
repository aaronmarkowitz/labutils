"""Shared pytest fixtures and path setup for measure_actuator_gain tests."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

# Make the script importable (it lives one directory up).
_DIPOLE_DIR = Path(__file__).resolve().parents[1]
if str(_DIPOLE_DIR) not in sys.path:
    sys.path.insert(0, str(_DIPOLE_DIR))

import measure_actuator_gain as mag  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_RESULT = DATA_DIR / "sample_sine_result.xml"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "loopback: live hardware self-test via ACTS_8_8->LOS_IN1 loopback")


@pytest.fixture
def base_cfg():
    """A minimal-but-complete config dict for the pure-logic functions."""
    return yaml.safe_load("""
prefix: "Y1:DMD"
electrodes: ["E1","E2","E3","E4"]
dofs:
  x: {channel: "Y1:DMD-PARTICLE_X_IN1", f0: 39.0, Q: 30.0, n_tones: 6, tone_spacing_hz: 1.5, fit_plant: true}
  y: {channel: "Y1:DMD-PARTICLE_Y_IN1", f0: 54.0, Q: 30.0, n_tones: 6, tone_spacing_hz: 1.5, fit_plant: true}
  z: {channel: "Y1:DMD-PARTICLE_Z_IN1", f0: 5.0,  Q: 10.0, n_tones: 4, tone_spacing_hz: 0.6, fit_plant: false}
measurement_channel_rate: 65536
frequency_plan: {guard_band_hz: [10.0, 20.0], min_bin_separation: 1, fft_bin_snap: true}
amplitude: {max_amplitude_counts: 32000, initial_amplitude_counts: 1000, amp_step_factor: 1.7}
schroeder: {enabled: true}
trim: {target_coherence: 0.9, max_trim_iters: 4, time_first: true, z_max_meas_time_s: 60.0, z_accept_low_coherence: true}
diag:
  template_xml: ""
  premeasure: {min_time_s: 12.0, cycles: 10, settling_frac: 0.1}
  measure: {min_time_s: 28.0, cycles: 10, settling_frac: 0.1}
  rampup_s: 1.0
  rampdown_s: 1.0
  average_type: 0
  window: 1
  diag_timeout_s: 300
analysis:
  nds2_server: "192.168.1.11"
  nds2_port: 8088
  segment_s: 2.0
  premeasure_capture_s: 6.0
  measure_capture_s: 20.0
  warmup_s: 3.0
guard_monitor:
  nds2_server: "127.0.0.1"
  nds2_port: 8088
  channels: ["Y1:DMD-PARTICLE_X_IN1","Y1:DMD-PARTICLE_Y_IN1","Y1:DMD-PARTICLE_Z_IN1"]
  band_hz: [10.0, 20.0]
  factor: 2.0
  baseline_seconds: 5
  poll_seconds: 1
safety: {poles_assume_open_loop: true, sw1_input_on_bit: 4, sw2_output_on_bit: 1024, restore_tramp_s: 2.0}
acts:
  enabled: true
  electrode_row: {E1: 1, E2: 2, E3: 3, E4: 4}
abort: {sentinel_path: "/tmp/abort_actuator_gain_test"}
output_root: "/tmp/actgain_test_out"
run_label: "actgain"
""")


@pytest.fixture
def synthetic_records():
    """Build noiseless per-tone records from a known G matrix + plant.

    Returns (records, true_G, electrodes, dof_params).
    """
    electrodes = ["E1", "E2", "E3", "E4"]
    dof_params = {"x": (39.0, 30.0), "y": (54.0, 30.0), "z": (5.0, 10.0)}
    rng = np.random.default_rng(0)
    # known complex gains, rows x/y/z, cols E1..E4
    true_G = {
        "x": np.array([1.0, 0.6 + 0.2j, -0.4, 0.1 - 0.3j]),
        "y": np.array([0.2j, 1.0, 0.5, -0.7 + 0.1j]),
        "z": np.array([0.05, -0.05, 0.05, -0.05]),  # symmetry-suppressed
    }

    def plant(f, f0, Q):
        return mag.plant_lorentzian(f, f0, Q)

    # build a tone plan: per dof, n tones near f0, electrodes round-robin
    tones = []
    for dof, (f0, Q) in dof_params.items():
        n = 6 if dof in ("x", "y") else 4
        spacing = 1.5 if dof in ("x", "y") else 0.6
        offs = (np.arange(n) - (n - 1) / 2) * spacing
        for j, off in enumerate(offs):
            tones.append((float(f0 + off), electrodes[j % 4], dof))

    records = []
    for freq, elec, dof in tones:
        ei = electrodes.index(elec)
        tf = {}
        coh = {}
        for d, (f0, Q) in dof_params.items():
            val = true_G[d][ei] * plant(freq, f0, Q)
            tf[d] = complex(val)
            coh[d] = 0.99
        records.append({"electrode": elec, "dof_intended": dof,
                        "freq": freq, "tf": tf, "coh": coh})
    return records, true_G, electrodes, dof_params
