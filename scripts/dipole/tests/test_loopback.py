"""Opt-in live hardware self-test using ACTS_8_1..8_4 -> LOS_IN1 physical loopback.

Uses ACTS row 8 columns 1-4 as four DISTINCT excitation channels (one per tone).
This gives diag sizeA == sizeExc, enabling its native per-tone coefficient
extraction to work correctly alongside our NDS2 path.

Run ONLY against the live Y1:DMD front end, with the loopback cable in place, and
ONLY when explicitly enabled:

    ACTGAIN_LOOPBACK=1 /var/lib/cds-conda/base/envs/cds-testing/bin/python3 \\
        -m pytest tests/ -m loopback

The ACTS channels are snapshotted and restored by the test. setup_acts_for_measurement
sets GAIN=1, input OFF, output ON on each used column; restore_acts reverses this.
"""
import os
import threading
from pathlib import Path

import numpy as np
import pytest

import measure_actuator_gain as mag

pytestmark = pytest.mark.loopback

# Four distinct ACTS row-8 EXC channels (GAIN=0 by default; test sets them to 1)
EXC_CHANNELS = [
    "Y1:DMD-ACTS_8_1_EXC",
    "Y1:DMD-ACTS_8_2_EXC",
    "Y1:DMD-ACTS_8_3_EXC",
    "Y1:DMD-ACTS_8_4_EXC",
]
READBACK = "Y1:DMD-LOS_IN1"
TONES_HZ = [29.0, 43.0, 61.0, 73.0]     # distinct, on 0.5 Hz bins, away from 10-20 Hz
EXPECTED_GAIN = 0.5                       # DAC/ADC full-scale ratio


def _enabled():
    """Require explicit opt-in (ACTGAIN_LOOPBACK=1) and reachable channels."""
    if os.environ.get("ACTGAIN_LOOPBACK") != "1":
        return False
    try:
        mag.caget_t("Y1:DMD-ACTS_8_1_GAIN")
        return True
    except Exception:
        return False


def _test_cfg():
    return mag.load_config(Path(__file__).resolve().parents[1] /
                           "measure_actuator_gain_config_test.yml")


def test_loopback_flat_real_transfer(tmp_path):
    if not _enabled():
        pytest.skip("set ACTGAIN_LOOPBACK=1 on a host that reaches the Y1:DMD FE to run")
    cfg = _test_cfg()

    # Build four tones, each on a distinct ACTS_8_N_EXC channel
    tones = [mag.Tone(freq=f, electrode="LB", dof="x", amp_counts=500.0,
                      channel=ch)
             for f, ch in zip(TONES_HZ, EXC_CHANNELS)]
    mag.assign_schroeder_phases(tones)

    # Override DOF readbacks to LOS_IN1
    for d in ("x", "y", "z"):
        cfg["dofs"][d]["channel"] = READBACK

    # Snapshot, set up, run, restore
    acts_snap = mag.snapshot_acts(cfg, EXC_CHANNELS)
    mag.setup_acts_for_measurement(cfg, EXC_CHANNELS, acts_snap, dry_run=False)
    try:
        meas = {"min_time_s": 12.0, "cycles": 10, "settling_frac": 0.1, "capture_s": 8.0}
        abort = threading.Event()
        captured, fs, result_xml = mag.inject_and_capture(
            cfg, tones, meas, tmp_path, "loopback", abort,
            Path("/tmp/__no_sentinel__"))
        records = mag.compute_tfs(captured, fs, tones, cfg)
    finally:
        mag.restore_acts(cfg, EXC_CHANNELS, acts_snap, dry_run=False)

    mags = np.array([abs(r["tf"]["x"]) for r in records])
    cohs = np.array([r["coh"]["x"] for r in records])
    assert result_xml.exists(), "diag result XML not written"
    assert np.all(cohs > 0.95), f"low coherence: {cohs}"
    assert np.allclose(mags, EXPECTED_GAIN, atol=0.02), \
        f"loopback gain not flat {EXPECTED_GAIN}: {mags}"


def test_loopback_phase_recovery(tmp_path):
    """compute_tfs recovers the correct relative phase through the loopback path.

    Uses ACTS_8_5_EXC and ACTS_8_6_EXC with different Schroeder phases; verifies that
    the phase difference between the two channels' TFs matches the injected difference.
    The loopback is a flat real cable — phase(TF) should equal the injected Schroeder phase
    modulo any constant loopback delay (which is common to both channels and cancels in
    the difference).
    """
    if not _enabled():
        pytest.skip("set ACTGAIN_LOOPBACK=1 on a host that reaches the Y1:DMD FE to run")
    cfg = _test_cfg()
    for d in ("x", "y", "z"):
        cfg["dofs"][d]["channel"] = READBACK

    exc_ch = ["Y1:DMD-ACTS_8_5_EXC", "Y1:DMD-ACTS_8_6_EXC"]
    tones = [
        mag.Tone(freq=37.0, electrode="LB", dof="x", amp_counts=500.0,
                 channel="Y1:DMD-ACTS_8_5_EXC", phase_rad=0.0),
        mag.Tone(freq=43.0, electrode="LB", dof="y", amp_counts=500.0,
                 channel="Y1:DMD-ACTS_8_6_EXC", phase_rad=np.pi / 2),
    ]

    acts_snap = mag.snapshot_acts(cfg, exc_ch)
    mag.setup_acts_for_measurement(cfg, exc_ch, acts_snap, dry_run=False)
    try:
        meas = {"min_time_s": 12.0, "cycles": 10, "settling_frac": 0.1, "capture_s": 8.0}
        abort = threading.Event()
        captured, fs, _ = mag.inject_and_capture(
            cfg, tones, meas, tmp_path, "phase", abort,
            Path("/tmp/__no_sentinel__"))
        records = mag.compute_tfs(captured, fs, tones, cfg)
    finally:
        mag.restore_acts(cfg, exc_ch, acts_snap, dry_run=False)

    by_e = {r["freq"]: r for r in records}
    phi0 = np.angle(by_e[37.0]["tf"]["x"])
    phi1 = np.angle(by_e[43.0]["tf"]["y"])
    # Phase difference should be π/2 (modulo loopback common delay)
    delta = (phi1 - phi0 + np.pi) % (2 * np.pi) - np.pi
    assert abs(delta - np.pi / 2) < 0.1, f"phase difference wrong: {delta:.4f} rad (expected π/2)"


def test_loopback_coherence_vs_averages(tmp_path):
    """Longer capture (more Welch averages) gives higher coherence when drive is low.

    Uses ACTS_8_7_EXC at low amplitude so thermal/electronic noise is non-negligible.
    Coherence with 4 s capture should be lower than with 16 s capture.
    """
    if not _enabled():
        pytest.skip("set ACTGAIN_LOOPBACK=1 on a host that reaches the Y1:DMD FE to run")
    cfg = _test_cfg()
    for d in ("x", "y", "z"):
        cfg["dofs"][d]["channel"] = READBACK

    exc_ch = ["Y1:DMD-ACTS_8_7_EXC"]
    tone = mag.Tone(freq=31.0, electrode="LB", dof="x", amp_counts=10.0,
                    channel="Y1:DMD-ACTS_8_7_EXC", phase_rad=0.0)

    acts_snap = mag.snapshot_acts(cfg, exc_ch)
    mag.setup_acts_for_measurement(cfg, exc_ch, acts_snap, dry_run=False)
    try:
        abort = threading.Event()
        sentinel = Path("/tmp/__no_sentinel__")
        meas_short = {"min_time_s": 6.0, "cycles": 5, "settling_frac": 0.1, "capture_s": 4.0}
        meas_long  = {"min_time_s": 18.0, "cycles": 5, "settling_frac": 0.1, "capture_s": 16.0}
        cap_short, fs, _ = mag.inject_and_capture(cfg, [tone], meas_short, tmp_path,
                                                   "coh_short", abort, sentinel)
        cap_long, fs, _  = mag.inject_and_capture(cfg, [tone], meas_long,  tmp_path,
                                                   "coh_long",  abort, sentinel)
    finally:
        mag.restore_acts(cfg, exc_ch, acts_snap, dry_run=False)

    coh_short = mag.compute_tfs(cap_short, fs, [tone], cfg)[0]["coh"]["x"]
    coh_long  = mag.compute_tfs(cap_long,  fs, [tone], cfg)[0]["coh"]["x"]
    assert coh_long >= coh_short, (
        f"coherence did not improve: short={coh_short:.3f} long={coh_long:.3f}")


def test_guard_aborts_on_injected_band_power():
    if not _enabled():
        pytest.skip("set ACTGAIN_LOOPBACK=1 on a host that reaches the Y1:DMD FE to run")
    # Manual procedure: with a comb running on ACTS_8_*, inject a growing 10-20 Hz
    # component and confirm GuardMonitor trips and excitation is ramped to zero.
    pytest.skip("manual: inject growing 10-20 Hz on ACTS_8_*, confirm guard trips")
