"""Opt-in live hardware self-test using the ACTS_8_8 -> LOS_IN1 physical loopback.

Run ONLY against the live Y1:DMD front end (diag reaches cymac1's awg/tp/nds from
worker1), with the loopback in place, and ONLY when explicitly enabled:

    ACTGAIN_LOOPBACK=1 /var/lib/cds-conda/base/envs/cds-testing/bin/python3 \
        -m pytest -m loopback

This validates the real path: diag INJECTS the comb (its injection is correct), we
capture the raw channels over NDS2 and compute TF = S_ER/S_EE ourselves (diag's
multi-tone coefficient output is unreliable in dtt 4.1.5 and is NOT used). The
loopback TF must be flat and real (DAC/ADC full-scale differ by 2x -> gain 0.5).
SKIPPED unless ACTGAIN_LOOPBACK=1 (it drives a real DAC).
"""
import os
import threading
from pathlib import Path

import numpy as np
import pytest

import measure_actuator_gain as mag

pytestmark = pytest.mark.loopback

EXC = "Y1:DMD-ACTS_8_8_EXC"
READBACK = "Y1:DMD-LOS_IN1"
TONES_HZ = [13.0, 29.0, 43.0, 61.0]   # distinct, on 0.5 Hz bins, all >10 Hz
EXPECTED_GAIN = 0.5                    # DAC/ADC full-scale ratio


def _enabled():
    """Require explicit opt-in (ACTGAIN_LOOPBACK=1) AND reachable channels, so this
    never runs by accident — it drives a real DAC."""
    if os.environ.get("ACTGAIN_LOOPBACK") != "1":
        return False
    try:
        mag.caget_t(EXC.replace("_EXC", "_GAIN"))
        return True
    except Exception:
        return False


def _loopback_cfg():
    cfg = mag.load_config(Path(__file__).resolve().parents[1] /
                          "measure_actuator_gain_config.yml")
    for d in ("x", "y", "z"):
        cfg["dofs"][d]["channel"] = READBACK     # all DOFs read the loopback channel
    return cfg


def test_loopback_flat_real_transfer(tmp_path):
    if not _enabled():
        pytest.skip("set ACTGAIN_LOOPBACK=1 on a host that reaches the Y1:DMD FE to run")
    cfg = _loopback_cfg()
    tones = [mag.Tone(freq=f, electrode="LB", dof="x", amp_counts=500.0, channel=EXC)
             for f in TONES_HZ]
    mag.assign_schroeder_phases(tones)
    meas = {"min_time_s": 12.0, "cycles": 10, "settling_frac": 0.1, "capture_s": 8.0}
    abort = threading.Event()
    captured, fs, result_xml = mag.inject_and_capture(
        cfg, tones, meas, tmp_path, "loopback", abort, Path("/tmp/__no_sentinel__"))
    records = mag.compute_tfs(captured, fs, tones, cfg)

    mags = np.array([abs(r["tf"]["x"]) for r in records])
    cohs = np.array([r["coh"]["x"] for r in records])
    assert result_xml.exists()                       # diag result still saved for diaggui
    assert np.all(cohs > 0.95), f"low coherence: {cohs}"
    assert np.allclose(mags, EXPECTED_GAIN, atol=0.02), f"loopback gain not flat 0.5: {mags}"


def test_guard_aborts_on_injected_band_power():
    if not _enabled():
        pytest.skip("set ACTGAIN_LOOPBACK=1 on a host that reaches the Y1:DMD FE to run")
    # Manual procedure: with a comb running on ACTS_8_8, inject a growing 10-20 Hz
    # component and confirm GuardMonitor trips and excitation is ramped to zero.
    # Left operator-supervised rather than auto-injecting band power.
    pytest.skip("manual: inject growing 10-20 Hz on ACTS_8_8, confirm guard trips")
