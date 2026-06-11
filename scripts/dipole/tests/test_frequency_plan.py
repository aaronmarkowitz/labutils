"""Frequency-plan generation: guard-band exclusion, distinctness, density, snapping."""
import numpy as np
import pytest

import measure_actuator_gain as mag


def test_no_tone_in_guard_band(base_cfg):
    bin_hz = 1.0 / 8.0
    tones = mag.generate_frequency_plan(base_cfg, bin_hz)
    lo, hi = base_cfg["frequency_plan"]["guard_band_hz"]
    for t in tones:
        assert not (lo <= t.freq <= hi), f"tone {t.freq} landed in guard band"


def test_tones_distinct_and_separated(base_cfg):
    bin_hz = 1.0 / 8.0
    tones = mag.generate_frequency_plan(base_cfg, bin_hz)
    freqs = sorted(t.freq for t in tones)
    diffs = np.diff(freqs)
    assert np.all(diffs >= bin_hz - 1e-9), "tones closer than one FFT bin"
    assert len(set(round(f, 6) for f in freqs)) == len(freqs), "duplicate frequencies"


def test_every_electrode_driven_near_every_dof(base_cfg):
    bin_hz = 1.0 / 8.0
    tones = mag.generate_frequency_plan(base_cfg, bin_hz)
    elecs = set(base_cfg["electrodes"])
    for dof in ("x", "y", "z"):
        got = {t.electrode for t in tones if t.dof == dof}
        assert got == elecs, f"DOF {dof}: not all electrodes driven ({got})"


def test_xy_denser_than_z(base_cfg):
    bin_hz = 1.0 / 8.0
    tones = mag.generate_frequency_plan(base_cfg, bin_hz)
    n_x = sum(t.dof == "x" for t in tones)
    n_z = sum(t.dof == "z" for t in tones)
    assert n_x > n_z, "X cluster should have more tones than Z"


def test_tones_bin_snapped(base_cfg):
    bin_hz = 1.0 / 8.0
    tones = mag.generate_frequency_plan(base_cfg, bin_hz)
    for t in tones:
        assert abs(round(t.freq / bin_hz) * bin_hz - t.freq) < 1e-6


def test_infeasible_guard_raises(base_cfg):
    # force a DOF resonance into the guard band with tight spacing -> infeasible
    base_cfg["dofs"]["z"]["f0"] = 15.0
    base_cfg["dofs"]["z"]["tone_spacing_hz"] = 0.1
    with pytest.raises(ValueError):
        mag.generate_frequency_plan(base_cfg, 1.0 / 8.0)
