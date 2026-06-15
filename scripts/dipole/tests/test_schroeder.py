"""Schroeder phasing: crest-factor reduction and basic invariants."""
import numpy as np

import measure_actuator_gain as mag


def test_phases_in_range():
    amps = np.array([1.0, 2.0, 0.5, 1.5, 1.0])
    ph = mag.schroeder_phases(amps)
    assert ph.shape == amps.shape
    assert np.all(ph >= 0) and np.all(ph < 2 * np.pi + 1e-9)


def test_crest_factor_reduced_vs_zero_phase():
    # 8 equal-amplitude tones; Schroeder phasing should beat all-zero phases.
    n = 8
    amps = np.ones(n)
    freqs = np.arange(1, n + 1) * 3.0  # 3,6,...,24 Hz, all resolvable at fs/dur below
    zero_cf = mag.crest_factor(amps, freqs, np.zeros(n), fs=2048.0, dur=4.0)
    sch_cf = mag.crest_factor(amps, freqs, mag.schroeder_phases(amps), fs=2048.0, dur=4.0)
    assert sch_cf < zero_cf, f"Schroeder crest {sch_cf:.2f} not < zero-phase {zero_cf:.2f}"
    # zero-phase comb crest factor approaches sqrt(2*n); Schroeder should be far lower.
    assert sch_cf < 0.6 * zero_cf


def test_recompute_changes_with_amplitudes():
    a1 = np.array([1.0, 1.0, 1.0])
    a2 = np.array([1.0, 3.0, 1.0])
    assert not np.allclose(mag.schroeder_phases(a1), mag.schroeder_phases(a2))


def test_assign_schroeder_phases_per_electrode():
    tones = [mag.Tone(freq=f, electrode=e, dof="x", amp_counts=1.0)
             for e in ("E1", "E2") for f in (38.0, 39.0, 40.0)]
    mag.assign_schroeder_phases(tones)
    # first tone of each electrode group has phase 0 (cumulative sum empty)
    e1 = [t for t in tones if t.electrode == "E1"]
    assert abs(e1[0].phase_rad) < 1e-9
