"""compute_tfs: recover known transfer functions from synthetic captured data.

This is the analysis we use instead of diag's (unreliable) multi-tone coefficient
output: inject the comb, capture raw channels over NDS2, and compute TF = S_ER/S_EE
with coherence from Welch averaging. Here we synthesize the captured timeseries with
known gains and confirm recovery.
"""
import numpy as np

import measure_actuator_gain as mag


def _multitone(freqs, amps, phases, fs, dur):
    t = np.arange(int(fs * dur)) / fs
    x = np.zeros_like(t)
    for f, a, p in zip(freqs, amps, phases):
        x += a * np.cos(2 * np.pi * f * t + p)
    return x


def test_compute_tfs_recovers_known_gains(base_cfg):
    fs, dur = 1024.0, 16.0
    base_cfg["analysis"]["segment_s"] = 2.0          # 0.5 Hz bins
    # distinct DOF channels so compute_tfs can tell them apart
    base_cfg["dofs"]["x"]["channel"] = "Y1:DMD-PARTICLE_X_IN1"
    base_cfg["dofs"]["y"]["channel"] = "Y1:DMD-PARTICLE_Y_IN1"
    base_cfg["dofs"]["z"]["channel"] = "Y1:DMD-PARTICLE_Z_IN1"

    # two electrodes, one tone each (snapped to 0.5 Hz bins)
    tones = [mag.Tone(freq=13.0, electrode="E1", dof="x", amp_counts=500.0),
             mag.Tone(freq=29.0, electrode="E2", dof="y", amp_counts=500.0)]
    mag.assign_schroeder_phases(tones)

    # known real gains into each DOF (X responds to E1, Y responds to E2; small cross)
    g = {("E1", "x"): 0.50, ("E1", "y"): 0.05, ("E1", "z"): 0.0,
         ("E2", "x"): 0.10, ("E2", "y"): 0.80, ("E2", "z"): 0.0}

    exc = {
        "Y1:DMD-POLES_E1_EXC": _multitone([13.0], [500.0], [tones[0].phase_rad], fs, dur),
        "Y1:DMD-POLES_E2_EXC": _multitone([29.0], [500.0], [tones[1].phase_rad], fs, dur),
    }
    captured = dict(exc)
    for d, ch in (("x", "Y1:DMD-PARTICLE_X_IN1"), ("y", "Y1:DMD-PARTICLE_Y_IN1"),
                  ("z", "Y1:DMD-PARTICLE_Z_IN1")):
        sig = (g[("E1", d)] * exc["Y1:DMD-POLES_E1_EXC"]
               + g[("E2", d)] * exc["Y1:DMD-POLES_E2_EXC"])
        captured[ch] = sig

    records = mag.compute_tfs(captured, fs, tones, base_cfg)
    by_e = {r["electrode"]: r for r in records}

    # E1 tone at 13 Hz -> recover gains into x/y
    assert abs(abs(by_e["E1"]["tf"]["x"]) - 0.50) < 1e-3
    assert abs(abs(by_e["E1"]["tf"]["y"]) - 0.05) < 1e-3
    # E2 tone at 29 Hz -> recover gains into x/y
    assert abs(abs(by_e["E2"]["tf"]["x"]) - 0.10) < 1e-3
    assert abs(abs(by_e["E2"]["tf"]["y"]) - 0.80) < 1e-3
    # noiseless -> coherence ~ 1
    assert by_e["E1"]["coh"]["x"] > 0.99
    assert by_e["E2"]["coh"]["y"] > 0.99


def test_compute_tfs_coherence_drops_with_noise(base_cfg):
    fs, dur = 1024.0, 16.0
    base_cfg["analysis"]["segment_s"] = 2.0
    base_cfg["dofs"]["x"]["channel"] = "Y1:DMD-PARTICLE_X_IN1"
    base_cfg["dofs"]["y"]["channel"] = "Y1:DMD-PARTICLE_Y_IN1"
    base_cfg["dofs"]["z"]["channel"] = "Y1:DMD-PARTICLE_Z_IN1"
    tones = [mag.Tone(freq=13.0, electrode="E1", dof="x", amp_counts=500.0)]
    mag.assign_schroeder_phases(tones)
    rng = np.random.default_rng(0)
    e = _multitone([13.0], [500.0], [0.0], fs, dur)
    captured = {"Y1:DMD-POLES_E1_EXC": e,
                "Y1:DMD-PARTICLE_X_IN1": 0.5 * e + rng.normal(0, 200, e.shape),
                "Y1:DMD-PARTICLE_Y_IN1": rng.normal(0, 200, e.shape),
                "Y1:DMD-PARTICLE_Z_IN1": rng.normal(0, 200, e.shape)}
    records = mag.compute_tfs(captured, fs, tones, base_cfg)
    r = records[0]
    # X still correlates with the drive; Y (pure noise) does not
    assert r["coh"]["x"] > r["coh"]["y"]
    assert r["coh"]["y"] < 0.5


def test_raw_capture_roundtrip_and_analyze(base_cfg, tmp_path):
    import yaml
    fs, dur = 1024.0, 16.0
    base_cfg["analysis"]["segment_s"] = 2.0
    for d in ("x", "y", "z"):
        base_cfg["dofs"][d]["channel"] = f"Y1:DMD-PARTICLE_{d.upper()}_IN1"
    t = np.arange(int(fs * dur)) / fs
    tones = [mag.Tone(freq=13.0, electrode="E1", dof="x", amp_counts=500.0),
             mag.Tone(freq=29.0, electrode="E2", dof="y", amp_counts=500.0)]
    e1 = 500 * np.cos(2 * np.pi * 13 * t)
    e2 = 500 * np.cos(2 * np.pi * 29 * t)
    captured = {"Y1:DMD-POLES_E1_EXC": e1, "Y1:DMD-POLES_E2_EXC": e2,
                "Y1:DMD-PARTICLE_X_IN1": 0.5 * e1, "Y1:DMD-PARTICLE_Y_IN1": 0.8 * e2,
                "Y1:DMD-PARTICLE_Z_IN1": np.zeros_like(t)}
    p = mag.save_raw_capture(tmp_path, captured, fs, tones, yaml.safe_dump(base_cfg))
    cap2, fs2, tones2, _ = mag.load_raw_capture(p)
    assert fs2 == fs and len(tones2) == 2
    assert np.allclose(cap2["Y1:DMD-POLES_E1_EXC"], e1, atol=1e-2)
    # compute_tfs runs on the reloaded capture and recovers the gains
    records = mag.compute_tfs(cap2, fs2, tones2, base_cfg)
    by_e = {r["electrode"]: r for r in records}
    assert abs(abs(by_e["E1"]["tf"]["x"]) - 0.5) < 1e-3
    assert abs(abs(by_e["E2"]["tf"]["y"]) - 0.8) < 1e-3


def test_analyze_and_write_writes_files(base_cfg, synthetic_records, tmp_path):
    """analyze_and_write (shared by the live run and --analyze) writes the matrix
    HDF5 + report from a full set of per-tone records."""
    records, _true_G, electrodes, _ = synthetic_records
    gm, fits = mag.analyze_and_write(base_cfg, records, tmp_path, "params_text", electrodes)
    assert gm.shape == (3, 4)
    assert (tmp_path / "actuator_gain_results.h5").exists()
    assert (tmp_path / "actuator_gain_report.txt").exists()
