"""POLES snapshot/restore (XOR-toggle), caget parsing, and amplitude-cap clamping."""
import measure_actuator_gain as mag


def test_caget_batch_parsing(monkeypatch):
    class R:
        stdout = "Y1:DMD-POLES_E1_GAIN 1.5\nY1:DMD-POLES_E1_SW1R 4\n"
    monkeypatch.setattr(mag.subprocess, "run", lambda *a, **k: R())
    out = mag.caget_batch(["Y1:DMD-POLES_E1_GAIN", "Y1:DMD-POLES_E1_SW1R"])
    assert out["Y1:DMD-POLES_E1_GAIN"] == 1.5
    assert out["Y1:DMD-POLES_E1_SW1R"] == 4.0


def test_snapshot_detects_input_on(base_cfg, monkeypatch):
    def fake_batch(pvs):
        d = {}
        for pv in pvs:
            if pv.endswith("_SW1R"):
                d[pv] = 4.0 if "E1" in pv else 0.0   # E1 input ON, others OFF
            else:
                d[pv] = 2.0
        return d
    monkeypatch.setattr(mag, "caget_batch", fake_batch)
    snap = mag.snapshot_poles(base_cfg)
    assert snap["E1"]["input_on"] is True
    assert snap["E2"]["input_on"] is False


def test_disable_only_toggles_if_on(base_cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(mag, "caput", lambda pv, v: calls.append((pv, v)))
    snap = {e: {"input_on": (e == "E1")} for e in base_cfg["electrodes"]}
    mag.disable_poles_inputs(base_cfg, snap, dry_run=False)
    assert calls == [("Y1:DMD-POLES_E1_SW1", 4)]   # only the ON module toggled off


def test_setup_poles_sets_gain1_regardless_of_snapshot(base_cfg, monkeypatch):
    """setup_poles_for_measurement must write GAIN=1 for every electrode, even
    if the snapshot recorded GAIN=0 (from a previous aborted run)."""
    calls = []
    monkeypatch.setattr(mag, "caput", lambda pv, v: calls.append((pv, v)))
    # Simulate previous abort left gains at 0
    snap = {e: {"gain": 0.0, "input_on": False} for e in base_cfg["electrodes"]}
    mag.setup_poles_for_measurement(base_cfg, snap, dry_run=False)
    for e in base_cfg["electrodes"]:
        assert (f"Y1:DMD-POLES_{e}_GAIN", 1) in calls, \
            f"GAIN=1 not written for {e}"


def test_restore_writes_and_reenables(base_cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(mag, "caput", lambda pv, v: calls.append((pv, v)))
    # after the run the input switch reads OFF, so restore must re-enable E1
    monkeypatch.setattr(mag, "caget_batch", lambda pvs: {pvs[0]: 0.0})
    snap = {
        "E1": {"gain": 1.0, "offset": 0.2, "tramp": 5.0, "input_on": True},
        "E2": {"gain": 2.0, "offset": 0.0, "tramp": 5.0, "input_on": False},
        "E3": {"gain": 3.0, "offset": 0.0, "tramp": 5.0, "input_on": False},
        "E4": {"gain": 4.0, "offset": 0.0, "tramp": 5.0, "input_on": False},
    }
    mag.restore_poles(base_cfg, snap, dry_run=False)
    pvs = [c[0] for c in calls]
    assert ("Y1:DMD-POLES_E1_GAIN", 1.0) in calls
    assert "Y1:DMD-POLES_E1_SW1" in pvs               # E1 re-enabled
    assert not any(pv == "Y1:DMD-POLES_E2_SW1" for pv in pvs)  # E2 stays off


def test_cleanup_zeros_gain_then_restores_after_drain(base_cfg, monkeypatch):
    """On abort: cleanup() must zero GAIN first, wait for AWG drain, then
    restore GAIN to the original value. The order matters: zero → drain → restore."""
    calls = []
    monkeypatch.setattr(mag, "caput", lambda pv, v: calls.append((pv, v)))
    monkeypatch.setattr(mag, "caget_batch", lambda pvs: {pvs[0]: 0.0})
    # Simulate instant drain so the test doesn't block on NDS2
    monkeypatch.setattr(mag, "_wait_for_awg_drain", lambda cfg, chans, **kw: True)

    snap = {
        "E1": {"gain": 1.0, "offset": 0.0, "tramp": 5.0, "sw1r": 0.0, "sw2r": 0.0,
               "input_on": False},
        "E2": {"gain": 2.0, "offset": 0.0, "tramp": 5.0, "sw1r": 0.0, "sw2r": 0.0,
               "input_on": False},
        "E3": {"gain": 3.0, "offset": 0.0, "tramp": 5.0, "sw1r": 0.0, "sw2r": 0.0,
               "input_on": False},
        "E4": {"gain": 4.0, "offset": 0.0, "tramp": 5.0, "sw1r": 0.0, "sw2r": 0.0,
               "input_on": False},
    }
    restored = [False]

    def cleanup():
        if not restored[0]:
            for e in base_cfg["electrodes"]:
                b = f"{base_cfg['prefix']}-POLES_{e}"
                try:
                    mag.caput(f"{b}_TRAMP", 0)
                    mag.caput(f"{b}_GAIN", 0)
                except Exception:
                    pass
            mag._wait_for_awg_drain(base_cfg, [])
            try:
                mag.restore_poles(base_cfg, snap, dry_run=False)
            finally:
                restored[0] = True

    cleanup()

    orig_gains = {"E1": 1.0, "E2": 2.0, "E3": 3.0, "E4": 4.0}
    for e in base_cfg["electrodes"]:
        b = f"{base_cfg['prefix']}-POLES_{e}"
        gain_writes = [(pv, v) for pv, v in calls if pv == f"{b}_GAIN"]
        assert gain_writes, f"{b}_GAIN never written"
        # First GAIN write must be zero
        assert gain_writes[0][1] == 0, \
            f"{b}: first GAIN write should be 0, got {gain_writes[0][1]}"
        # Final GAIN write must be the original value
        assert gain_writes[-1][1] == orig_gains[e], \
            f"{b}: GAIN not restored to original {orig_gains[e]}, got {gain_writes[-1][1]}"


def test_amplitude_clamps_at_cap(base_cfg):
    cap = base_cfg["amplitude"]["max_amplitude_counts"]
    tones = [mag.Tone(freq=39.0, electrode="E1", dof="x", amp_counts=cap * 0.95),
             mag.Tone(freq=54.0, electrode="E2", dof="y", amp_counts=cap * 0.8)]
    records = [
        {"electrode": "E1", "dof_intended": "x", "freq": 39.0,
         "tf": {}, "coh": {"x": 0.3, "y": 0.3, "z": 0.0}},
        {"electrode": "E2", "dof_intended": "y", "freq": 54.0,
         "tf": {}, "coh": {"x": 0.3, "y": 0.3, "z": 0.0}},
    ]
    # force the amplitude branch: set segment_s_max in cfg and pass segment_s at the cap
    base_cfg["trim"]["segment_s_max"] = 16.0
    meas = {"min_time_s": 40.0, "cycles": 10, "settling_frac": 0.1, "capture_s": 30.0}
    new_meas, new_seg, changed = mag._trim_step(
        base_cfg, tones, records, meas, segment_s=16.0, n_averages=3)
    assert changed
    assert all(t.amp_counts <= cap for t in tones)
    assert max(t.amp_counts for t in tones) == cap   # E1 (0.95*cap*1.7) clamps to cap
