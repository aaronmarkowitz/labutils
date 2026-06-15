"""Pure-logic tests for ACTS snapshot/setup/restore and channel assignment."""
import pytest
import yaml

import measure_actuator_gain as mag


def _acts_cfg(electrode_row=None):
    if electrode_row is None:
        electrode_row = {"E1": 1, "E2": 2}
    return yaml.safe_load(f"""
prefix: "Y1:DMD"
acts:
  enabled: true
  electrode_row: {electrode_row}
safety:
  sw1_input_on_bit: 4
  sw2_output_on_bit: 1024
  restore_tramp_s: 2.0
""")


def test_assign_acts_channels_maps_correctly():
    """assign_acts_channels maps each tone to a distinct ACTS EXC channel."""
    cfg = _acts_cfg()
    tones = [
        mag.Tone(freq=10.0, electrode="E1", dof="x"),
        mag.Tone(freq=20.0, electrode="E1", dof="x"),
        mag.Tone(freq=30.0, electrode="E2", dof="y"),
    ]
    mag.assign_acts_channels(tones, cfg)
    chans = [t.channel for t in tones]
    assert chans == [
        "Y1:DMD-ACTS_1_1_EXC",
        "Y1:DMD-ACTS_1_2_EXC",
        "Y1:DMD-ACTS_2_1_EXC",
    ], f"unexpected channels: {chans}"
    assert len(set(chans)) == 3, "channels must be distinct"


def test_assign_acts_channels_raises_on_overflow():
    """>8 tones per electrode raises ValueError (8-column-per-row limit)."""
    cfg = _acts_cfg({"E1": 1})
    tones = [mag.Tone(freq=float(i), electrode="E1", dof="x") for i in range(9)]
    with pytest.raises(ValueError, match="only has 8 columns"):
        mag.assign_acts_channels(tones, cfg)


def test_assign_acts_channels_raises_on_awg_slot_limit():
    """Total tones > MAX_NUM_AWG=9 raises ValueError (AWG hardware limit)."""
    cfg = _acts_cfg({"E1": 1, "E2": 2})
    # 10 tones across two electrodes (5 each, within 8-col limit but 10 > 9 AWG slots)
    tones = (
        [mag.Tone(freq=float(i), electrode="E1", dof="x") for i in range(5)] +
        [mag.Tone(freq=float(i + 100), electrode="E2", dof="y") for i in range(5)]
    )
    with pytest.raises(ValueError, match="MAX_NUM_AWG"):
        mag.assign_acts_channels(tones, cfg)


def test_assign_acts_channels_accepts_exactly_max_awg():
    """Exactly MAX_NUM_AWG=9 tones across rows does NOT raise (boundary condition)."""
    cfg = _acts_cfg({"E1": 1, "E2": 2})
    tones = (
        [mag.Tone(freq=float(i), electrode="E1", dof="x") for i in range(5)] +
        [mag.Tone(freq=float(i + 100), electrode="E2", dof="y") for i in range(4)]
    )
    # Should not raise; 9 == MAX_NUM_AWG is exactly at the limit
    mag.assign_acts_channels(tones, cfg)


def test_assign_acts_channels_missing_electrode_raises():
    cfg = _acts_cfg()  # only E1, E2 defined
    tones = [mag.Tone(freq=1.0, electrode="E3", dof="x")]
    with pytest.raises(ValueError, match="not in acts.electrode_row"):
        mag.assign_acts_channels(tones, cfg)


def test_assign_acts_channels_8_tones_fills_all_columns():
    """8 tones on one electrode fills columns 1-8 exactly."""
    cfg = _acts_cfg({"E1": 3})
    tones = [mag.Tone(freq=float(i), electrode="E1", dof="x") for i in range(8)]
    mag.assign_acts_channels(tones, cfg)
    expected = [f"Y1:DMD-ACTS_3_{c}_EXC" for c in range(1, 9)]
    assert [t.channel for t in tones] == expected


def test_snapshot_acts_parses_batch_output(monkeypatch):
    """snapshot_acts returns correct snap dict from mocked caget output."""
    # SW1R=4 (input ON bit 4), SW2R=1024 (output ON bit 1024)
    fake_raw = {
        "Y1:DMD-ACTS_5_2_GAIN": 1.0,
        "Y1:DMD-ACTS_5_2_OFFSET": 0.0,
        "Y1:DMD-ACTS_5_2_TRAMP": 1.0,
        "Y1:DMD-ACTS_5_2_SW1R": 4.0,
        "Y1:DMD-ACTS_5_2_SW2R": 1024.0,
    }
    monkeypatch.setattr(mag, "caget_batch", lambda pvs: {p: fake_raw[p] for p in pvs if p in fake_raw})
    cfg = _acts_cfg({"E1": 5})
    snap = mag.snapshot_acts(cfg, ["Y1:DMD-ACTS_5_2_EXC"])
    s = snap["Y1:DMD-ACTS_5_2_EXC"]
    assert s["gain"] == 1.0
    assert s["input_on"] is True
    assert s["output_on"] is True


def test_snapshot_acts_detects_off_state(monkeypatch):
    """snapshot_acts correctly reports input/output OFF when bits are clear."""
    fake_raw = {
        "Y1:DMD-ACTS_8_1_GAIN": 0.0,
        "Y1:DMD-ACTS_8_1_OFFSET": 0.0,
        "Y1:DMD-ACTS_8_1_TRAMP": 1.0,
        "Y1:DMD-ACTS_8_1_SW1R": 0.0,    # input OFF (bit 4 not set)
        "Y1:DMD-ACTS_8_1_SW2R": 512.0,  # output OFF (bit 1024 not set in 512)
    }
    monkeypatch.setattr(mag, "caget_batch", lambda pvs: {p: fake_raw[p] for p in pvs if p in fake_raw})
    cfg = _acts_cfg({"LB": 8})
    snap = mag.snapshot_acts(cfg, ["Y1:DMD-ACTS_8_1_EXC"])
    s = snap["Y1:DMD-ACTS_8_1_EXC"]
    assert s["gain"] == 0.0
    assert s["input_on"] is False
    assert s["output_on"] is False
