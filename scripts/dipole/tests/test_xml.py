"""diaggui XML generation: round-trip the index-aligned Stimulus arrays."""
import xml.etree.ElementTree as ET

import measure_actuator_gain as mag


def _params(xml_text):
    root = ET.fromstring(xml_text)
    out = {}
    for p in root.iter("Param"):
        out[p.get("Name")] = (p.text or "").strip()
    return out


def test_xml_roundtrip_index_aligned(base_cfg):
    tones = [
        mag.Tone(freq=38.0, electrode="E1", dof="x", amp_counts=1000.0, phase_rad=0.1),
        mag.Tone(freq=39.5, electrode="E2", dof="x", amp_counts=1500.0, phase_rad=0.2),
        mag.Tone(freq=5.0, electrode="E3", dof="z", amp_counts=800.0, phase_rad=0.3),
    ]
    meas = base_cfg["diag"]["measure"]
    xml_text = mag.build_sine_response_xml(base_cfg, tones, meas)
    p = _params(xml_text)

    assert p["TestType"] == "SineResponse"
    assert p["Subtype"] == "SineResponse"
    # rows are emitted sorted by frequency: z(5), x(38), x(39.5)
    assert float(p["StimulusFrequency[0]"]) == 5.0
    assert p["StimulusChannel[0]"] == "Y1:DMD-POLES_E3_EXC"
    assert float(p["StimulusFrequency[1]"]) == 38.0
    assert p["StimulusChannel[1]"] == "Y1:DMD-POLES_E1_EXC"
    assert float(p["StimulusFrequency[2]"]) == 39.5
    assert p["StimulusChannel[2]"] == "Y1:DMD-POLES_E2_EXC"
    # amplitude/phase index-aligned with the same sort order
    assert float(p["StimulusAmplitude[0]"]) == 800.0
    assert abs(float(p["StimulusPhase[2]"]) - 0.2) < 1e-9


def test_measurement_channels_are_particle_dofs(base_cfg):
    tones = [mag.Tone(freq=39.0, electrode="E1", dof="x", amp_counts=1.0)]
    xml_text = mag.build_sine_response_xml(base_cfg, tones, base_cfg["diag"]["measure"])
    p = _params(xml_text)
    chans = {p[f"MeasurementChannel[{i}]"] for i in range(3)}
    assert chans == {"Y1:DMD-PARTICLE_X_IN1", "Y1:DMD-PARTICLE_Y_IN1", "Y1:DMD-PARTICLE_Z_IN1"}


def test_multitone_one_electrode_repeats_channel(base_cfg):
    # one electrode at three tones -> channel repeated across three rows
    tones = [mag.Tone(freq=f, electrode="E1", dof="x", amp_counts=1.0)
             for f in (37.0, 39.0, 41.0)]
    xml_text = mag.build_sine_response_xml(base_cfg, tones, base_cfg["diag"]["measure"])
    p = _params(xml_text)
    chans = [p[f"StimulusChannel[{i}]"] for i in range(3)]
    assert chans == ["Y1:DMD-POLES_E1_EXC"] * 3
    assert {float(p[f"StimulusFrequency[{i}]"]) for i in range(3)} == {37.0, 39.0, 41.0}
