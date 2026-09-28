"""dropper_server.py against the fake Moku: stop latency and the deadman paths."""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

MOKU_DIR = Path(__file__).resolve().parents[1]

CONF = """amp: 0.5
amplitude increment: 1.1
base frequency: 3e3
channel: 1
dc offset: 0
end amp: 2
ip_address: "192.0.2.1"
no_pulses: 10
stop frequency: 3e4
sweep duration: 4
"""


def _spawn(tmp_path, *extra):
    pytest.importorskip("yaml")
    conf = tmp_path / "conf_sweep.yaml"
    conf.write_text(CONF)
    return subprocess.Popen(
        [sys.executable, str(MOKU_DIR / "dropper_server.py"), "--conf", str(conf),
         "--fake", "--no-conf-copy", "--fake-latency-s", "0.02", *extra],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)


def _events(p, until, timeout=10.0):
    evs = []
    t_end = time.time() + timeout
    while time.time() < t_end:
        line = p.stdout.readline()
        if not line:
            break
        ev = json.loads(line)
        evs.append(ev)
        if ev["ev"] == until:
            return evs
    raise AssertionError(f"no {until!r} event; got {evs}")


def _send(p, cmd):
    p.stdin.write(json.dumps({"cmd": cmd}) + "\n")
    p.stdin.flush()


def test_nothing_emitted_before_start_and_stop_cancels_mid_pulse(tmp_path):
    p = _spawn(tmp_path)
    ready = _events(p, "ready")[-1]
    assert ready["n_emitted"] > 1
    _send(p, "start")
    _events(p, "pulse")
    time.sleep(0.3)              # mid-way through a 4 s pulse
    _send(p, "stop")
    st = _events(p, "stopped")[-1]
    p.wait(5)
    assert st["reason"] == "stop"
    assert st["latency_s"] < 0.5          # not the remaining ~3.7 s of the pulse
    calls = st["calls"]
    i_off = max(i for i, c in enumerate(calls) if c == "generate_waveform")
    assert calls[i_off + 1:] == ["set_defaults", "relinquish_ownership"]
    assert calls.count("set_sweep_mode") == 1   # only one pulse went out


def test_deadman_eof_turns_output_off(tmp_path):
    p = _spawn(tmp_path)
    _events(p, "ready")
    _send(p, "start")
    _events(p, "pulse")
    p.stdin.close()              # loader died
    st = _events(p, "stopped")[-1]
    assert st["reason"] == "eof"
    assert st["calls"][-2:] == ["set_defaults", "relinquish_ownership"]


def test_deadman_heartbeat(tmp_path):
    p = _spawn(tmp_path, "--heartbeat-s", "0.6")
    _events(p, "ready")
    _send(p, "start")
    _events(p, "pulse")
    st = _events(p, "stopped", timeout=5)[-1]   # no pings sent
    assert st["reason"] == "heartbeat"
    p.stdin.close()


@pytest.mark.control
def test_control_pings_keep_it_running(tmp_path):
    """Pinging must prevent the heartbeat stop -- otherwise the test above would
    pass for any server that simply stops early."""
    p = _spawn(tmp_path, "--heartbeat-s", "0.6")
    _events(p, "ready")
    _send(p, "start")
    _events(p, "pulse")
    for _ in range(8):
        time.sleep(0.2)
        _send(p, "ping")
    _send(p, "stop")
    st = _events(p, "stopped")[-1]
    assert st["reason"] == "stop"


@pytest.mark.control
def test_control_refuses_ramp_beyond_rail(tmp_path):
    pytest.importorskip("yaml")
    conf = tmp_path / "conf_sweep.yaml"
    conf.write_text(CONF.replace("dc offset: 0", "dc offset: 4.5"))
    p = subprocess.run([sys.executable, str(MOKU_DIR / "dropper_server.py"), "--conf", str(conf),
                        "--fake"], input="", capture_output=True, text=True, timeout=10)
    evs = [json.loads(l) for l in p.stdout.splitlines()]
    assert p.returncode == 2
    assert evs[-1]["reason"] == "refused"
    assert not any(e["ev"] == "ready" for e in evs)
