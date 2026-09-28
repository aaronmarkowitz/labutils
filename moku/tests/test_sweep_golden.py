"""sweep.py refactor must not change what reaches the instrument.

Runs the pre-refactor sweep.py (from git, commit ca43d95) and the current one as
subprocesses against a stub `moku.instruments` backed by FakeWaveformGenerator,
feeding 'y' on stdin, with time.sleep patched out and HOME redirected so the conf
copy lands in a temp dir. The call sequences must match, except for the one
intended change: the current CLI switches the output Off before set_defaults().
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

MOKU_DIR = Path(__file__).resolve().parents[1]
REPO = MOKU_DIR.parent
OLD_REV = "ca43d95"

CONF = """amp: 0.5
amplitude increment: 1.5
base frequency: 3e3
channel: 1
dc offset: 0.25
end amp: 2
ip_address: "192.0.2.1"
no_pulses: 10
stop frequency: 3e4
sweep duration: 2
"""

STUB_INSTRUMENTS = """
import atexit, json, os, sys, time
sys.path.insert(0, {moku_dir!r})
from fake_moku import FakeWaveformGenerator
time.sleep = lambda s: None
_made = []
class WaveformGenerator(FakeWaveformGenerator):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        _made.append(self)
@atexit.register
def _dump():
    with open(os.environ['FAKE_MOKU_LOG'], 'w') as f:
        json.dump([[n, kw] for g in _made for (_, n, kw) in g.calls], f)
"""


def _run(script: Path, tmp: Path, tag: str):
    stub = tmp / f"stub_{tag}" / "moku"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text("")
    (stub / "instruments.py").write_text(STUB_INSTRUMENTS.format(moku_dir=str(MOKU_DIR)))
    work = tmp / f"work_{tag}"
    work.mkdir()
    (work / "conf_sweep.yaml").write_text(CONF)
    log = tmp / f"calls_{tag}.json"
    env = dict(os.environ, HOME=str(tmp / f"home_{tag}"), FAKE_MOKU_LOG=str(log),
               PYTHONPATH=os.pathsep.join([str(stub.parent), str(MOKU_DIR)]))
    p = subprocess.run([sys.executable, str(script)], cwd=work, input="y\n",
                       env=env, capture_output=True, text=True, timeout=60)
    assert "Program terminated." in p.stderr + p.stdout, p.stdout + p.stderr
    saved = list((tmp / f"home_{tag}").rglob("*_conf_sweep.yaml"))
    return json.loads(log.read_text()), p.stdout, saved


def _old_script(tmp: Path) -> Path:
    try:
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{OLD_REV}:moku/sweep.py"],
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("pre-refactor sweep.py not available from git")
    d = tmp / "old"
    d.mkdir()
    (d / "sweep.py").write_text(src)
    return d / "sweep.py"


OFF = ["generate_waveform", {"channel": 1, "type": "Off", "amplitude": 1,
                             "frequency": 10000, "offset": 0}]


def sequences_match(old_calls, new_calls):
    """True iff new == old plus exactly one Off call immediately before set_defaults."""
    if new_calls.count(OFF) != 1:
        return False
    i = new_calls.index(OFF)
    if i + 1 >= len(new_calls) or new_calls[i + 1][0] != "set_defaults":
        return False
    return new_calls[:i] + new_calls[i + 1:] == old_calls


def test_call_sequence_unchanged_except_output_off(tmp_path):
    pytest.importorskip("yaml")
    old_calls, old_out, old_saved = _run(_old_script(tmp_path), tmp_path, "old")
    new_calls, new_out, new_saved = _run(MOKU_DIR / "sweep.py", tmp_path, "new")

    assert sequences_match(old_calls, new_calls)
    # Same pulse count and the same printed progress lines.
    grep = lambda s: [l for l in s.splitlines() if l.startswith(("Running", "Calculated", "Total"))]
    assert grep(new_out) == grep(old_out)
    assert len(old_saved) == len(new_saved) == 1
    import yaml
    assert yaml.safe_load(new_saved[0].read_text()) == yaml.safe_load(old_saved[0].read_text())


@pytest.mark.control
def test_control_golden_detects_a_changed_ramp(tmp_path):
    """sequences_match must reject a ramp missing one pulse, a wrong amplitude,
    and a missing Off -- otherwise the golden test above proves nothing."""
    pytest.importorskip("yaml")
    old_calls, _, _ = _run(_old_script(tmp_path), tmp_path, "old")
    k = next(j for j, c in enumerate(old_calls) if c[0] == "set_defaults")
    good = old_calls[:k] + [OFF] + old_calls[k:]
    assert sequences_match(old_calls, good)
    fewer = good[:1] + good[3:]
    assert not sequences_match(old_calls, fewer)
    wrong_amp = json.loads(json.dumps(good))
    j = [n for n, c in enumerate(wrong_amp) if c[1].get("type") == "Sine"][1]
    wrong_amp[j][1]["amplitude"] *= 1.001
    assert not sequences_match(old_calls, wrong_amp)
    assert not sequences_match(old_calls, old_calls)
