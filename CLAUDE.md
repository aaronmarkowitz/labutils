<!-- lab-context cross-link -->
> **Context layer:** `~/Moore/lab-context/` (`experiment-overview.md`, `projects.yml`, `conventions.md`) — synced, no-Obsidian.
> **Obsidian note:** [[Lab-Automation-Architecture]] *(laptop only)* · **Registry:** `lab-context/projects.yml` → `labutils`
> **Branch-first** and the other global conventions: see `~/.claude/CLAUDE.md`. Not restated here.

# labutils

Utilities and scripts for the MAST-QG magnetically levitated microdiamond experiment.

## Which host am I on? (check first)

**Run `hostname` before anything else.** This repo is checked out on two machines and the
rules differ:

| `hostname` | What you can do |
|---|---|
| **`worker1`** (Debian lab workstation) | Hardware is reachable. **All safety rules below apply.** |
| anything else (Alma's macOS laptop) | **No hardware.** No EPICS, no `caget`/`caput`/`diag`, no camera or Moku scripts — they will fail or hang, not damage anything. Edit code, read, and run the pure-python tests only. |

`cymac1` is the rtcds/front-end server (FEC-11) — never a Claude host. Reach it **only**
from worker1 via `diag`/`caget`/`caput`, and **only when explicitly instructed**. The CDS
paths below (`/var/lib/cds-conda`, `/opt/rtcds`, `/home/controls`) exist on worker1 only —
a failed path check for them from the laptop is expected, not a bug.

## Safety — hardware (worker1 only)

- Servo/hardware scripts support **`--dry-run`; always prefer it** when testing.
- **Never command hardware unsupervised** without the viability preconditions
  (particle trapped / UHV / laser locked / in-range).
- Alma is the orchestrator for hardware work; Claude runs workers.
- **Never place an excitation tone in a band also used as a safety/trap-loss guard band**
  (e.g. 10–20 Hz for the particle) — the drive pollutes the guard and causes false aborts.

## Environment (canonical — `lab-context/conventions.md` points here)

Interpreters and envs on **worker1**:

| Interpreter / env | Contains | Use for |
|---|---|---|
| `/var/lib/cds-conda/base/bin/python3` | no `epics` module | EPICS work via `subprocess` + `caget`/`caput` |
| `/var/lib/cds-conda/base/envs/cds-testing/bin/python3` | `nds2`, `dttxml`, `numpy`, `scipy`, `h5py`, `yaml`, `pytest` | **the pytest interpreter**; diag/NDS2 analysis |
| conda env `mastqg` | `dttxml`, `pytest`, but **no** `nds2` | general analysis; cannot run this repo's tests (they import `nds2`) |
| conda env `thorcam` | Thorlabs + IDS SDKs | camera GUI (`cameras/run_thorcam.py`) |
| conda env `controls` | — | teem laser service |

- **Platform**: Linux Debian 11 (amd64). **EPICS tools** `caget`, `caput`, `medm` in PATH via cds-conda.
- **DAQ**: NDS2 server `192.168.1.11:8088`. **EPICS models**: `y1dmd` (`Y1:DMD`) + `auxioc` soft IOC (`Y1:AUX`).

## Instrument inventory

- **Cameras:** Thorlabs CS165MU + IDS DCC1545M (`cameras/run_thorcam.py`, `cameras/dualcam_fast.py`).
- **Moku:** waveform/pulse generator (`moku/sweep.py`, `moku/pulse.py`).
- **Vacuum:** Leybold turbo pump (`epics/run_leybold_turbolab.py`).
- **Laser:** Teem Photonics (`teemController/`, conda env `controls`).

## rtcds model: y1dmd

- **Prefix** `Y1:DMD`, FEC-11, 65536 Hz. **Source model** `/opt/rtcds/userapps/mastqg/y1dmd.mdl`
- **MEDM** `/opt/rtcds/yqg/y1/medm/y1dmd/` (auto-generated), `y1dmd_overview/`, `y1dmd_scripts/`; sitemap `/opt/rtcds/yqg/y1/medm/sitemap.adl`
- **BURT snapshots** `/opt/rtcds/yqg/y1/target/y1dmd/y1dmdepics/burt/`
- Filter-module switch states: read the `_SWSTR` channel for human-readable form
  (e.g. `caget Y1:DMD-LASER_CTLX_SWSTR`).

## auxioc: Y1:AUX soft IOC

`auxioc` systemd service hosts EPICS channels for ISS setpoints, UV laser control, and
vacuum pumps. It is a plain softIoc (not an rtcds model), so it has **no built-in
SDF/BURT restore** — settings persist via BURT-compatible snapshots in `epics/`:
`autoBurt.req` (channel list: ISS cal factors, mW setpoints), `auxioc.snap` (latest,
gitignored, auto-restored on restart), `auxioc_YYMMDD_HHMMSS.snap` (timestamped archives,
gitignored), `save_aux_settings.sh` (save → `auxioc.snap`), `restore_aux_settings.sh`
(restore, called by systemd `ExecStartPost`).

**After any ISS calibration change**, run `epics/save_aux_settings.sh` (on worker1:
`/home/controls/labutils/epics/save_aux_settings.sh`). Channels intentionally excluded
from the snapshot: `UVD_LASER_ON`, `UVD_TURN_OFF` — must default to safe/off on restart.

## Scripts in this repo

All paths verified 2026-09-10.

| Script | Purpose |
|---|---|
| `scripts/map_y1dmd_state.py` | Read all EPICS channels; display Y1DMD control-system state |
| `scripts/engage_intensity_servo_x.sh` / `disengage_intensity_servo_x.sh` | Activate / deactivate X laser intensity stabilization servo (`_z.sh` variants also exist) |
| `scripts/particle_lo_scan.py` | Particle LO scan |
| `mdl_to_adl.py` | Generate MEDM `.adl` overview screens from Simulink `.mdl` models |
| `cameras/run_thorcam.py` | Dual-camera GUI. At record-stop writes a `<video>.json` sidecar (`cameras/video_metadata.py::write_sidecar`, best-effort) with the TRUE `measured_fps` = frames/duration — the mp4 container fps is only a nominal acquisition setting, so analysis prefers the sidecar for length calibration. |
| `cameras/dualcam_fast.py` | High-rate dual-camera GUI. FPS control commands a **fixed hardware frame rate** (IDS `is_SetFrameRate`, Thorlabs `frame_rate_control_value`), clamps exposure to the frame period, re-applies after every ROI change; display refresh decoupled at `DISPLAY_FPS_CAP` (30), IDS ring depth `IDS_RING_BUFFERS` (12). Writes `_timestamps.npy` (software), `_hwclock.npz` (hardware frame counter + device timestamp), and the schema-v2 `.json` sidecar. **Software dequeue jitter is OS scheduling noise — analysis should prefer the hardware clock.** |
| `cameras/video_metadata.py` | JSON video-sidecar writer (stdlib-only): measured_fps, n_frames, duration, ROI |
| `epics/run_leybold_turbolab.py` | Leybold turbo pump monitoring via EPICS |
| `fetch_nds2_data.py` | NDS2 data fetching (server `192.168.1.11:8088`) |
| `scripts/dipole/upload_sense_matrix.py` | Upload a SENSE matrix to EPICS from pipeline step_01 HDF5. Also force-enables the input filter modules feeding each written column — a zeroed input silently deletes a W column, collapsing PARTICLE_X/Y onto one mode. |
| `scripts/dipole/verify_particle_equipartition.py` | Validate a deployed SENSE matrix on recorded PARTICLE_X/Y/Z |
| `scripts/dipole/measure_actuator_gain.py` | Measure electrode→particle actuator gains via a `diag` SineResponse (see `scripts/dipole/README.md`) |
| `scripts/dipole/dipole_sideband_model.py` | Turns the naive rotating-field E-field ellipse into the dipole **libration sideband** (ωφ=√(d·\|E₊\|/I), co-rotating projection — NOT ⟨\|E\|⟩) and the headline d-inference error `d_inferred/d_true = \|E₊\|_true/E_believed` |
| `scripts/dipole/plot_actuator_gain.py`, `plot_naive_vs_measured_acts.py` | Plotting for the above |
| `moku/sweep.py`, `moku/pulse.py` | Moku waveform-generator sweep / pulse control. `conf_sweep.yaml`'s `dc offset` (volts, fixed across the ramp) reaches `generate_waveform(offset=)`; amplitude is **peak-to-peak**, so the excursion is `\|offset\| + amp/2` against a ±5 V Moku:Go rail — `moku/sweep_limits.py` (`--self-test`, no hardware, no env) names the first pulse that would exceed it before the run is confirmed. |
| `teemController/` | Teem Photonics laser controller daemon + EPICS integration |

**Equipartition / common-unit anchor semantics** (what `common_unit_anchor`,
`relT`, and the deprecated `--disagreement` mean, and why `equipartition` and not
`white_force`) are defined in **one** place — the analysis repo's skill at
`analysis/.claude/skills/verify-equipartition/SKILL.md`. Read it there; do not
re-derive or restate the rule here.

## diag / DTT transfer-function measurements

Detailed procedure lives in `scripts/dipole/README.md`. Reusable facts:

- **Fast vs slow channels**: filter-module slow records (`_GAIN`, `_OFFSET`, `_TRAMP`,
  `_SW1R`, `_SW2R`) answer `caget`. Fast channels (`_EXC`, `_IN1`, `_OUT16`) are **test
  points** served by the front end — reach them via `diag`/`awg`/`nds2`, **not** `caget`.
- **`diag` = headless diaggui.** `diag -l -f <cmdfile>` (`-l` local kernel, `-f` command
  script); `diag -l` from worker1 reaches cymac1's awg/tp/nds. Verified post-connect verbs:
  `restore <xml>` / `run -w` / `save <result.xml>` / `quit` — paths **unquoted**.
- **SineResponse XML**: index-aligned `Stimulus{Frequency,Amplitude,Offset,Phase}[i]`
  (phase in radians) + `StimulusChannel[i]`; multi-tone on one channel = repeat the channel
  across rows (#rows == #tones). `MeasurementChannel[i]` are readbacks. Must include
  `FFTResult` or `run` aborts. `MeasurementTime` Dim2 = `[min_time_seconds, n_cycles]`
  (runs for max of the two). `SettlingTime` is a RELATIVE fraction of measurement time.
- **Multi-tone coefficient fix — two approaches** (dtt 4.1.x, both in `measure_actuator_gain.py`):
  (a) *ACTS distinct-channel* (preferred): give each tone its own `ACTS_{row}_{col}_EXC`
  channel (`assign_acts_channels`) so `sizeA == sizeExc` and diag's normalization indexes
  correctly; set `StimulusReadback[i]` to the EXC channel. Requires GAIN=1, input switch
  OFF, output switch ON per ACTS element (snapshot/restore on exit). **Pending hardware
  validation on ACTS_8_1..8_4 loopback.** (b) *NDS2 workaround* (always valid, and the
  **primary analysis path**): inject via diag, capture raw channels over NDS2, compute
  `TF = S_ER/S_EE` via `scipy.signal.csd` — validated on ACTS_8_8→LOS_IN1 loopback, 0.500
  flat at every tone.
- **`dttxml`** reads diag result XML (`DiagAccess(xml).sine_response(...)`), but don't
  trust its multi-tone coefficients unless each stimulus has a distinct channel.
- **DAC**: 16-bit signed, ±32768 counts; practical safe clip ~32000. Loopback DAC↔ADC
  full-scale differ by 2× → loopback gain 0.5.
- A real `Y1:DMD` diag measurement inherently runs against **cymac1** — a data-path fact.

## Tests

`pytest` from the repo root discovers the dipole safety tests (`pytest.ini` sets the
paths and `pythonpath`). On worker1 use the cds-testing interpreter:
`/var/lib/cds-conda/base/envs/cds-testing/bin/python3 -m pytest -q`. The `control`
marker (`-m control`) holds tests that prove a safety check *fails* on bad input.
Camera tests under `cameras/` need the `thorcam` env and are not collected by default.

## Conventions

- MEDM shell-command buttons launch scripts in xterm windows.
- Moku sweep logs → `~/Dropbox/Microspheres/MAST-QG/worker1/data/YYMMDD/`; `*.log` gitignored.
- Session context lives in `HANDOFF.md` at the repo root (`SESSION_LOG.md` is retired).
