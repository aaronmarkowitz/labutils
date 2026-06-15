# labutils

Utilities and scripts for the MAST-QG magnetically levitated microdiamond experiment.

## Environment

- **Platform**: Linux Debian 11 (amd64), workstation `cymac1`
- **EPICS tools**: `caget`, `caput`, `medm` in PATH (via cds-conda)
- **Python**: System python3 at `/var/lib/cds-conda/base/bin/python3` (no `epics` module; use `subprocess` + `caget`/`caput`)
- **Conda env `thorcam`**: For camera GUI (`run_thorcam.py`)
- **Conda env `controls`**: For teem laser service

## rtcds model: y1dmd

- **Prefix**: `Y1:DMD`, FEC-11, 65536 Hz
- **Source model**: `/opt/rtcds/userapps/mastqg/y1dmd.mdl`
- **MEDM screens**: `/opt/rtcds/yqg/y1/medm/y1dmd/` (auto-generated), `y1dmd_overview/`, `y1dmd_scripts/`
- **Sitemap**: `/opt/rtcds/yqg/y1/medm/sitemap.adl`
- **BURT snapshots**: `/opt/rtcds/yqg/y1/target/y1dmd/y1dmdepics/burt/`
- **Filter module switch states**: Use `_SWSTR` channel for human-readable format (e.g., `caget Y1:DMD-LASER_CTLX_SWSTR`)

## auxioc: Y1:AUX soft IOC

The `auxioc` systemd service hosts EPICS channels for ISS setpoints, UV laser control, and vacuum pumps.
Because it uses a plain softIoc (not an rtcds model), it has no built-in SDF/BURT restore.
Settings are persisted via BURT-compatible snapshots in `epics/`:

| File | Purpose |
|---|---|
| `epics/autoBurt.req` | Channel list to snapshot (ISS cal factors and mW setpoints) |
| `epics/auxioc.snap` | Latest snapshot (gitignored, auto-restored on restart) |
| `epics/auxioc_YYMMDD_HHMMSS.snap` | Timestamped archive copies (gitignored) |
| `epics/save_aux_settings.sh` | Save current values → `auxioc.snap` |
| `epics/restore_aux_settings.sh` | Restore from `auxioc.snap` (called by systemd `ExecStartPost`) |

**After any ISS calibration change**, run:
```bash
/home/controls/labutils/epics/save_aux_settings.sh
```

Channels intentionally excluded from snapshot: `UVD_LASER_ON`, `UVD_TURN_OFF` (must default to safe/off on restart).

## Scripts in this repo

| Script | Purpose |
|---|---|
| `map_y1dmd_state.py` | Read all EPICS channels and display Y1DMD control system state |
| `engage_intensity_servo_x.sh` | Safely activate X laser intensity stabilization servo |
| `disengage_intensity_servo_x.sh` | Safely deactivate X laser intensity servo |
| `mdl_to_adl.py` | Generate MEDM .adl overview screens from Simulink .mdl models |
| `run_thorcam.py` | Dual-camera GUI (Thorlabs + IDS) |
| `run_leybold_turbolab.py` | Leybold turbo pump monitoring with EPICS |
| `fetch_nds2_data.py` | NDS2 data fetching (server 192.168.1.11:8088) |
| `scripts/dipole/upload_sense_matrix.py` | Upload a sensor (SENSE) matrix to EPICS from pipeline step_01 HDF5 |
| `scripts/dipole/measure_actuator_gain.py` | Measure electrode→particle actuator gains via a `diag` SineResponse (see `scripts/dipole/README.md`) |
| `moku/sweep.py` | Moku waveform generator sweep control |
| `moku/pulse.py` | Moku pulse generator control |
| `teemController/` | Teem Photonics laser controller daemon + EPICS integration |

## diag / DTT transfer-function measurements (general)

For driving sine/transfer-function measurements on the front end (e.g.
`scripts/dipole/measure_actuator_gain.py`), see the detailed notes in
`scripts/dipole/README.md`. Key reusable facts:

- **Fast vs slow channels**: filter-module slow records (`_GAIN`, `_OFFSET`,
  `_TRAMP`, `_SW1R`, `_SW2R`) answer `caget`. Fast channels (`_EXC`, `_IN1`,
  `_OUT16`) are **test points** served by the front end — reach them via
  `diag`/`awg`/`nds2`, **not** `caget`/`caput`.
- **`diag` = headless diaggui.** Run a measurement template with
  `diag -l -f <cmdfile>` (`-l` local kernel, `-f` command script). `diag -l` from
  worker1 reaches cymac1's awg/tp/nds (per `diag -i`). VERIFIED post-connect verbs:
  `restore <xml>` / `run -w` (run+wait) / `save <result.xml>` / `quit` — paths
  **unquoted** (diag's help prints `'filename'` as placeholder notation, not literal).
- **SineResponse XML**: index-aligned `Stimulus{Frequency,Amplitude,Offset,Phase}[i]`
  (phase radians) + `StimulusChannel[i]`; multi-tone on one channel = repeat the
  channel across rows (#rows == #tones). `MeasurementChannel[i]` are the readbacks.
  Required gotchas: must include `FFTResult` (boolean) or `run` aborts; `MeasurementTime`
  Dim2 = `[min_time_seconds, n_cycles]` (runs for max(min_time, cycles/freq) — seconds
  floor low, cycles floor high); `SettlingTime` is a RELATIVE fraction of meas time.
- **Multi-tone coefficient fix — two approaches** (dtt 4.1.x, both implemented):
  1. **ACTS distinct-channel approach** (preferred): assign each tone to its own
     `ACTS_{row}_{col}_EXC` channel (`assign_acts_channels`). With one distinct
     channel per stimulus row, `sizeA == sizeExc` and diag's normalization loop
     indexes correctly. Also set `StimulusReadback[i]` to the EXC channel so diag
     uses the measured excitation rather than the commanded value. Requires GAIN=1,
     input switch OFF, output switch ON on each ACTS element (EXC goes through GAIN;
     snapshot/restore on exit).
  2. **NDS2 workaround** (always valid): inject via diag, capture the raw channels
     over NDS2 during the injection, compute `TF = S_ER/S_EE` via
     `scipy.signal.csd`. Validated on ACTS_8_8→LOS_IN1 loopback: 0.500 flat at
     every tone. Does NOT rely on diag's coefficient output at all.
  Both paths are implemented in `measure_actuator_gain.py`; NDS2 is the primary
  analysis path (robust to any diag version). ACTS distinct-channel approach pending
  hardware validation on ACTS_8_1..8_4 loopback.
- **`dttxml`** still reads diag result XML (`DiagAccess(xml).sine_response(...)`), but
  don't trust its multi-tone coefficients unless each stimulus has a distinct channel.
- **Interpreter**: `/var/lib/cds-conda/base/envs/cds-testing/bin/python3` has
  `nds2`, `dttxml`, `numpy`, `scipy`, `h5py`, `yaml`, `pytest` (the `mastqg` env has
  `dttxml` but not `nds2`/`pytest`).
- **dtt version**: installed `4.1.5~rc1` is from the `bullseye-unstable` apt channel;
  stable (`bullseye`) is `4.1.4`. Three CDS channels are enabled. The sine-response
  behavior above is the same across these (sineresponse.cc unchanged since Jan 2024).
- **DAC**: 16-bit signed, ±32768 counts; practical safe clip ~32000. (Loopback DAC↔ADC
  full-scale differ by 2× → loopback gain 0.5.)
- **Pitfall**: never place an excitation tone in a band you are also using as a
  safety/trap-loss guard band (e.g. 10–20 Hz for the particle) — the drive pollutes
  the guard and causes false aborts.
- A real Y1:DMD `diag` measurement inherently runs against **cymac1** (FEC-11), the
  exception to the usual "default to worker1" rule.

## Conventions

- Servo scripts support `--dry-run` for safe testing
- MEDM shell command buttons launch scripts in xterm windows
- Moku sweep logs are saved to `~/Dropbox/Microspheres/MAST-QG/worker1/data/YYMMDD/`
- Large log files (*.log) are gitignored
