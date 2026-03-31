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
| `moku/sweep.py` | Moku waveform generator sweep control |
| `moku/pulse.py` | Moku pulse generator control |
| `teemController/` | Teem Photonics laser controller daemon + EPICS integration |

## Conventions

- Servo scripts support `--dry-run` for safe testing
- MEDM shell command buttons launch scripts in xterm windows
- Moku sweep logs are saved to `~/Dropbox/Microspheres/MAST-QG/worker1/data/YYMMDD/`
- Large log files (*.log) are gitignored
