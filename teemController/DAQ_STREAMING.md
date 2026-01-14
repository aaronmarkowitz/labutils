# Teem Laser DAQ Streaming Configuration

## Overview

The Teem Photonics laser controller EPICS channels are now streaming to the DAQ on cymac1 at 16 Hz via the standalone_edc (EPICS data concentrator).

## Channels Being Streamed

### Status Channels (integer)
- `Y1:AUX-UVD_LASER_STATE` - Laser state (0=OFF, 1=STARTING, 2=ON, 3=STOPPING, 4=ERROR)
- `Y1:AUX-UVD_EMITTING` - Currently emitting (0=OFF, 1=ON)
- `Y1:AUX-UVD_READY` - Ready for emission (0/1)
- `Y1:AUX-UVD_TEMP_OK` - Temperature regulation OK (0/1)

### Temperature Channels (float, °C)
- `Y1:AUX-UVD_DIODE_TEMP` - Diode temperature
- `Y1:AUX-UVD_CRYSTAL_TEMP` - Crystal temperature
- `Y1:AUX-UVD_HEATSINK_TEMP` - Electronics heatsink temperature
- `Y1:AUX-UVD_LASER_HEATSINK_TEMP` - Laser heatsink temperature

### Error/Info Registers (integer, hex values)
- `Y1:AUX-UVD_EREG1/2/3` - Error registers
- `Y1:AUX-UVD_IREG1/2/3` - Info registers

### Diagnostic Channels (float, seconds)
- `Y1:AUX-UVD_HEARTBEAT_TIMEOUT` - Deadman switch timeout setting
- `Y1:AUX-UVD_LAST_HEARTBEAT` - Last heartbeat timestamp
- `Y1:AUX-UVD_UPTIME` - Service uptime

## Configuration Files

### cymac1:/etc/advligorts/edc.ini
Contains channel definitions for standalone_edc. Added 18 Teem laser channels with appropriate datatypes:
- `datatype=4` for integers (state, registers, binary)
- `datatype=5` for floats (temperatures, timestamps)

### cymac1:/etc/advligorts/systemd_env
Already configured with:
- `standalone_edc_args='--sync-to=y1iop_daq -p "Y1:AUX-"'`
- `local_dc_args` includes `"edc:52:16"` (dcuid=52, rate=16Hz)

### cymac1:/etc/advligorts/master
Already includes `/etc/advligorts/edc.ini` in the list for daqd.

## Services Restarted

All DAQ services were restarted to pick up the new channels:
1. `rts-edc.service` - standalone_edc
2. `rts-local_dc.service` - local data concentrator
3. `rts-daqd.service` - data acquisition daemon
4. `rts-nds.service` - network data server

## Testing

Verified channels are accessible from cymac1:
```bash
caget Y1:AUX-UVD_LASER_STATE Y1:AUX-UVD_DIODE_TEMP
```

## Data Access

Channels can be accessed via:
- **EPICS**: `caget Y1:AUX-UVD_*` from cymac1
- **NDS2**: Available for historical data retrieval
- **Dataviewer**: Can plot trends and minute trends

## Configuration Date

- Added to DAQ: 2026-01-14
- Backup created: `/etc/advligorts/edc.ini.backup.20260114_*`

## References

- Teem controller setup: [teemController/README.md](README.md)
- EPICS database: [../teem_laser.db](../teem_laser.db)
- EDC configuration guide: https://git.ligo.org/cds/software/advligorts/-/wikis/sysadmin/Edcu
