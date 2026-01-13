# Teem Photonics Laser Controller Integration

EPICS IOC integration for Teem Photonics MLC-03A-DR1 laser controller with deadman switch safety mechanism.

## Overview

This system provides:
- EPICS soft IOC channels for laser control and monitoring
- Deadman switch safety mechanism requiring continuous heartbeat
- Automatic shutdown on critical errors
- Temperature and error register monitoring
- Control script for timed laser operation

## Architecture

```
Hardware (Serial) → run_teem_laser.py → EPICS PVs ← User Scripts
                                             ↑
                                      teem_laser_control.py
```

- **teem_laser.db**: EPICS database with Y1:AUX-UVD_* channels
- **run_teem_laser.py**: Service daemon that polls controller and updates PVs
- **teem_laser_control.py**: User-friendly control script
- **teem-laser.service**: Systemd service configuration

## Installation

### 1. Identify Serial Device

Plug in the USB-serial adapter and find the device:

```bash
# Before plugging in
ls /dev/tty* > /tmp/before.txt

# After plugging in
ls /dev/tty* > /tmp/after.txt
diff /tmp/before.txt /tmp/after.txt

# Or check dmesg
dmesg | tail -20 | grep tty
```

Common devices: `/dev/ttyUSB0`, `/dev/ttyACM0`

### 2. Set Serial Permissions

Add user to dialout group:

```bash
sudo usermod -a -G dialout controls
# Log out and log back in for changes to take effect
```

Or create udev rule for persistent symlink (recommended):

```bash
# Find USB vendor/product ID
lsusb
# Look for your USB-serial adapter, note idVendor and idProduct

# Create udev rule
sudo nano /etc/udev/rules.d/99-teem-laser.rules

# Add (replace XXXX and YYYY with your values):
SUBSYSTEM=="tty", ATTRS{idVendor}=="XXXX", ATTRS{idProduct}=="YYYY", SYMLINK+="teem_laser", MODE="0666", GROUP="dialout"

# Reload rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 3. Test Serial Communication

```bash
# Install minicom if needed
sudo apt-get install minicom

# Test connection (19200 8N1)
minicom -D /dev/ttyUSB0 -b 19200

# In minicom, type: GSER<Enter>
# Should see response like: GSER_00_00_00_00_00_00>
# Press Ctrl+A then X to exit minicom
```

### 4. Update Service Configuration

Edit the service file if your device is not `/dev/ttyUSB0`:

```bash
nano /home/controls/labutils/teemController/teem-laser.service

# Change this line:
# ExecStart=... --device /dev/ttyUSB0 ...
# To your device, e.g.:
# ExecStart=... --device /dev/teem_laser ...
```

### 5. Install Service

```bash
# Copy service file
sudo cp /home/controls/labutils/teemController/teem-laser.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable teem-laser.service
```

### 6. Restart IOC to Load Database

```bash
# Restart auxioc to load teem_laser.db
sudo systemctl restart auxioc.service

# Verify PVs are loaded
caget Y1:AUX-UVD_LASER_STATE
caget Y1:AUX-UVD_SERIAL_NUMBER
```

### 7. Start Service

```bash
# Start teem-laser service
sudo systemctl start teem-laser.service

# Check status
systemctl status teem-laser.service

# View logs
journalctl -u teem-laser.service -f
```

## Usage

### Control Script

The `teem_laser_control.py` script provides user-friendly laser control:

```bash
# Check status
./teem_laser_control.py status

# Turn on for 10 seconds
./teem_laser_control.py on 10

# Turn on continuously (Ctrl+C to stop)
./teem_laser_control.py on -1

# Turn off immediately
./teem_laser_control.py off
```

### Direct EPICS Channel Control

```bash
# Turn on laser (requires heartbeat to keep alive)
caput Y1:AUX-UVD_LASER_ON 1

# Send heartbeat (must do this faster than timeout)
while true; do
    caput Y1:AUX-UVD_TURN_OFF 0
    sleep 0.5
done

# Turn off
caput Y1:AUX-UVD_LASER_ON 0

# Emergency stop
caput Y1:AUX-UVD_EMERGENCY_STOP 1
```

### Configuration

```bash
# View/change heartbeat timeout (default: 2.0 seconds)
caget Y1:AUX-UVD_HEARTBEAT_TIMEOUT
caput Y1:AUX-UVD_HEARTBEAT_TIMEOUT 5.0  # Change to 5 seconds
```

## EPICS Channels

### Control Channels

- `Y1:AUX-UVD_LASER_ON` (bo) - Master on/off command
- `Y1:AUX-UVD_TURN_OFF` (bo) - Deadman switch (auto-resets to 1)
- `Y1:AUX-UVD_HEARTBEAT_TIMEOUT` (ao) - Timeout in seconds (default: 2.0)
- `Y1:AUX-UVD_EMERGENCY_STOP` (bo) - Immediate emergency stop

### Status Channels

- `Y1:AUX-UVD_LASER_STATE` (mbbi) - 0=OFF, 1=STARTING, 2=ON, 3=STOPPING, 4=ERROR
- `Y1:AUX-UVD_READY` (bi) - Laser ready for emission
- `Y1:AUX-UVD_EMITTING` (bi) - Laser currently emitting
- `Y1:AUX-UVD_TEMP_OK` (bi) - Temperature regulation OK

### Temperature Monitoring

- `Y1:AUX-UVD_DIODE_TEMP` (ai) - Diode temperature (°C)
- `Y1:AUX-UVD_CRYSTAL_TEMP` (ai) - Crystal temperature (°C)
- `Y1:AUX-UVD_HEATSINK_TEMP` (ai) - Electronics heatsink temp (°C)
- `Y1:AUX-UVD_LASER_HEATSINK_TEMP` (ai) - Laser heatsink temp (°C)

### Error Monitoring

- `Y1:AUX-UVD_EREG1/2/3` (longin) - Error registers (hex)
- `Y1:AUX-UVD_IREG1/2/3` (longin) - Info registers (hex)
- `Y1:AUX-UVD_ERR_*` (bi) - Individual error bits (24 total)

### Runtime Tracking

- `Y1:AUX-UVD_EMISSION_HOURS` (longin) - Total emission hours
- `Y1:AUX-UVD_EMISSION_MINUTES` (longin) - Total emission minutes
- `Y1:AUX-UVD_DIODE_HOURS` (longin) - Diode supply hours
- `Y1:AUX-UVD_DIODE_MINUTES` (longin) - Diode supply minutes

### System Info

- `Y1:AUX-UVD_SERIAL_NUMBER` (stringin) - Laser serial number
- `Y1:AUX-UVD_FW_HEAD` (stringin) - Head firmware version
- `Y1:AUX-UVD_FW_CONTROLLER` (stringin) - Controller firmware version

## Safety Features

### Deadman Switch

The deadman switch prevents the laser from running unattended:

1. User must continuously write `0` to `Y1:AUX-UVD_TURN_OFF` channel
2. If no heartbeat is received within the timeout period, laser automatically stops
3. Timeout is configurable via `Y1:AUX-UVD_HEARTBEAT_TIMEOUT` (default: 2.0s)
4. Service auto-resets the channel to `1` after every read

**Example heartbeat loop**:
```python
import time
from epics import caput

timeout = 2.0  # seconds
heartbeat_rate = 4  # Hz (must be > 1/timeout)

while laser_should_be_on:
    caput('Y1:AUX-UVD_TURN_OFF', 0, wait=False)
    time.sleep(1.0 / heartbeat_rate)
```

### Critical Error Handling

The service monitors 24 error conditions and triggers immediate shutdown on:

- E1: Heatsink overtemp
- E3: Interlock open
- E4: Laser head overtemp
- E5/E6: Diode temperature out of range
- E7/E8: Crystal temperature out of range
- E11: Diode temperature boundary
- E13-E16: TEC open/short circuits
- E17/E18: Diode open/short circuit
- E24: Crystal temperature boundary

### Interlock

Per manual requirements:
- Interlock immediately stops laser emission
- Laser will NOT auto-restart after interlock reset
- User must manually restart via front panel or command

## Troubleshooting

### Service Won't Start

```bash
# Check service status
systemctl status teem-laser.service

# View detailed logs
journalctl -u teem-laser.service -n 100

# Common issues:
# 1. Serial device permission denied
ls -la /dev/ttyUSB0
sudo usermod -a -G dialout controls  # Then log out/in

# 2. Serial device not found
ls /dev/tty*  # Find correct device
# Update service file with correct device path

# 3. IOC not running
systemctl status auxioc.service
sudo systemctl start auxioc.service
```

### Cannot Connect to PVs

```bash
# Check if IOC is running
systemctl status auxioc.service

# Check if database is loaded
caget Y1:AUX-UVD_LASER_STATE

# Restart IOC
sudo systemctl restart auxioc.service
```

### Laser Won't Turn On

```bash
# Check status
./teem_laser_control.py status

# Check error registers
caget Y1:AUX-UVD_EREG1
caget Y1:AUX-UVD_EREG2
caget Y1:AUX-UVD_EREG3

# Check temperatures
caget Y1:AUX-UVD_DIODE_TEMP
caget Y1:AUX-UVD_CRYSTAL_TEMP

# Check interlock
caget Y1:AUX-UVD_ERR_INTERLOCK

# View service logs
journalctl -u teem-laser.service -f
```

### Deadman Switch Keeps Stopping Laser

```bash
# Increase timeout
caput Y1:AUX-UVD_HEARTBEAT_TIMEOUT 5.0

# Check heartbeat rate in your script
# Must send heartbeat FASTER than 1/timeout
# e.g., for 2s timeout, send at least every 1s (prefer every 0.5s)
```

## Testing

### Manual Command Testing

Test serial commands directly:

```bash
# From Python
python3 << 'EOF'
from teemController.run_teem_laser import TeemController
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
logger.addHandler(handler)

controller = TeemController('/dev/ttyUSB0', logger)

# Get status
status = controller.get_status_registers()
print(f"Status: {status}")

# Get temperatures
temps = controller.get_temperatures()
print(f"Temps: {temps}")

# Get serial number
sn = controller.get_serial_number()
print(f"Serial: {sn}")

controller.close()
EOF
```

### IOC Database Test

```bash
# Load database in test IOC
softIoc -d /home/controls/labutils/teem_laser.db \
        -S -P Y1: -R AUX- &

# Test PV access
caget Y1:AUX-UVD_LASER_STATE
caput Y1:AUX-UVD_HEARTBEAT_TIMEOUT 3.0

# Kill test IOC
pkill -f "softIoc.*teem_laser"
```

## Monitoring

### Real-time Monitoring

```bash
# Watch key channels
watch -n 0.5 'caget Y1:AUX-UVD_LASER_STATE Y1:AUX-UVD_EMITTING Y1:AUX-UVD_DIODE_TEMP Y1:AUX-UVD_CRYSTAL_TEMP'

# Monitor service logs
journalctl -u teem-laser.service -f

# Check error registers
watch -n 1 'printf "EREG1: 0x%02X  EREG2: 0x%02X  EREG3: 0x%02X\n" $(caget -t Y1:AUX-UVD_EREG1) $(caget -t Y1:AUX-UVD_EREG2) $(caget -t Y1:AUX-UVD_EREG3)'
```

### Service Health

```bash
# Check service uptime
caget Y1:AUX-UVD_UPTIME

# Check heartbeat counter
caget Y1:AUX-UVD_HEARTBEAT_COUNT

# Check last error
caget Y1:AUX-UVD_LAST_ERROR
```

## Files

- `teem_laser.db` - EPICS database definition (→ /home/controls/labutils/)
- `run_teem_laser.py` - Service daemon
- `teem_laser_control.py` - User control script
- `teem-laser.service` - Systemd service (→ /etc/systemd/system/)
- `README.md` - This file

## References

- Manual: `/home/controls/Downloads/MLC03AxR1UserManual.1070254556.pdf`
- Serial Protocol: Section 5 of manual
- Error Codes: Section 5.5 of manual
- Safety: Section 1.4 of manual

## Support

For issues:
1. Check service logs: `journalctl -u teem-laser.service`
2. Check IOC logs: `journalctl -u auxioc.service`
3. Verify serial communication with minicom
4. Check manual for error code definitions

## Safety Warnings

- ⚠️ **NEVER** watch directly into the beam
- ⚠️ **NEVER** disconnect laser head while powered
- ⚠️ 5-second time delay before laser emission starts
- ⚠️ Use interlock to prevent unauthorized access
- ⚠️ Monitor temperature channels continuously
- ⚠️ Check error registers before each operation

## License

Created by Claude Sonnet 4.5 for controls@LIGO
Date: 2026-01-13
