#!/bin/bash
# abort_actuator_gain.sh — emergency abort for measure_actuator_gain.py
#
# Authoritative action: touch the sentinel file that measure_actuator_gain.py
# polls every loop. The script then raises AbortRequested, kills the diag
# excitation, ramps to zero, and restores POLES state. As a belt-and-suspenders
# fast stop we also pkill the diag process directly.
#
# Wire this to an MEDM "shell command" button (red) e.g. in
#   /opt/rtcds/yqg/y1/medm/y1dmd_scripts/Y1DMD_SCRIPTS.adl :
#   command[0].name = "xterm -hold -T 'ABORT ActGain' -e \
#       /home/controls/labutils/scripts/dipole/abort_actuator_gain.sh"
#
# Usage: abort_actuator_gain.sh [SENTINEL_PATH]
#   SENTINEL_PATH must match abort.sentinel_path in the config YAML
#   (default /tmp/abort_actuator_gain).
set -euo pipefail

SENTINEL="${1:-/tmp/abort_actuator_gain}"

touch "$SENTINEL"
echo "Abort requested: touched sentinel $SENTINEL"

# Secondary fast stop: terminate the diag measurement process if present.
if pkill -f "diag -l -f" 2>/dev/null; then
    echo "Sent terminate to running diag process."
else
    echo "No running diag process found (the python loop will safe on next poll)."
fi
