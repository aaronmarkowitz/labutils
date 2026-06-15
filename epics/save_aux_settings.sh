#!/bin/bash
# save_aux_settings.sh — Save Y1:AUX EPICS channel values to a BURT-compatible snapshot.
#
# Run this after any calibration or setpoint change to persist values across restarts.
# Writes to auxioc.snap (latest) and auxioc_YYYYMMDD_HHMMSS.snap (timestamped archive).
#
# Usage:
#   /home/controls/labutils/epics/save_aux_settings.sh

set -euo pipefail

CAGET=/var/lib/cds-conda/base/envs/cds/epics/bin/linux-x86_64/caget
REQ_FILE="$(dirname "$0")/autoBurt.req"
SNAP_FILE="$(dirname "$0")/auxioc.snap"
TIMESTAMP=$(date '+%y%m%d_%H%M%S')
SNAP_TS="$(dirname "$0")/auxioc_${TIMESTAMP}.snap"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

if [[ ! -f "$REQ_FILE" ]]; then
    echo "ERROR: request file not found: $REQ_FILE" >&2
    exit 1
fi

# Collect channels (skip blank lines and comments)
mapfile -t CHANNELS < <(grep -v '^\s*#' "$REQ_FILE" | grep -v '^\s*$')

if [[ ${#CHANNELS[@]} -eq 0 ]]; then
    echo "ERROR: no channels found in $REQ_FILE" >&2
    exit 1
fi

log "Saving ${#CHANNELS[@]} channels from $REQ_FILE"

# Write BURT header
write_header() {
    local file="$1"
    cat > "$file" <<HEADER
--- Start BURT header
Time:      $(date)
Login ID: $(whoami) ()
Eff  UID: $(id -u)
Group ID: $(id -g)
Keywords:
Comments:
Type:     Absolute
Directory: $(dirname "$SNAP_FILE")/
Req File: autoBurt.req
--- End BURT header
HEADER
}

write_header "$SNAP_FILE"
write_header "$SNAP_TS"

ERRORS=0
for ch in "${CHANNELS[@]}"; do
    val=$("$CAGET" -t "$ch" 2>/dev/null) || { log "WARNING: could not read $ch"; ERRORS=$((ERRORS+1)); continue; }
    echo "$ch 1 $val 1" | tee -a "$SNAP_FILE" >> "$SNAP_TS"
    log "  $ch = $val"
done

if [[ $ERRORS -gt 0 ]]; then
    log "WARNING: $ERRORS channel(s) could not be read and were skipped."
fi

log "Snapshot written to: $SNAP_FILE"
log "Timestamped copy:    $SNAP_TS"
