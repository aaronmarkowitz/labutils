#!/bin/bash
# restore_aux_settings.sh — Restore Y1:AUX EPICS channel values from a BURT snapshot.
#
# Called automatically by the auxioc systemd service via ExecStartPost.
# Safe to run manually at any time.
#
# Usage:
#   /home/controls/labutils/epics/restore_aux_settings.sh [snapshot_file]
#
# If snapshot_file is omitted, defaults to auxioc.snap in the same directory.

set -euo pipefail

CAPUT=/var/lib/cds-conda/base/envs/cds/epics/bin/linux-x86_64/caput
SNAP_FILE="${1:-$(dirname "$0")/auxioc.snap}"

log() { echo "[$(date '+%H:%M:%S')] restore_aux_settings: $*"; }

if [[ ! -f "$SNAP_FILE" ]]; then
    log "No snapshot file found at $SNAP_FILE — skipping restore (first boot?)"
    exit 0
fi

log "Restoring from $SNAP_FILE"

RESTORED=0
ERRORS=0

# Skip the BURT header block (lines between --- Start BURT header and --- End BURT header)
# then process data lines: "<channel> 1 <value> 1"
in_header=0
while IFS= read -r line; do
    if [[ "$line" == "--- Start BURT header" ]]; then
        in_header=1; continue
    fi
    if [[ "$line" == "--- End BURT header" ]]; then
        in_header=0; continue
    fi
    [[ $in_header -eq 1 ]] && continue
    # Skip blank lines and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

    # Parse: channel count value mask
    read -r channel _count value _mask <<< "$line" || continue
    [[ -z "$channel" || -z "$value" ]] && continue

    if "$CAPUT" "$channel" "$value" > /dev/null 2>&1; then
        log "  $channel = $value"
        RESTORED=$((RESTORED + 1))
    else
        log "  WARNING: failed to set $channel = $value"
        ERRORS=$((ERRORS + 1))
    fi
done < "$SNAP_FILE"

log "Restored $RESTORED channel(s)${ERRORS:+, $ERRORS error(s)}."
[[ $ERRORS -gt 0 ]] && exit 1 || exit 0
