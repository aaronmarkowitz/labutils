#!/bin/bash
# disengage_intensity_servo_x.sh — Safely deactivate X laser intensity stabilization servo
#
# This script:
#   1. Ramps CTLX gain to 0 (smoothly disables control output)
#   2. Waits for ramp to complete
#   3. Sets CTLX LIMIT to 0 (clamp output)
#   4. Restores SWITCHX to route particle control
#
# Usage:
#   ./disengage_intensity_servo_x.sh            # Execute
#   ./disengage_intensity_servo_x.sh --dry-run  # Print without executing

set -euo pipefail

GAIN_TRAMP=10         # seconds for gain ramp to 0
SWITCH_TRAMP=5        # seconds for SWITCHX ramp
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        --help|-h)
            echo "Usage: $0 [--dry-run]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PREFIX="Y1:DMD"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

do_caput() {
    if [[ $DRY_RUN -eq 1 ]]; then
        log "DRY-RUN: caput $*"
    else
        log "Setting: $1 = $2"
        caput "$1" "$2" > /dev/null
    fi
}

do_caget() {
    caget -t "$1" 2>/dev/null
}

log "=== X Laser Intensity Servo Disengagement ==="
if [[ $DRY_RUN -eq 1 ]]; then
    log "*** DRY RUN MODE — no changes will be made ***"
fi

# Show current state
CURRENT_GAIN=$(do_caget "${PREFIX}-LASER_CTLX_GAIN")
CTLX_OUT=$(do_caget "${PREFIX}-LASER_CTLX_OUT16")
POX_DC=$(do_caget "${PREFIX}-LASER_POX_OUT16")
SW_1_1=$(do_caget "${PREFIX}-LASER_SWITCHX_1_1")
SW_1_2=$(do_caget "${PREFIX}-LASER_SWITCHX_1_2")

log "Current state:"
log "  CTLX GAIN:     $CURRENT_GAIN"
log "  CTLX OUT16:    $CTLX_OUT"
log "  POX OUT16:     $POX_DC"
log "  SWITCHX [1,1]: $SW_1_1"
log "  SWITCHX [1,2]: $SW_1_2"

# --- Step 1: Ramp CTLX gain to 0 ---
log ""
log "Step 1: Ramping CTLX gain to 0 (TRAMP=${GAIN_TRAMP}s)..."
do_caput "${PREFIX}-LASER_CTLX_TRAMP" "$GAIN_TRAMP"
do_caput "${PREFIX}-LASER_CTLX_GAIN" 0

if [[ $DRY_RUN -eq 0 ]]; then
    log "  Waiting ${GAIN_TRAMP}s for gain ramp..."
    sleep $((GAIN_TRAMP + 2))
    log "  CTLX GAIN = $(do_caget ${PREFIX}-LASER_CTLX_GAIN)"
    log "  CTLX OUT16 = $(do_caget ${PREFIX}-LASER_CTLX_OUT16)"
fi

# --- Step 2: Set CTLX LIMIT to 0 ---
log ""
log "Step 2: Setting CTLX LIMIT to 0 (clamp output)..."
do_caput "${PREFIX}-LASER_CTLX_LIMIT" 0

# --- Step 3: Restore SWITCHX to particle control ---
log ""
log "Step 3: Restoring SWITCHX to particle control (1_1=0, 1_2=1)..."
do_caput "${PREFIX}-LASER_SWITCHX_SETTING_1_1" 0
do_caput "${PREFIX}-LASER_SWITCHX_SETTING_1_2" 1
do_caput "${PREFIX}-LASER_SWITCHX_TRAMP" "$SWITCH_TRAMP"
log "  Loading matrix..."
do_caput "${PREFIX}-LASER_SWITCHX_LOAD_MATRIX" 1

if [[ $DRY_RUN -eq 0 ]]; then
    log "  Waiting ${SWITCH_TRAMP}s for SWITCHX ramp..."
    sleep $((SWITCH_TRAMP + 1))
fi

# --- Verify ---
log ""
log "Verification:"
if [[ $DRY_RUN -eq 0 ]]; then
    log "  CTLX GAIN:     $(do_caget ${PREFIX}-LASER_CTLX_GAIN)"
    log "  CTLX OUT16:    $(do_caget ${PREFIX}-LASER_CTLX_OUT16)"
    log "  CTLX LIMIT:    $(do_caget ${PREFIX}-LASER_CTLX_LIMIT)"
    log "  SWITCHX [1,1]: $(do_caget ${PREFIX}-LASER_SWITCHX_1_1)"
    log "  SWITCHX [1,2]: $(do_caget ${PREFIX}-LASER_SWITCHX_1_2)"
    log "  IMODX OUT16:   $(do_caget ${PREFIX}-LASER_IMODX_OUT16)"
else
    log "  (dry-run: verification skipped)"
fi

log ""
log "=== Servo disengaged ==="
