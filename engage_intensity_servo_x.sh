#!/bin/bash
# engage_intensity_servo_x.sh — Safely activate X laser intensity stabilization servo
#
# This script:
#   1. Zeros CTLX gain to prevent integrator windup
#   2. Sets CTLX offset (setpoint) to match current POX DC level
#   3. Routes CTLX output through SWITCHX to IMODX
#   4. Sets CTLX LIMIT to allow servo output
#   5. Ramps CTLX gain to working value (-100)
#   6. Verifies servo is working
#
# Usage:
#   ./engage_intensity_servo_x.sh            # Execute the engagement sequence
#   ./engage_intensity_servo_x.sh --dry-run  # Print what would be done without changing anything
#   ./engage_intensity_servo_x.sh --gain -50 # Use a different target gain
#   ./engage_intensity_servo_x.sh --limit 2000  # Use a different output limit

set -euo pipefail

# --- Configuration ---
TARGET_GAIN="${TARGET_GAIN:--100}"
TARGET_LIMIT="${TARGET_LIMIT:-3000}"
SWITCH_TRAMP=5        # seconds for SWITCHX ramp
GAIN_TRAMP=30         # seconds for CTLX gain ramp
DRY_RUN=0

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        --gain)     TARGET_GAIN="$2"; shift 2 ;;
        --limit)    TARGET_LIMIT="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--gain VALUE] [--limit VALUE]"
            echo "  --dry-run    Print actions without executing"
            echo "  --gain       Target CTLX gain (default: -100)"
            echo "  --limit      Target CTLX limit (default: 3000)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PREFIX="Y1:DMD"

# --- Helper functions ---
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

save_state() {
    # Save current state for potential rollback
    local statefile="/tmp/y1dmd_servo_x_pre_engage_$(date +%Y%m%d_%H%M%S).txt"
    log "Saving pre-engagement state to $statefile"
    {
        echo "# Pre-engagement state saved at $(date)"
        echo "CTLX_GAIN=$(do_caget ${PREFIX}-LASER_CTLX_GAIN)"
        echo "CTLX_OFFSET=$(do_caget ${PREFIX}-LASER_CTLX_OFFSET)"
        echo "CTLX_LIMIT=$(do_caget ${PREFIX}-LASER_CTLX_LIMIT)"
        echo "CTLX_TRAMP=$(do_caget ${PREFIX}-LASER_CTLX_TRAMP)"
        echo "SWITCHX_1_1=$(do_caget ${PREFIX}-LASER_SWITCHX_1_1)"
        echo "SWITCHX_1_2=$(do_caget ${PREFIX}-LASER_SWITCHX_1_2)"
        echo "SWITCHX_TRAMP=$(do_caget ${PREFIX}-LASER_SWITCHX_TRAMP)"
        echo "POX_OUT16=$(do_caget ${PREFIX}-LASER_POX_OUT16)"
    } > "$statefile"
    echo "$statefile"
}

# --- Pre-flight checks ---
log "=== X Laser Intensity Servo Engagement ==="
if [[ $DRY_RUN -eq 1 ]]; then
    log "*** DRY RUN MODE — no changes will be made ***"
fi

# Check that EPICS is responsive
POX_DC=$(do_caget "${PREFIX}-LASER_POX_OUT16")
if [[ -z "$POX_DC" ]]; then
    log "ERROR: Cannot read ${PREFIX}-LASER_POX_OUT16. Is the model running?"
    exit 1
fi

CTLX_SWSTR=$(do_caget "${PREFIX}-LASER_CTLX_SWSTR")
CURRENT_GAIN=$(do_caget "${PREFIX}-LASER_CTLX_GAIN")
CURRENT_OFFSET=$(do_caget "${PREFIX}-LASER_CTLX_OFFSET")
CURRENT_LIMIT=$(do_caget "${PREFIX}-LASER_CTLX_LIMIT")
SW_1_1=$(do_caget "${PREFIX}-LASER_SWITCHX_1_1")
SW_1_2=$(do_caget "${PREFIX}-LASER_SWITCHX_1_2")

log "Current state:"
log "  POX DC level:     $POX_DC counts"
log "  CTLX switches:    $CTLX_SWSTR"
log "  CTLX GAIN:        $CURRENT_GAIN"
log "  CTLX OFFSET:      $CURRENT_OFFSET"
log "  CTLX LIMIT:       $CURRENT_LIMIT"
log "  SWITCHX [1,1]:    $SW_1_1 (CTLX path)"
log "  SWITCHX [1,2]:    $SW_1_2 (particle path)"

# Compute setpoint (negate POX DC)
SETPOINT=$(python3 -c "print(int(-round(float('$POX_DC'))))")
log "  Computed setpoint: $SETPOINT (= -round($POX_DC))"
log ""

# Check CTLX has required filters
if [[ "$CTLX_SWSTR" != *"IN"* ]] || [[ "$CTLX_SWSTR" != *"OT"* ]]; then
    log "WARNING: CTLX does not have INPUT and/or OUTPUT enabled: $CTLX_SWSTR"
    log "         Expected at minimum: IN,OF,1,OT"
    if [[ $DRY_RUN -eq 0 ]]; then
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Aborted."
            exit 1
        fi
    fi
fi

# --- Save pre-engagement state ---
if [[ $DRY_RUN -eq 0 ]]; then
    STATEFILE=$(save_state)
    log "State saved to: $STATEFILE"
fi

# --- Step 1: Zero CTLX gain ---
log ""
log "Step 1: Zeroing CTLX gain (prevent integrator windup)..."
do_caput "${PREFIX}-LASER_CTLX_GAIN" 0

if [[ $DRY_RUN -eq 0 ]]; then
    # Wait briefly for gain to start ramping to 0
    sleep 2
    log "  CTLX OUT16 = $(do_caget ${PREFIX}-LASER_CTLX_OUT16)"
fi

# --- Step 2: Set offset (setpoint) ---
log ""
log "Step 2: Setting CTLX offset (setpoint) to $SETPOINT..."
do_caput "${PREFIX}-LASER_CTLX_OFFSET" "$SETPOINT"

# --- Step 3: Route CTLX through SWITCHX ---
log ""
log "Step 3: Setting SWITCHX to route CTLX output (1_1=1, 1_2=0)..."
do_caput "${PREFIX}-LASER_SWITCHX_SETTING_1_1" 1
do_caput "${PREFIX}-LASER_SWITCHX_SETTING_1_2" 0
do_caput "${PREFIX}-LASER_SWITCHX_TRAMP" "$SWITCH_TRAMP"
log "  Loading matrix..."
do_caput "${PREFIX}-LASER_SWITCHX_LOAD_MATRIX" 1

if [[ $DRY_RUN -eq 0 ]]; then
    log "  Waiting ${SWITCH_TRAMP}s for SWITCHX ramp..."
    sleep $((SWITCH_TRAMP + 1))
    log "  SWITCHX [1,1] = $(do_caget ${PREFIX}-LASER_SWITCHX_1_1)"
    log "  SWITCHX [1,2] = $(do_caget ${PREFIX}-LASER_SWITCHX_1_2)"
fi

# --- Step 4: Set CTLX LIMIT ---
log ""
log "Step 4: Setting CTLX LIMIT to $TARGET_LIMIT..."
do_caput "${PREFIX}-LASER_CTLX_LIMIT" "$TARGET_LIMIT"

# --- Step 5: Ramp GAIN to target ---
log ""
log "Step 5: Ramping CTLX gain to $TARGET_GAIN (TRAMP=${GAIN_TRAMP}s)..."
do_caput "${PREFIX}-LASER_CTLX_TRAMP" "$GAIN_TRAMP"
do_caput "${PREFIX}-LASER_CTLX_GAIN" "$TARGET_GAIN"

if [[ $DRY_RUN -eq 0 ]]; then
    log "  Waiting ${GAIN_TRAMP}s for gain ramp to complete..."
    sleep $((GAIN_TRAMP + 5))
fi

# --- Step 6: Verify ---
log ""
log "Step 6: Verification..."
if [[ $DRY_RUN -eq 0 ]]; then
    CTLX_OUT=$(do_caget "${PREFIX}-LASER_CTLX_OUT16")
    POX_NEW=$(do_caget "${PREFIX}-LASER_POX_OUT16")
    IMODX_OUT=$(do_caget "${PREFIX}-LASER_IMODX_OUT16")
    GAIN_NOW=$(do_caget "${PREFIX}-LASER_CTLX_GAIN")

    log "  CTLX GAIN:    $GAIN_NOW  (target: $TARGET_GAIN)"
    log "  CTLX OUT16:   $CTLX_OUT  (should be non-zero)"
    log "  POX OUT16:    $POX_NEW  (setpoint: $SETPOINT)"
    log "  IMODX OUT16:  $IMODX_OUT  (should be non-zero)"

    # Simple sanity check
    CTLX_ABS=$(python3 -c "print(abs(float('$CTLX_OUT')))")
    if python3 -c "import sys; sys.exit(0 if float('$CTLX_ABS') > 0.1 else 1)"; then
        log ""
        log "SUCCESS: Servo appears to be active (CTLX output is non-zero)."
    else
        log ""
        log "WARNING: CTLX output is near zero. Servo may not be working."
        log "  Check: Is LIMIT still 0? Is the integrator (FM1) enabled?"
        log "  CTLX switches: $(do_caget ${PREFIX}-LASER_CTLX_SWSTR)"
    fi
else
    log "  (dry-run: verification skipped)"
fi

log ""
log "=== Done ==="
