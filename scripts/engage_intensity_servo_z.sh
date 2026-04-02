#!/bin/bash
# engage_intensity_servo_z.sh — Safely activate Z laser intensity stabilization servo
#
# This script:
#   1. Zeros CTLZ gain to prevent integrator windup
#   2. Ensures CTLZ OFFSET switch is enabled
#   3. Reads POZ DC level, converts to mW using cal factor from Y1:AUX-ISS_Z_CAL,
#      and writes mW setpoint to Y1:AUX-ISS_Z_SETPOINT_MW (calcout pushes OFFSET)
#   4. Routes CTLZ output through SWITCHZ to IMODZ
#   5. Sets CTLZ LIMIT to allow servo output
#   6. Ramps CTLZ gain to working value (-100)
#   7. Verifies servo is working
#
# Usage:
#   ./engage_intensity_servo_z.sh            # Execute the engagement sequence
#   ./engage_intensity_servo_z.sh --dry-run  # Print what would be done without changing anything
#   ./engage_intensity_servo_z.sh --gain -50 # Use a different target gain
#   ./engage_intensity_servo_z.sh --limit 2000  # Use a different output limit

set -euo pipefail

# --- Configuration ---
TARGET_GAIN="${TARGET_GAIN:--100}"
TARGET_LIMIT="${TARGET_LIMIT:-3000}"
SWITCH_TRAMP=5        # seconds for SWITCHZ ramp
GAIN_TRAMP=30         # seconds for CTLZ gain ramp
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
            echo "  --gain       Target CTLZ gain (default: -100)"
            echo "  --limit      Target CTLZ limit (default: 3000)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PREFIX="Y1:DMD"
AUX_PREFIX="Y1:AUX"

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
    local statefile="/tmp/y1dmd_servo_z_pre_engage_$(date +%Y%m%d_%H%M%S).txt"
    log "Saving pre-engagement state to $statefile"
    {
        echo "# Pre-engagement state saved at $(date)"
        echo "CTLZ_GAIN=$(do_caget ${PREFIX}-LASER_CTLZ_GAIN)"
        echo "CTLZ_OFFSET=$(do_caget ${PREFIX}-LASER_CTLZ_OFFSET)"
        echo "CTLZ_LIMIT=$(do_caget ${PREFIX}-LASER_CTLZ_LIMIT)"
        echo "CTLZ_TRAMP=$(do_caget ${PREFIX}-LASER_CTLZ_TRAMP)"
        echo "SWITCHZ_1_1=$(do_caget ${PREFIX}-LASER_SWITCHZ_1_1)"
        echo "SWITCHZ_1_2=$(do_caget ${PREFIX}-LASER_SWITCHZ_1_2)"
        echo "SWITCHZ_TRAMP=$(do_caget ${PREFIX}-LASER_SWITCHZ_TRAMP)"
        echo "POZ_OUT16=$(do_caget ${PREFIX}-LASER_POZ_OUT16)"
    } > "$statefile"
    echo "$statefile"
}

# --- Pre-flight checks ---
log "=== Z Laser Intensity Servo Engagement ==="
if [[ $DRY_RUN -eq 1 ]]; then
    log "*** DRY RUN MODE — no changes will be made ***"
fi

# Check that EPICS is responsive
POZ_DC=$(do_caget "${PREFIX}-LASER_POZ_OUT16")
if [[ -z "$POZ_DC" ]]; then
    log "ERROR: Cannot read ${PREFIX}-LASER_POZ_OUT16. Is the model running?"
    exit 1
fi

CTLZ_SWSTR=$(do_caget "${PREFIX}-LASER_CTLZ_SWSTR")
CURRENT_GAIN=$(do_caget "${PREFIX}-LASER_CTLZ_GAIN")
CURRENT_OFFSET=$(do_caget "${PREFIX}-LASER_CTLZ_OFFSET")
CURRENT_LIMIT=$(do_caget "${PREFIX}-LASER_CTLZ_LIMIT")
SW_1_1=$(do_caget "${PREFIX}-LASER_SWITCHZ_1_1")
SW_1_2=$(do_caget "${PREFIX}-LASER_SWITCHZ_1_2")

# Read calibration factor from softIOC
CAL_FACTOR=$(do_caget "${AUX_PREFIX}-ISS_Z_CAL")
if [[ -z "$CAL_FACTOR" ]] || python3 -c "import sys; sys.exit(0 if abs(float('$CAL_FACTOR')) < 0.001 else 1)" 2>/dev/null; then
    log "ERROR: Cannot read calibration factor from ${AUX_PREFIX}-ISS_Z_CAL (got: '$CAL_FACTOR')."
    log "  Set it first: caput ${AUX_PREFIX}-ISS_Z_CAL <counts_per_mW>"
    exit 1
fi

log "Current state:"
log "  POZ DC level:     $POZ_DC counts"
log "  Cal factor:       $CAL_FACTOR counts/mW"
log "  CTLZ switches:    $CTLZ_SWSTR"
log "  CTLZ GAIN:        $CURRENT_GAIN"
log "  CTLZ OFFSET:      $CURRENT_OFFSET"
log "  CTLZ LIMIT:       $CURRENT_LIMIT"
log "  SWITCHZ [1,1]:    $SW_1_1 (CTLZ path)"
log "  SWITCHZ [1,2]:    $SW_1_2 (particle path)"

# Compute setpoint in mW from POZ DC level and cal factor
SETPOINT_MW=$(python3 -c "print(round(float('$POZ_DC') / float('$CAL_FACTOR'), 3))")
SETPOINT_COUNTS=$(python3 -c "print(int(-round(float('$POZ_DC'))))")
log "  Setpoint:         $SETPOINT_MW mW (= $POZ_DC / $CAL_FACTOR)"
log "  Expected OFFSET:  $SETPOINT_COUNTS counts (= -round($POZ_DC))"
log ""

# Check CTLZ has required filters
if [[ "$CTLZ_SWSTR" != *"IN"* ]] || [[ "$CTLZ_SWSTR" != *"OT"* ]]; then
    log "WARNING: CTLZ does not have INPUT and/or OUTPUT enabled: $CTLZ_SWSTR"
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

# --- Step 1: Zero CTLZ gain ---
log ""
log "Step 1: Zeroing CTLZ gain (prevent integrator windup)..."
do_caput "${PREFIX}-LASER_CTLZ_GAIN" 0

if [[ $DRY_RUN -eq 0 ]]; then
    # Wait briefly for gain to start ramping to 0
    sleep 2
    log "  CTLZ OUT16 = $(do_caget ${PREFIX}-LASER_CTLZ_OUT16)"
fi

# --- Step 2: Ensure OFFSET switch is enabled ---
log ""
if [[ "$CTLZ_SWSTR" != *"OF"* ]]; then
    log "Step 2a: Enabling CTLZ OFFSET switch..."
    # Read current SW1 state, set OFFSET bit (bit 1), write back
    CURRENT_SW1=$(do_caget "${PREFIX}-LASER_CTLZ_SW1S")
    NEW_SW1=$(python3 -c "print(int('$CURRENT_SW1') | 0x2)")
    do_caput "${PREFIX}-LASER_CTLZ_SW1S" "$NEW_SW1"
    if [[ $DRY_RUN -eq 0 ]]; then
        sleep 1
        log "  CTLZ switches: $(do_caget ${PREFIX}-LASER_CTLZ_SWSTR)"
    fi
else
    log "Step 2a: CTLZ OFFSET switch already enabled."
fi

# --- Step 2b: Set setpoint in mW (calcout converts to counts and pushes OFFSET) ---
log ""
log "Step 2b: Setting ISS X setpoint to $SETPOINT_MW mW..."
do_caput "${AUX_PREFIX}-ISS_Z_SETPOINT_MW" "$SETPOINT_MW"
if [[ $DRY_RUN -eq 0 ]]; then
    sleep 1
    log "  CTLZ OFFSET = $(do_caget ${PREFIX}-LASER_CTLZ_OFFSET) (expected: $SETPOINT_COUNTS)"
fi

# --- Step 3: Route CTLZ through SWITCHZ ---
log ""
log "Step 3: Setting SWITCHZ to route CTLZ output (1_1=1, 1_2=0)..."
do_caput "${PREFIX}-LASER_SWITCHZ_SETTING_1_1" 1
do_caput "${PREFIX}-LASER_SWITCHZ_SETTING_1_2" 0
do_caput "${PREFIX}-LASER_SWITCHZ_TRAMP" "$SWITCH_TRAMP"
log "  Loading matrix..."
do_caput "${PREFIX}-LASER_SWITCHZ_LOAD_MATRIX" 1

if [[ $DRY_RUN -eq 0 ]]; then
    log "  Waiting ${SWITCH_TRAMP}s for SWITCHZ ramp..."
    sleep $((SWITCH_TRAMP + 1))
    log "  SWITCHZ [1,1] = $(do_caget ${PREFIX}-LASER_SWITCHZ_1_1)"
    log "  SWITCHZ [1,2] = $(do_caget ${PREFIX}-LASER_SWITCHZ_1_2)"
fi

# --- Step 4: Set CTLZ LIMIT ---
log ""
log "Step 4: Setting CTLZ LIMIT to $TARGET_LIMIT..."
do_caput "${PREFIX}-LASER_CTLZ_LIMIT" "$TARGET_LIMIT"

# --- Step 5: Ramp GAIN to target ---
log ""
log "Step 5: Ramping CTLZ gain to $TARGET_GAIN (TRAMP=${GAIN_TRAMP}s)..."
do_caput "${PREFIX}-LASER_CTLZ_TRAMP" "$GAIN_TRAMP"
do_caput "${PREFIX}-LASER_CTLZ_GAIN" "$TARGET_GAIN"

if [[ $DRY_RUN -eq 0 ]]; then
    log "  Waiting ${GAIN_TRAMP}s for gain ramp to complete..."
    sleep $((GAIN_TRAMP + 5))
fi

# --- Step 6: Verify ---
log ""
log "Step 6: Verification..."
if [[ $DRY_RUN -eq 0 ]]; then
    CTLZ_OUT=$(do_caget "${PREFIX}-LASER_CTLZ_OUT16")
    POZ_NEW=$(do_caget "${PREFIX}-LASER_POZ_OUT16")
    IMODZ_OUT=$(do_caget "${PREFIX}-LASER_IMODZ_OUT16")
    GAIN_NOW=$(do_caget "${PREFIX}-LASER_CTLZ_GAIN")

    log "  CTLZ GAIN:    $GAIN_NOW  (target: $TARGET_GAIN)"
    log "  CTLZ OUT16:   $CTLZ_OUT  (should be non-zero)"
    log "  POZ OUT16:    $POZ_NEW  (setpoint: $SETPOINT_MW mW)"
    log "  IMODZ OUT16:  $IMODZ_OUT  (should be non-zero)"

    # Simple sanity check
    CTLZ_ABS=$(python3 -c "print(abs(float('$CTLZ_OUT')))")
    if python3 -c "import sys; sys.exit(0 if float('$CTLZ_ABS') > 0.1 else 1)"; then
        log ""
        log "SUCCESS: Servo appears to be active (CTLZ output is non-zero)."
    else
        log ""
        log "WARNING: CTLZ output is near zero. Servo may not be working."
        log "  Check: Is LIMIT still 0? Is the integrator (FM1) enabled?"
        log "  CTLZ switches: $(do_caget ${PREFIX}-LASER_CTLZ_SWSTR)"
    fi
else
    log "  (dry-run: verification skipped)"
fi

log ""
log "=== Done ==="
