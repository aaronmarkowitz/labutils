#!/usr/bin/env python3
"""Map and display the current state of the Y1DMD rtcds control system.

Reads EPICS channels for all filter modules, matrices, and switches,
and optionally parses the .mdl file for connectivity information.
Outputs a human-readable summary and optionally saves a JSON snapshot.

Usage:
    python3 map_y1dmd_state.py              # Print summary to stdout
    python3 map_y1dmd_state.py --json       # Also save timestamped JSON snapshot
    python3 map_y1dmd_state.py --compact    # Compact output (non-zero values only)
    python3 map_y1dmd_state.py --section LASER  # Only show LASER subsystem
"""

import subprocess
import json
import argparse
import re
import sys
import os
from datetime import datetime
from collections import OrderedDict

PREFIX = "Y1:DMD"

# Filter modules grouped by subsystem
FILTER_MODULES = OrderedDict([
    ("LASER", [
        "LASER_POX", "LASER_POZ",
        "LASER_CTLX", "LASER_CTLZ",
        "LASER_IMODX", "LASER_IMODZ",
    ]),
    ("LESX", ["LESX_PIT", "LESX_SUM", "LESX_YAW"]),
    ("LESZ", ["LESZ_PIT", "LESZ_SUM", "LESZ_YAW"]),
    ("PARTICLE", [
        "PARTICLE_X", "PARTICLE_Y", "PARTICLE_Z",
        "PARTICLE_I", "PARTICLE_Q", "PARTICLE_PHI",
        "PARTICLE_CHARGE",
    ]),
    ("POLES", ["POLES_DC", "POLES_E1", "POLES_E2", "POLES_E3", "POLES_E4"]),
    ("MISC", ["FIL", "FM10", "FM11", "UVD"]),
])

# Matrix blocks: name -> (rows, cols)
MATRICES = OrderedDict([
    ("ACTS", (6, 6)),
    ("SENSE", (4, 6)),
])

# RampMuxMatrix switches: name -> (outputs, inputs)
SWITCHES = OrderedDict([
    ("LASER_SWITCHX", (1, 2)),
    ("LASER_SWITCHZ", (1, 2)),
])

# Per-filter-module channels to read
FM_CHANNELS = ["GAIN", "OFFSET", "LIMIT", "TRAMP", "SWSTR", "OUT16", "INMON"]


def caget(channels):
    """Read multiple EPICS channels via caget. Returns dict of channel->value."""
    if not channels:
        return {}
    # Use caget with -t (terse) for numeric, but SWSTR needs string format
    # Split into batches to avoid command line length limits
    results = {}
    batch_size = 50
    for i in range(0, len(channels), batch_size):
        batch = channels[i:i + batch_size]
        try:
            proc = subprocess.run(
                ["caget"] + batch,
                capture_output=True, text=True, timeout=10
            )
            for line in proc.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    results[parts[0]] = parts[1].strip()
                elif len(parts) == 1:
                    results[parts[0]] = ""
        except subprocess.TimeoutExpired:
            for ch in batch:
                results[ch] = "<timeout>"
        except Exception as e:
            for ch in batch:
                results[ch] = f"<error: {e}>"
    return results


def read_filter_module(mod_name):
    """Read all channels for a single filter module."""
    channels = [f"{PREFIX}-{mod_name}_{ch}" for ch in FM_CHANNELS]
    raw = caget(channels)
    result = {}
    for ch in FM_CHANNELS:
        full = f"{PREFIX}-{mod_name}_{ch}"
        val = raw.get(full, "<missing>")
        # Try to convert numeric values
        if ch != "SWSTR":
            try:
                val = float(val)
                if val == int(val):
                    val = int(val)
            except (ValueError, TypeError):
                pass
        result[ch] = val
    return result


def read_matrix(mat_name, rows, cols):
    """Read all elements of a matrix (ACTS, SENSE)."""
    channels = []
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            channels.append(f"{PREFIX}-{mat_name}_{r}_{c}")
            channels.append(f"{PREFIX}-{mat_name}_{r}_{c}_GAIN")
    raw = caget(channels)
    matrix = {}
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            key = f"{r}_{c}"
            gain_ch = f"{PREFIX}-{mat_name}_{r}_{c}_GAIN"
            gain_val = raw.get(gain_ch, "<missing>")
            try:
                gain_val = float(gain_val)
                if gain_val == int(gain_val):
                    gain_val = int(gain_val)
            except (ValueError, TypeError):
                pass
            matrix[key] = gain_val
    return matrix


def read_switch(sw_name, outputs, inputs):
    """Read a RampMuxMatrix switch state."""
    channels = []
    for o in range(1, outputs + 1):
        for i in range(1, inputs + 1):
            channels.append(f"{PREFIX}-{sw_name}_{o}_{i}")
            channels.append(f"{PREFIX}-{sw_name}_SETTING_{o}_{i}")
            channels.append(f"{PREFIX}-{sw_name}_RAMPING_{o}_{i}")
    channels.append(f"{PREFIX}-{sw_name}_TRAMP")
    raw = caget(channels)

    result = {"TRAMP": None, "elements": {}}
    tramp_ch = f"{PREFIX}-{sw_name}_TRAMP"
    try:
        result["TRAMP"] = float(raw.get(tramp_ch, "0"))
    except (ValueError, TypeError):
        result["TRAMP"] = raw.get(tramp_ch, "<missing>")

    for o in range(1, outputs + 1):
        for i in range(1, inputs + 1):
            key = f"{o}_{i}"
            current_ch = f"{PREFIX}-{sw_name}_{o}_{i}"
            setting_ch = f"{PREFIX}-{sw_name}_SETTING_{o}_{i}"
            ramping_ch = f"{PREFIX}-{sw_name}_RAMPING_{o}_{i}"
            elem = {}
            for label, ch in [("current", current_ch), ("setting", setting_ch), ("ramping", ramping_ch)]:
                val = raw.get(ch, "<missing>")
                try:
                    val = float(val)
                    if val == int(val):
                        val = int(val)
                except (ValueError, TypeError):
                    pass
                elem[label] = val
            result["elements"][key] = elem
    return result


def read_extra_channels():
    """Read miscellaneous channels of interest."""
    extras = [
        f"{PREFIX}-PARTICLE_LO_FREQ",
        f"{PREFIX}-PARTICLE_LO_TRAMP",
        "Y1:FEC-11_BURT_RESTORE",
        "Y1:FEC-11_DACDT_ENABLE",
    ]
    return caget(extras)


def format_filter_module(name, data, compact=False):
    """Format a filter module for display."""
    if compact:
        # Skip if all outputs are zero and no interesting settings
        if (data.get("GAIN") == 0 and data.get("OFFSET") == 0
                and data.get("OUT16") == 0):
            return None
    swstr = data.get("SWSTR", "")
    gain = data.get("GAIN", "?")
    offset = data.get("OFFSET", "?")
    limit = data.get("LIMIT", "?")
    tramp = data.get("TRAMP", "?")
    out16 = data.get("OUT16", "?")
    inmon = data.get("INMON", "?")
    lines = []
    lines.append(f"  {name}:")
    lines.append(f"    Switches: {swstr}")
    lines.append(f"    GAIN={gain}  OFFSET={offset}  LIMIT={limit}  TRAMP={tramp}")
    lines.append(f"    INMON={inmon}  OUT16={out16}")
    return "\n".join(lines)


def format_matrix(name, matrix, rows, cols, compact=False):
    """Format a matrix as a grid."""
    lines = []
    lines.append(f"  {name} ({rows}x{cols}):")
    # Find non-zero elements
    nonzero = {k: v for k, v in matrix.items() if v != 0}
    if compact and not nonzero:
        lines.append("    (all zeros)")
        return "\n".join(lines)
    if compact:
        for k, v in sorted(nonzero.items()):
            lines.append(f"    [{k}] = {v}")
    else:
        # Full grid display
        header = "      " + "".join(f"{'C'+str(c):>8}" for c in range(1, cols + 1))
        lines.append(header)
        for r in range(1, rows + 1):
            row_vals = []
            for c in range(1, cols + 1):
                val = matrix.get(f"{r}_{c}", 0)
                row_vals.append(f"{val:>8}")
            lines.append(f"  R{r}  " + "".join(row_vals))
    return "\n".join(lines)


def format_switch(name, data):
    """Format a RampMuxMatrix switch."""
    lines = []
    lines.append(f"  {name}:  (TRAMP={data['TRAMP']}s)")
    for key, elem in sorted(data["elements"].items()):
        cur = elem["current"]
        sett = elem["setting"]
        ramp = elem["ramping"]
        lines.append(f"    [{key}] current={cur}  setting={sett}  ramping={ramp}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Map Y1DMD control system state")
    parser.add_argument("--json", action="store_true", help="Save JSON snapshot")
    parser.add_argument("--compact", action="store_true", help="Show only non-zero/active values")
    parser.add_argument("--section", type=str, default=None,
                        help="Only show a specific subsystem (LASER, LESX, LESZ, PARTICLE, POLES, MISC)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"=" * 70)
    print(f"  Y1DMD Control System State — {timestamp}")
    print(f"=" * 70)

    all_data = {"timestamp": timestamp, "filter_modules": {}, "matrices": {}, "switches": {}, "extras": {}}

    # Read extra channels first
    extras = read_extra_channels()
    all_data["extras"] = extras
    print(f"\n--- System Status ---")
    for ch, val in sorted(extras.items()):
        short = ch.replace("Y1:FEC-11_", "FEC-11:").replace(f"{PREFIX}-", "")
        print(f"  {short} = {val}")

    # Read filter modules
    print(f"\n{'=' * 70}")
    print(f"  FILTER MODULES")
    print(f"{'=' * 70}")
    for section, modules in FILTER_MODULES.items():
        if args.section and args.section.upper() != section:
            continue
        section_has_output = False
        section_lines = []
        for mod in modules:
            data = read_filter_module(mod)
            all_data["filter_modules"][mod] = data
            formatted = format_filter_module(mod, data, compact=args.compact)
            if formatted is not None:
                section_lines.append(formatted)
                section_has_output = True
        if section_has_output or not args.compact:
            print(f"\n--- {section} ---")
            if section_lines:
                print("\n".join(section_lines))
            elif args.compact:
                print("  (all inactive)")

    # Read matrices
    if not args.section or args.section.upper() in ("ACTS", "SENSE", "MATRICES"):
        print(f"\n{'=' * 70}")
        print(f"  MATRICES")
        print(f"{'=' * 70}")
        for mat_name, (rows, cols) in MATRICES.items():
            matrix = read_matrix(mat_name, rows, cols)
            all_data["matrices"][mat_name] = matrix
            print(f"\n{format_matrix(mat_name, matrix, rows, cols, compact=args.compact)}")

    # Read switches
    if not args.section or args.section.upper() in ("LASER", "SWITCHES"):
        print(f"\n{'=' * 70}")
        print(f"  SWITCHES (RampMuxMatrix)")
        print(f"{'=' * 70}")
        for sw_name, (outputs, inputs) in SWITCHES.items():
            data = read_switch(sw_name, outputs, inputs)
            all_data["switches"][sw_name] = data
            print(f"\n{format_switch(sw_name, data)}")

    print(f"\n{'=' * 70}")

    # Save JSON snapshot if requested
    if args.json:
        snap_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        snap_file = os.path.join(snap_dir, f"y1dmd_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(snap_file, "w") as f:
            json.dump(all_data, f, indent=2, default=str)
        print(f"\nSnapshot saved to: {snap_file}")


if __name__ == "__main__":
    main()
