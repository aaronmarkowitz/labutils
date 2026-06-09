#!/usr/bin/env python3
"""Upload W matrix from step 01 HDF5 to the Y1:DMD SENSE matrix via EPICS caput.

Reads the demodulation matrix W (3xN) produced by dipole_pipeline step 01
and writes the corresponding elements to Y1:DMD-SENSE_{row}_{col}_GAIN.

Only the SENSE columns corresponding to channels actually used in the
diagonalization are written; all other matrix elements are left untouched.

Usage:
    python3 upload_sense_matrix.py <hdf5_path> [--config <yml>] [--dry-run]

Examples:
    # Preview what would be written (no caput calls made):
    python3 upload_sense_matrix.py results.h5 --dry-run

    # Write to live EPICS channels:
    python3 upload_sense_matrix.py results.h5
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_hdf5(hdf5_path: Path) -> dict:
    """Read W, channel_names, and diagnostic metadata from the step 01 HDF5."""
    with h5py.File(hdf5_path, "r") as f:
        W = f["W"][:]  # (3, N)
        channel_names = json.loads(f.attrs["channel_names"])
        peak_x = f.attrs.get("peak_frequency_hz_x", float("nan"))
        peak_y = f.attrs.get("peak_frequency_hz_y", float("nan"))
        peak_z = f.attrs.get("peak_frequency_hz_z", float("nan"))
        er_x = f.attrs.get("eigenratio_x", float("nan"))
        er_y = f.attrs.get("eigenratio_y", float("nan"))
        er_z = f.attrs.get("eigenratio_z", float("nan"))
    return {
        "W": W,
        "channel_names": channel_names,
        "peak_hz": {"x": peak_x, "y": peak_y, "z": peak_z},
        "eigenratio": {"x": er_x, "y": er_y, "z": er_z},
    }


def build_mapping(hdf5: dict, config: dict) -> list[dict]:
    """Return list of {epics_channel, value, row_label, col_label, mode, ch_name}."""
    W = hdf5["W"]  # (3, N)
    channel_names = hdf5["channel_names"]
    prefix = config["prefix"]
    mat = config["matrix_name"]

    # Map mode string -> W row index
    mode_to_row_idx = {"x": 0, "y": 1, "z": 2}

    # Map channel_suffix -> col config entry
    suffix_to_col = {c["channel_suffix"]: c for c in config["cols"]}

    # Map channel_suffix -> W column index, by matching _IN1 channel names
    # e.g. "Y1:DMD-LESZ_YAW_IN1" -> suffix "LESZ_YAW"
    ch_suffix_to_w_col = {}
    for w_col_idx, ch_name in enumerate(channel_names):
        # Strip prefix "Y1:DMD-" and suffix "_IN1" to get the filter module name
        stripped = ch_name.replace(f"{prefix}-", "").removesuffix("_IN1")
        ch_suffix_to_w_col[stripped] = w_col_idx

    entries = []
    for row_cfg in config["rows"]:
        mode = row_cfg["mode"]
        w_row = mode_to_row_idx[mode]
        sense_row = row_cfg["index"]

        for col_cfg in config["cols"]:
            suffix = col_cfg["channel_suffix"]
            sense_col = col_cfg["index"]
            if suffix not in ch_suffix_to_w_col:
                continue  # this column not present in diagonalization; skip
            w_col = ch_suffix_to_w_col[suffix]
            value = float(W[w_row, w_col])
            epics_ch = f"{prefix}-{mat}_{sense_row}_{sense_col}_GAIN"
            entries.append({
                "epics_channel": epics_ch,
                "value": value,
                "row_label": row_cfg["label"],
                "col_label": col_cfg["label"],
                "mode": mode,
                "ch_name": channel_names[w_col],
            })

    return entries


def print_sparsity_warning(hdf5: dict, config: dict, entries: list[dict]) -> None:
    channel_names = hdf5["channel_names"]
    peaks = hdf5["peak_hz"]
    eratios = hdf5["eigenratio"]
    N = len(channel_names)

    # Determine which cols are NOT being written
    written_col_labels = {e["col_label"] for e in entries}
    all_col_labels = {c["label"] for c in config["cols"]}
    untouched_cols = sorted(all_col_labels - written_col_labels)

    mode_info = {
        "x": (peaks["x"], eratios["x"]),
        "y": (peaks["y"], eratios["y"]),
        "z": (peaks["z"], eratios["z"]),
    }
    # Build per-channel mode assignment: dominant mode is the W row with
    # the largest absolute value for that channel's column.
    W = hdf5["W"]
    mode_to_row_idx = {"x": 0, "y": 1, "z": 2}
    ch_mode: dict[str, str] = {}
    for w_col, ch in enumerate(channel_names):
        dominant_mode = max(mode_to_row_idx, key=lambda m: abs(W[mode_to_row_idx[m], w_col]))
        ch_mode[ch] = dominant_mode

    print()
    print("=" * 70)
    print("  WARNING — SENSE matrix sparsity and diagonalization validity")
    print("=" * 70)
    print(f"  This diagonalization was computed using N={N} sensor channel(s):")
    for ch in channel_names:
        mode = ch_mode.get(ch, "?")
        f0, er = mode_info.get(mode, (float("nan"), float("nan")))
        f0_str = f"peak={f0:.2f} Hz" if not np.isnan(f0) else "peak=unknown"
        er_str = f"eigenratio={er:.1f}" if not np.isnan(er) else ""
        print(f"    {ch}  ({mode}-mode, {f0_str}{', ' + er_str if er_str else ''})")
    print()
    if untouched_cols:
        print(f"  SENSE columns left UNTOUCHED (not in diagonalization):")
        for lbl in untouched_cols:
            print(f"    {lbl}")
        print()
    print("  IMPORTANT: If any of the channels listed above lose signal content")
    print("  after this matrix is loaded — for example because a laser is turned")
    print("  off, a photodetector is blocked, or an optical path changes — the")
    print("  diagonalization WILL be compromised.")
    print()
    print("  The W matrix was computed assuming all N channels carry their")
    print("  expected noise and signal content. The diagonalization exploits")
    print("  noise projections and common-mode rejection that were optimised")
    print("  for this specific set of active sensors. Removing or degrading")
    print("  one channel's information content breaks this optimisation and")
    print("  will corrupt the PARTICLE_X/Y/Z outputs.")
    print()
    print("  If the sensor configuration changes, re-run step 01 with the new")
    print("  set of active channels and re-upload the SENSE matrix.")
    print("=" * 70)
    print()


def caput(channel: str, value: float) -> bool:
    """Write a single EPICS channel via caput. Returns True on success."""
    result = subprocess.run(
        ["caput", channel, str(value)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: caput failed for {channel}: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload step 01 W matrix to Y1:DMD SENSE matrix via EPICS caput."
    )
    parser.add_argument("hdf5_path", help="Path to step_01_sensor_diagonalization_results.h5")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "sense_matrix_config.yml"),
        help="Path to sense_matrix_config.yml (default: same directory as this script)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned caput calls without executing them",
    )
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5_path)
    config_path = Path(args.config)

    if not hdf5_path.exists():
        print(f"ERROR: HDF5 file not found: {hdf5_path}", file=sys.stderr)
        sys.exit(1)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    hdf5 = load_hdf5(hdf5_path)
    entries = build_mapping(hdf5, config)

    if not entries:
        print("ERROR: No SENSE entries could be mapped. Check that channel_names in the", file=sys.stderr)
        print("HDF5 match the channel_suffix values in sense_matrix_config.yml.", file=sys.stderr)
        sys.exit(1)

    print_sparsity_warning(hdf5, config, entries)

    print(f"  {'DRY RUN — ' if args.dry_run else ''}Writing {len(entries)} SENSE element(s):")
    print(f"  {'Channel':<40} {'Value':>14}")
    print(f"  {'-'*40} {'-'*14}")
    for e in entries:
        print(f"  {e['epics_channel']:<40} {e['value']:>14.6f}  ({e['row_label']} <- {e['col_label']})")
    print()

    if args.dry_run:
        print("  Dry run complete — no channels were written.")
        return

    n_ok = 0
    n_fail = 0
    for e in entries:
        ok = caput(e["epics_channel"], e["value"])
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"  Done: {n_ok} written, {n_fail} failed.")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
