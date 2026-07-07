#!/usr/bin/env python3
"""Upload W matrix from step 01 HDF5 to the Y1:DMD SENSE matrix via EPICS caput.

Reads the demodulation matrix W (KxN) produced by dipole_pipeline step 01
and writes the corresponding elements to Y1:DMD-SENSE_{row}_{col}_GAIN.
K is determined from the HDF5 'dofs' attribute (e.g. K=2 for dofs=[x,y]).

Each SENSE element is an rtcds filter module (cdsFiltMuxMatrix). For each
written element the script also:
  - Sets TRAMP (ramp time) before writing the gain
  - Ensures the input switch (SW1, bit 2) and output switch (SW2, bit 10)
    are on. Because SW1/SW2 writes XOR-toggle bits, the current state is
    read first and the toggle is only issued if the bit is currently off.

Only the SENSE columns corresponding to channels actually used in the
diagonalization are written; all other matrix elements are left untouched.

Usage:
    python3 upload_sense_matrix.py <hdf5_path> [--config <yml>] [--tramp <sec>] [--dry-run]

Examples:
    # Preview what would be written (no caput calls made):
    python3 upload_sense_matrix.py results.h5 --dry-run

    # Write to live EPICS channels with default 5 s ramp:
    python3 upload_sense_matrix.py results.h5

    # Write with a 10 s ramp:
    python3 upload_sense_matrix.py results.h5 --tramp 10
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utility as ucoord  # noqa: E402  (shared coordinate system)

# Filter module switch bit masks (CDS cdsFilt convention)
_SW1_INPUT_ON_BIT = 4       # bit 2 of SW1 register: input ON/OFF
_SW2_OUTPUT_ON_BIT = 1024   # bit 10 of SW2 register: output ON/OFF


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_hdf5(hdf5_path: Path) -> dict:
    """Read W, channel_names, dofs, and diagnostic metadata from the step 01 HDF5."""
    with h5py.File(hdf5_path, "r") as f:
        W = f["W"][:]  # (K, N)
        channel_names = json.loads(f.attrs["channel_names"])
        dofs_raw = f.attrs.get("dofs", None)
        if dofs_raw is not None:
            dofs = json.loads(dofs_raw)
        else:
            dofs = ["x", "y", "z"][:W.shape[0]]
        peak_hz = {}
        eigenratio = {}
        for m in dofs:
            peak_hz[m] = f.attrs.get(f"peak_frequency_hz_{m}", float("nan"))
            eigenratio[m] = f.attrs.get(f"eigenratio_{m}", float("nan"))
    return {
        "W": W,
        "channel_names": channel_names,
        "dofs": dofs,
        "peak_hz": peak_hz,
        "eigenratio": eigenratio,
    }


def _row_is_pure_mode(row_cfg: dict) -> bool:
    """True if the row uses ONLY the axis shorthand (mode/axis), no direction spec."""
    direction_keys = {"angle_deg", "elevation_deg", "azimuth_deg", "vector"}
    return not (direction_keys & set(row_cfg.keys()))


def row_direction_vector(row_cfg: dict, dofs: list[str]) -> tuple[np.ndarray, str]:
    """Return (n, descr): the DOF-space unit vector for a SENSE row and a label.

    Accepts the same direction grammar as the ACTS uploader (shared utility):
    axis shorthand (mode/axis: x|y|z), angle_deg (in-plane), elevation_deg+
    azimuth_deg (full sphere), or explicit vector. The returned ``n`` has length
    len(dofs); the row's coefficients are ``n @ W`` (i.e. the linear combination
    of DOF readouts pointing along that direction).
    """
    u3 = ucoord.direction_unit_vector(row_cfg)
    n = ucoord.select_dofs(u3, dofs)
    if "vector" in row_cfg:
        descr = f"vector{np.round(u3, 3).tolist()}"
    elif "elevation_deg" in row_cfg or "azimuth_deg" in row_cfg:
        descr = f"el={row_cfg.get('elevation_deg', 0)},az={row_cfg.get('azimuth_deg', 0)}"
    elif "angle_deg" in row_cfg:
        descr = f"{row_cfg['angle_deg']}deg(xy)"
    else:
        descr = str(row_cfg.get("mode", row_cfg.get("axis")))
    return n, descr


def build_mapping(hdf5: dict, config: dict) -> tuple[list[dict], list[dict]]:
    """Return (entries, skipped_rows).

    Each SENSE row specifies a *direction* in DOF space (see row_direction_vector).
    The element written to SENSE[row, col] is the projection of the demodulation
    matrix onto that direction for the sensor channel feeding ``col``:
        value = sum_k n[k] * W[k, w_col]   ==  (n @ W)[w_col]
    A pure-axis row (mode: x) reduces to W[row_x, w_col] -- so existing configs
    are unchanged.

    entries: list of {base, epics_channel, value, row_label, col_label, ...}
    skipped_rows: rows whose pure-axis mode is absent from hdf5["dofs"]
                  (their SENSE elements retain previous values). An EXPLICIT
                  direction requesting an absent DOF raises instead of skipping.
    """
    W = hdf5["W"]  # (K, N)
    channel_names = hdf5["channel_names"]
    dofs = hdf5["dofs"]
    prefix = config["prefix"]
    mat = config["matrix_name"]
    subspace_tol = float(config.get("subspace_tol", 1e-6))

    # Map channel_suffix -> W column index via _IN1/_DQ channel name stripping
    # e.g. "Y1:DMD-LESZ_YAW_IN1_DQ" -> "LESZ_YAW"
    ch_suffix_to_w_col = {}
    for w_col_idx, ch_name in enumerate(channel_names):
        stripped = ch_name.replace(f"{prefix}-", "").removesuffix("_DQ").removesuffix("_IN1")
        ch_suffix_to_w_col[stripped] = w_col_idx

    entries = []
    skipped_rows = []
    for row_cfg in config["rows"]:
        u3 = ucoord.direction_unit_vector(row_cfg)
        rem = ucoord.out_of_subspace_fraction(u3, dofs)
        if rem > subspace_tol:
            # Direction has content outside the measured DOFs.
            if _row_is_pure_mode(row_cfg):
                # Back-compat: a pure axis (e.g. ZERR mode:z) absent from this
                # diagonalization is simply skipped; its row retains old values.
                skipped_rows.append(row_cfg)
                continue
            raise ValueError(
                f"SENSE row {row_cfg.get('index')} ({row_cfg.get('label')}) requests a "
                f"direction with {rem*100:.1f}% outside the measured DOFs {dofs}; "
                f"cannot realize. Re-run step 01 over the needed DOFs or change the row.")
        n, descr = row_direction_vector(row_cfg, dofs)
        sense_row = row_cfg["index"]

        for col_cfg in config["cols"]:
            suffix = col_cfg["channel_suffix"]
            sense_col = col_cfg["index"]
            base = f"{prefix}-{mat}_{sense_row}_{sense_col}"
            if suffix in ch_suffix_to_w_col:
                w_col = ch_suffix_to_w_col[suffix]
                value = float(n @ W[:, w_col])
                ch_name = channel_names[w_col]
            else:
                w_col = None
                value = 0.0
                ch_name = None
            entries.append({
                "base": base,
                "epics_channel": f"{base}_GAIN",
                "value": value,
                "row_label": row_cfg["label"],
                "col_label": col_cfg["label"],
                "mode": descr,
                "ch_name": ch_name,
                "w_row": None,
                "w_col": w_col,
                "sense_row": sense_row,
                "sense_col": sense_col,
                # input filter-module feeding this SENSE column (only if the column is
                # actually part of the diagonalization). Used to enable the module so the
                # uploaded W is actually realized — a gain-zeroed/switched-out input column
                # silently deletes that W column and corrupts the PARTICLE outputs.
                "input_base": f"{prefix}-{suffix}" if w_col is not None else None,
            })

    return entries, skipped_rows


def _caget_int(channel: str) -> int | None:
    """Read a single EPICS channel and return its value as int, or None on error."""
    result = subprocess.run(
        ["caget", channel],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        return None
    # caget output: "CHANNEL_NAME  value" — take last whitespace-separated token
    parts = result.stdout.strip().split()
    try:
        return int(float(parts[-1])) if parts else None
    except (ValueError, TypeError):
        return None


def _caput(channel: str, value) -> bool:
    """Write a single EPICS channel via caput. Returns True on success."""
    result = subprocess.run(
        ["caput", channel, str(value)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: caput failed for {channel}: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def read_switch_states(entries: list[dict]) -> dict[str, dict]:
    """Batch-read SW1R and SW2R for all entry bases. Returns {base: {sw1r, sw2r}}."""
    channels = []
    for e in entries:
        channels.append(f"{e['base']}_SW1R")
        channels.append(f"{e['base']}_SW2R")

    result = subprocess.run(
        ["caget"] + channels,
        capture_output=True, text=True, timeout=30
    )

    raw = {}
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                try:
                    raw[parts[0]] = int(float(parts[1]))
                except (ValueError, TypeError):
                    raw[parts[0]] = None

    states = {}
    for e in entries:
        base = e["base"]
        sw1r = raw.get(f"{base}_SW1R")
        sw2r = raw.get(f"{base}_SW2R")
        states[base] = {
            "sw1r": sw1r,
            "sw2r": sw2r,
            "input_on": (sw1r is not None) and bool(sw1r & _SW1_INPUT_ON_BIT),
            "output_on": (sw2r is not None) and bool(sw2r & _SW2_OUTPUT_ON_BIT),
        }
    return states


def write_entry(entry: dict, sw_state: dict, tramp: float, dry_run: bool) -> bool:
    """Set TRAMP, write GAIN, then ensure input and output switches are on.

    Returns True if all operations succeeded (or dry_run=True).
    """
    base = entry["base"]
    ok = True

    if dry_run:
        return True

    # 1. Set ramp time
    ok = _caput(f"{base}_TRAMP", tramp) and ok

    # 2. Write gain
    ok = _caput(f"{base}_GAIN", entry["value"]) and ok

    # 3. Enable input switch if currently off (XOR-toggle, so only write if off)
    if not sw_state["input_on"]:
        ok = _caput(f"{base}_SW1", _SW1_INPUT_ON_BIT) and ok

    # 4. Enable output switch if currently off
    if not sw_state["output_on"]:
        ok = _caput(f"{base}_SW2", _SW2_OUTPUT_ON_BIT) and ok

    return ok


def input_modules_for_entries(entries: list[dict]) -> list[str]:
    """Unique input filter-module bases feeding the SENSE columns being written.

    Only columns that are part of the diagonalization (w_col is not None) have an
    ``input_base``; untouched columns are excluded.
    """
    seen = []
    for e in entries:
        ib = e.get("input_base")
        if ib and ib not in seen:
            seen.append(ib)
    return seen


def read_input_module_states(input_bases: list[str]) -> dict[str, dict]:
    """Batch-read GAIN, SW1R, SW2R for each input filter module.

    Returns {base: {gain, input_on, output_on}}. Values are None if unreadable.
    """
    channels = []
    for b in input_bases:
        channels += [f"{b}_GAIN", f"{b}_SW1R", f"{b}_SW2R"]
    raw = {}
    if channels:
        result = subprocess.run(["caget"] + channels,
                                capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = line.split(None, 1)
                if len(parts) == 2:
                    try:
                        raw[parts[0]] = float(parts[1])
                    except (ValueError, TypeError):
                        raw[parts[0]] = None
    states = {}
    for b in input_bases:
        gain = raw.get(f"{b}_GAIN")
        sw1r = raw.get(f"{b}_SW1R")
        sw2r = raw.get(f"{b}_SW2R")
        states[b] = {
            "gain": gain,
            "input_on": (sw1r is not None) and bool(int(sw1r) & _SW1_INPUT_ON_BIT),
            "output_on": (sw2r is not None) and bool(int(sw2r) & _SW2_OUTPUT_ON_BIT),
        }
    return states


def enable_input_module(base: str, state: dict, tramp: float, dry_run: bool) -> bool:
    """Force an input filter module ON: GAIN=1 and input/output switches enabled.

    The SENSE matrix assumes each input column carries the unity-gain filter output the
    W diagonalization was computed against. A zeroed gain or open switch deletes that W
    column. We always set GAIN=1 (per design decision) and enable both switches if off.
    """
    if dry_run:
        return True
    ok = True
    ok = _caput(f"{base}_TRAMP", tramp) and ok
    ok = _caput(f"{base}_GAIN", 1.0) and ok
    if not state["input_on"]:
        ok = _caput(f"{base}_SW1", _SW1_INPUT_ON_BIT) and ok
    if not state["output_on"]:
        ok = _caput(f"{base}_SW2", _SW2_OUTPUT_ON_BIT) and ok
    return ok


def write_state_file(hdf5_path: Path, hdf5: dict, entries: list[dict], tramp: float) -> Path:
    """Record what's currently live on the SENSE matrix (gitignored, script-owned).

    Written only after a fully successful (non-dry-run) upload, so the file always
    reflects EPICS reality rather than an attempted-but-partially-failed write.
    """
    state_path = Path(__file__).parent / "sense_matrix_state.json"
    rows_written = sorted({e["row_label"] for e in entries})
    state = {
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hdf5_path": str(hdf5_path.resolve()),
        "dofs": hdf5["dofs"],
        "rows_written": rows_written,
        "tramp_s": tramp,
        "peak_hz": hdf5["peak_hz"],
        "eigenratio": hdf5["eigenratio"],
    }
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
        f.write("\n")
    return state_path


def print_sparsity_warning(hdf5: dict, config: dict, entries: list[dict],
                           skipped_rows: list[dict]) -> None:
    channel_names = hdf5["channel_names"]
    dofs = hdf5["dofs"]
    peaks = hdf5["peak_hz"]
    eratios = hdf5["eigenratio"]
    N = len(channel_names)

    written_col_labels = {e["col_label"] for e in entries}
    all_col_labels = {c["label"] for c in config["cols"]}
    untouched_cols = sorted(all_col_labels - written_col_labels)

    W = hdf5["W"]
    mode_to_w_row = {m: i for i, m in enumerate(dofs)}
    ch_mode: dict[str, str] = {}
    for w_col, ch in enumerate(channel_names):
        dominant_mode = max(mode_to_w_row, key=lambda m: abs(W[mode_to_w_row[m], w_col]))
        ch_mode[ch] = dominant_mode

    print()
    print("=" * 70)
    print("  WARNING — SENSE matrix sparsity and diagonalization validity")
    print("=" * 70)
    print(f"  Active DOFs: {dofs} (K={len(dofs)})")
    print(f"  This diagonalization was computed using N={N} sensor channel(s):")
    for ch in channel_names:
        mode = ch_mode.get(ch, "?")
        f0 = peaks.get(mode, float("nan"))
        er = eratios.get(mode, float("nan"))
        f0_str = f"peak={f0:.2f} Hz" if not np.isnan(f0) else "peak=unknown"
        er_str = f"eigenratio={er:.1f}" if not np.isnan(er) else ""
        print(f"    {ch}  ({mode}-mode, {f0_str}{', ' + er_str if er_str else ''})")
    print()
    if skipped_rows:
        print("  INACTIVE mode rows (not in this diagonalization):")
        for row_cfg in skipped_rows:
            axis = row_cfg.get("mode", row_cfg.get("axis", "?"))
            print(f"    {row_cfg['label']} ({axis})")
        print()
        print("  These rows RETAIN their previous SENSE values. If feedback for")
        print("  these modes was active, it is running on STALE coefficients.")
        print("  Disable it or re-run step 01 with all DOFs included.")
        print()
    if untouched_cols:
        print("  SENSE columns left UNTOUCHED (not in diagonalization):")
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
        "--tramp",
        type=float,
        default=5.0,
        metavar="SEC",
        help="Ramp time in seconds applied to each element before writing (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without executing any caput calls",
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
    entries, skipped_rows = build_mapping(hdf5, config)

    if not entries:
        print("ERROR: No SENSE entries could be mapped. Check that channel_names in the", file=sys.stderr)
        print("HDF5 match the channel_suffix values in sense_matrix_config.yml.", file=sys.stderr)
        sys.exit(1)

    print_sparsity_warning(hdf5, config, entries, skipped_rows)

    # Read current switch states (batch caget)
    if not args.dry_run:
        print("  Reading current switch states ...")
        sw_states = read_switch_states(entries)
    else:
        # In dry-run, populate with None so we show "unknown" for switch status
        sw_states = {e["base"]: {"input_on": None, "output_on": None} for e in entries}

    prefix = f"DRY RUN — " if args.dry_run else ""
    print(f"  {prefix}Writing {len(entries)} SENSE element(s)  (TRAMP={args.tramp:.1f} s):")
    print(f"  {'Channel':<40} {'Value':>14}  {'SW1(in)':>10}  {'SW2(out)':>10}  Mapping")
    print(f"  {'-'*40} {'-'*14}  {'-'*10}  {'-'*10}  {'-'*32}")
    for e in entries:
        sw = sw_states[e["base"]]
        if args.dry_run:
            sw1_str = "dry-run"
            sw2_str = "dry-run"
        else:
            sw1_str = "already on" if sw["input_on"] else "will enable"
            sw2_str = "already on" if sw["output_on"] else "will enable"
        if e["w_col"] is not None:
            mapping = (f"(n·W)[{e['w_col']}] ({e['row_label']}[{e['mode']}]<-{e['col_label']})"
                       f" → SENSE[{e['sense_row']},{e['sense_col']}]")
        else:
            mapping = (f"({e['row_label']}<-{e['col_label']} not in diag)"
                       f" → SENSE[{e['sense_row']},{e['sense_col']}] = 0")
        print(
            f"  {e['epics_channel']:<40} {e['value']:>14.6f}"
            f"  {sw1_str:>10}  {sw2_str:>10}  {mapping}"
        )
    print()

    # Input filter modules feeding the written SENSE columns: these must be ON (GAIN=1,
    # switches enabled) or the uploaded W column is silently deleted from the live map.
    input_bases = input_modules_for_entries(entries)
    if args.dry_run:
        in_states = {b: {"gain": None, "input_on": None, "output_on": None}
                     for b in input_bases}
    else:
        in_states = read_input_module_states(input_bases)

    if input_bases:
        print(f"  {prefix}Enabling {len(input_bases)} input filter module(s) "
              f"(force GAIN=1, switches on):")
        print(f"  {'Module':<28} {'cur.GAIN':>9}  {'SW1(in)':>11}  {'SW2(out)':>11}  Action")
        print(f"  {'-'*28} {'-'*9}  {'-'*11}  {'-'*11}  {'-'*24}")
        for b in input_bases:
            s = in_states[b]
            if args.dry_run:
                g_str, sw1_str, sw2_str = "dry-run", "dry-run", "dry-run"
            else:
                g_str = "?" if s["gain"] is None else f"{s['gain']:.3g}"
                sw1_str = "already on" if s["input_on"] else "will enable"
                sw2_str = "already on" if s["output_on"] else "will enable"
            print(f"  {b:<28} {g_str:>9}  {sw1_str:>11}  {sw2_str:>11}  set GAIN=1, ensure ON")
        print()

    if args.dry_run:
        print("  Dry run complete — no channels were written.")
        return

    n_ok = 0
    n_fail = 0
    for e in entries:
        ok = write_entry(e, sw_states[e["base"]], args.tramp, dry_run=False)
        status = "ok" if ok else "FAILED"
        print(f"  {e['epics_channel']:<40}  {status}")
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    # Enable the input filter modules feeding the written columns.
    for b in input_bases:
        ok = enable_input_module(b, in_states[b], args.tramp, dry_run=False)
        status = "ok" if ok else "FAILED"
        print(f"  {b + '_GAIN/SW':<40}  {status}")
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\n  Done: {n_ok} written, {n_fail} failed.")
    if n_fail:
        sys.exit(1)

    state_path = write_state_file(hdf5_path, hdf5, entries, args.tramp)
    print(f"  State recorded: {state_path}")


if __name__ == "__main__":
    main()
