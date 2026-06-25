#!/usr/bin/env python3
"""Invert measured electrode->particle actuator gains and write the ACTS matrix.

Reads one or more ``actuator_gain_results.h5`` files produced by
``measure_actuator_gain.py`` (each driving one or more electrodes E1..E4 and
measuring the diagonalized PARTICLE_X/Y[/Z] response), assembles the forward
actuation matrix ``A`` (DOF x electrode), inverts it, and writes per-electrode
coefficients into the Y1:DMD ACTS matrix so that each *configured ACTS column*
drives the electric field in a chosen direction in the x-y (or x-y-z) plane.

The physics, the five operating modes, and every YAML knob are documented in
``upload_actuation_matrix_config.yml`` and ``README.md``. Short version:

  * Each result file stores a complex gain G[dof, elec] = (counts->force/field
    coupling) x chi(omega0)_dof, where the plant in measure_actuator_gain is
    PEAK-normalized (|H(f0)|=1) so the fitted gain carries the full on-resonance
    susceptibility chi(omega0)_dof = Q_dof / (m * omega0_dof^2).
  * We reduce each complex G to a real number by SIGNED MAGNITUDE
    |G|*sign(cos(phase)) -- exact when the residual phase is ~0/180 deg (it is).
  * FIELD NORMALIZATION removes chi(omega0) to recover the pressure- and
    frequency-independent coupling. Gas-dominated damping => the damping rate
    gamma = f0/Q is common across modes WITHIN a single file (but drifts between
    files with pressure). So per file we pool the per-mode gamma_d = f0_d/Q_d
    (residual-weighted, with uncertainty) into one gamma_file, and scale each
    column by s_d = f0_d * gamma_file (= 1/chi(omega0) up to a constant). Each
    electrode's column uses ITS OWN file's gamma, which corrects cross-run
    pressure drift in the relative electrode strengths.
  * A_field = diag-per-(dof,file) applied to A; A+ = pinv(A_field). For a column
    whose field direction is the unit vector u (in DOF space), the electrode
    counts are v = gain * (A+ @ u). These go to ACTS_{1..4}_{col}_GAIN.

Only the electrode rows (1..4) of *coupled* columns are written. The laser
rows (LASER_CTLX/Z) and the unused DAC rows are never touched.

Usage:
    python3 upload_actuation_matrix.py [--config <yml>] [--dry-run]
                                       [--enable-switches] [--clear-uncoupled]

By default switches are NOT toggled (GAIN-only + warn); pass --enable-switches
to auto-enable input+output switches on written elements (signal goes live on
the trap electrodes). Always start with --dry-run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import yaml

# Reuse the EPICS switch helpers / bit conventions from the SENSE uploader.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import upload_sense_matrix as usm  # noqa: E402  (caput/switch helpers, bit masks)
import utility as ucoord  # noqa: E402  (shared coordinate system)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class FileGains:
    """Per-file extracted gains + plant parameters for the inversion."""
    path: Path
    electrodes: list[str]              # electrode labels in this file, ordered
    dof_order: list[str]               # DOF labels in this file, ordered
    gain: np.ndarray                   # complex (n_dof, n_elec)
    f0: dict[str, float]               # fitted resonance per dof
    Q: dict[str, float]                # fitted Q per dof
    residual: dict[str, float]         # fit residual_norm per dof (weighting)
    coherence: dict[str, np.ndarray]   # per-dof per-tone coherence arrays
    gamma: float = float("nan")        # pooled damping rate (Hz), this file
    gamma_sigma: float = float("nan")  # uncertainty on gamma (Hz)


@dataclass
class CellEntry:
    """One (dof, electrode) cell of the forward matrix, with provenance."""
    dof: str
    electrode: str
    g_complex: complex
    g_real: float                      # signed-magnitude reduction
    phase_deg: float
    file_index: int
    coherence: float                   # representative (max) coherence for this cell


@dataclass
class ColumnPlan:
    """Resolved write plan for one ACTS column."""
    index: int
    label: str
    coupled: bool
    clear: bool
    gain: float
    direction: np.ndarray | None       # 3-D unit vector, or None if uncoupled
    realized_field: np.ndarray | None  # (n_dof,) field actually produced
    electrode_values: dict[str, float] = field(default_factory=dict)  # elec -> count
    electrode_sigma: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Pure math (no EPICS / no I/O) -- unit tested
# --------------------------------------------------------------------------- #
def signed_magnitude(z: complex) -> float:
    """Reduce a complex gain to a real number: |z| * sign(cos(phase(z))).

    Exact when the residual phase is ~0 or ~180 deg. Phase near +-90 deg makes
    the reduction lossy (magnitude survives, sign ill-defined) -- callers should
    warn in that regime (see lossy_phase()).
    """
    mag = abs(z)
    c = math.cos(math.atan2(z.imag, z.real))
    s = 1.0 if c >= 0 else -1.0
    return mag * s


def lossy_phase(z: complex, tol_deg: float = 45.0) -> bool:
    """True if phase(z) is closer to +-90 deg than to 0/180 by more than tol.

    i.e. the signed-magnitude reduction discards a significant real part.
    """
    ph = abs(math.degrees(math.atan2(z.imag, z.real)))  # 0..180
    dist_to_axis = min(ph, abs(180.0 - ph))             # distance to 0 or 180
    return dist_to_axis > tol_deg


def pool_gamma(f0: dict, Q: dict, residual: dict,
               dofs: list[str]) -> tuple[float, float, dict]:
    """Pool per-mode gamma_d = f0_d/Q_d into one gamma per file, with uncertainty.

    Weights w_d ~ 1/residual_d^2 (fit quality). Returns
    (gamma_pool, gamma_sigma, per_mode_gamma). The reported sigma is the
    conservative max of the internal (weight-based) and external (scatter-based)
    standard errors -- honest given typically only 2 modes and that residual_norm
    is a fit-quality proxy, not a rigorous variance of gamma.
    """
    per_mode = {d: f0[d] / Q[d] for d in dofs if Q.get(d, 0) > 0 and not math.isnan(Q[d])}
    if not per_mode:
        return float("nan"), float("nan"), {}
    modes = list(per_mode.keys())
    g = np.array([per_mode[d] for d in modes])
    res = np.array([residual.get(d, np.nan) for d in modes], dtype=float)
    # Weights from residuals; fall back to equal weights if residuals missing.
    if np.all(np.isfinite(res)) and np.all(res > 0):
        w = 1.0 / res ** 2
    else:
        w = np.ones_like(g)
    W = w.sum()
    gbar = float((w * g).sum() / W)
    if len(modes) > 1:
        var = float((w * (g - gbar) ** 2).sum() / W)
        se_external = math.sqrt(var / (len(modes) - 1))
        se_internal = math.sqrt(1.0 / W)
        sigma = max(se_external, se_internal)
    else:
        sigma = 0.0
    return gbar, sigma, per_mode


def field_scale_factors(fg: FileGains, mode: str) -> dict[str, float]:
    """Per-DOF susceptibility-removal factor s_d for one file.

    mode:
      "common_gamma" (default): s_d = f0_d * gamma_file  (= 1/chi(omega0), with
            the per-file gas-dominated common damping rate gamma_file)
      "per_mode_q":             s_d = f0_d^2 / Q_d        (uses each mode's Q)
      "none":                   s_d = 1
    """
    if mode == "none":
        return {d: 1.0 for d in fg.dof_order}
    if mode == "per_mode_q":
        return {d: fg.f0[d] ** 2 / fg.Q[d] for d in fg.dof_order}
    if mode == "common_gamma":
        return {d: fg.f0[d] * fg.gamma for d in fg.dof_order}
    raise ValueError(f"unknown field_normalize mode {mode!r}")


def anchor_scale(A_field: np.ndarray, functional: str = "frobenius",
                 ref_col: int = 0) -> float:
    """Degree-1 functional N(A_field) whose division fixes the magnitude unit.

    field_normalize removes the per-DOF susceptibility chi(omega0), leaving
    A_field = c * B, where B[d,e] is the field-per-count matrix (electrode
    geometry only) and c = cal*q/(4pi^2 m) is a SINGLE global scalar that changes
    particle-to-particle (but NOT with pressure/Q -- A ~ chi and s_d ~ 1/chi
    cancel). Any degree-1 functional obeys N(A_field) = c * N(B), so
    A_field / N(A_field) = B / N(B) depends ONLY on the electrodes. Hence
    gain=1 produces a reproducible field whenever the electrodes are unchanged.

      "frobenius"        -> ||A_field||_F   (default; RMS over all cells, most
                            robust to single-cell measurement noise)
      "sigma_max"        -> largest singular value (conditioning-aware)
      "reference_column" -> ||A_field[:, ref_col]||_2  (one electrode's column)

    Returns 1.0 (no-op) for a degenerate (~zero) matrix so callers never divide
    by zero.
    """
    if functional == "frobenius":
        N = float(np.linalg.norm(A_field))
    elif functional == "sigma_max":
        N = float(np.linalg.svd(A_field, compute_uv=False)[0])
    elif functional == "reference_column":
        if not (0 <= ref_col < A_field.shape[1]):
            raise ValueError(f"reference_column index {ref_col} out of range "
                             f"[0, {A_field.shape[1]})")
        N = float(np.linalg.norm(A_field[:, ref_col]))
    else:
        raise ValueError(f"unknown anchor functional {functional!r}")
    if not np.isfinite(N) or N < 1e-300:
        return 1.0
    return N


def gain_rel_error(coherence: float, n_avg: int) -> float:
    """Relative 1-sigma error on a transfer-function magnitude (H1 estimator).

    sigma_|H|/|H| ~ sqrt((1 - coh) / (2 * coh * n_avg)). Standard coherent-
    averaging result; used to attach an uncertainty to each forward-matrix cell.
    """
    c = min(max(float(coherence), 1e-6), 0.999999)
    n = max(int(n_avg), 1)
    return math.sqrt((1.0 - c) / (2.0 * c * n))


def build_forward_matrix(file_gains: list[FileGains], dof_order: list[str],
                         elec_order: list[str], duplicate: str = "error"
                         ) -> tuple[np.ndarray, dict]:
    """Assemble the real forward matrix A (n_dof x n_elec) from N files.

    Each cell A[d, e] is keyed by (dof, electrode), not by file -- so any layout
    (4 single-electrode files, one 4-electrode file, 2x2, one-DOF-at-a-time)
    collapses to the same A. ``duplicate`` is "error" or "average".

    Returns (A, cells) where cells maps (dof, elec) -> CellEntry.
    """
    cells: dict[tuple[str, str], list[CellEntry]] = {}
    for fi, fg in enumerate(file_gains):
        for di, d in enumerate(fg.dof_order):
            if d not in dof_order:
                continue
            for ei, e in enumerate(fg.electrodes):
                if e not in elec_order:
                    continue
                z = complex(fg.gain[di, ei])
                coh = fg.coherence.get(d)
                coh_val = float(np.nanmax(coh)) if coh is not None and len(coh) else float("nan")
                entry = CellEntry(dof=d, electrode=e, g_complex=z,
                                  g_real=signed_magnitude(z),
                                  phase_deg=math.degrees(math.atan2(z.imag, z.real)),
                                  file_index=fi, coherence=coh_val)
                cells.setdefault((d, e), []).append(entry)

    A = np.full((len(dof_order), len(elec_order)), np.nan)
    resolved: dict[tuple[str, str], CellEntry] = {}
    missing = []
    for di, d in enumerate(dof_order):
        for ei, e in enumerate(elec_order):
            lst = cells.get((d, e))
            if not lst:
                missing.append((d, e))
                continue
            if len(lst) > 1:
                if duplicate == "error":
                    raise ValueError(
                        f"duplicate measurements for (dof={d}, electrode={e}) in "
                        f"{len(lst)} files; set duplicate: average to combine them")
                elif duplicate == "average":
                    vals = [c.g_real for c in lst]
                    merged = lst[0]
                    merged.g_real = float(np.mean(vals))
                    resolved[(d, e)] = merged
                else:
                    raise ValueError(f"unknown duplicate policy {duplicate!r}")
            else:
                resolved[(d, e)] = lst[0]
            A[di, ei] = resolved[(d, e)].g_real
    if missing:
        raise ValueError(
            "forward matrix is missing measurements for: "
            + ", ".join(f"(dof={d}, electrode={e})" for d, e in missing)
            + ". Provide a result file covering each cell, or remove the electrode/DOF.")
    return A, resolved


def apply_field_normalization(A: np.ndarray, dof_order: list[str], elec_order: list[str],
                              cells: dict, file_gains: list[FileGains], mode: str
                              ) -> tuple[np.ndarray, dict]:
    """Return (A_field, s_by_dof_file) with s_d applied per (dof, electrode's file)."""
    A_field = A.copy()
    s_lookup: dict = {}
    # Precompute per-file scale factors.
    per_file = [field_scale_factors(fg, mode) for fg in file_gains]
    for di, d in enumerate(dof_order):
        for ei, e in enumerate(elec_order):
            fi = cells[(d, e)].file_index
            s = per_file[fi].get(d, 1.0)
            A_field[di, ei] = A[di, ei] * s
            s_lookup[(d, e)] = s
    return A_field, s_lookup


def column_electrode_values(A_pinv: np.ndarray, u_hat: np.ndarray, gain: float
                            ) -> np.ndarray:
    """Min-norm electrode counts producing field direction u_hat at scale ``gain``."""
    return gain * (A_pinv @ u_hat)


def afield_abs_sigma(A_field: np.ndarray, dof_order: list[str], elec_order: list[str],
                     cells: dict, file_gains: list[FileGains], cell_sigma: dict,
                     mode: str) -> np.ndarray:
    """Absolute 1-sigma on each A_field element from gain + gamma uncertainties.

    Relative error of A_field[d,e] = quadrature of:
      * the gain magnitude error (coherence-derived, cell_sigma), and
      * the scale-factor error: for common_gamma, s_d = f0_d * gamma_file so the
        relative error is sigma_gamma/gamma (f0 is comparatively well determined).
    """
    sig = np.zeros_like(A_field)
    for di, d in enumerate(dof_order):
        for ei, e in enumerate(elec_order):
            rel_g = cell_sigma.get((d, e), 0.0)
            rel_s = 0.0
            if mode == "common_gamma":
                fg = file_gains[cells[(d, e)].file_index]
                if np.isfinite(fg.gamma) and fg.gamma > 0 and np.isfinite(fg.gamma_sigma):
                    rel_s = fg.gamma_sigma / fg.gamma
            rel = math.sqrt(rel_g ** 2 + rel_s ** 2)
            sig[di, ei] = abs(A_field[di, ei]) * rel
    return sig


def propagate_column_sigma(A_field: np.ndarray, A_sigma: np.ndarray, u_hat: np.ndarray,
                           gain: float, n_mc: int = 400, seed: int = 0) -> np.ndarray:
    """Monte-Carlo 1-sigma on electrode counts v = gain*pinv(A_field)@u_hat.

    pinv is nonlinear in A_field, so we perturb A_field by its per-element sigma
    and take the std of the resulting electrode-count vectors. Deterministic
    (fixed seed) for reproducible reporting.
    """
    rng = np.random.default_rng(seed)
    samples = np.empty((n_mc, A_field.shape[1]))
    for k in range(n_mc):
        pert = A_field + rng.standard_normal(A_field.shape) * A_sigma
        try:
            samples[k] = gain * (np.linalg.pinv(pert) @ u_hat)
        except np.linalg.LinAlgError:
            samples[k] = np.nan
    return np.nanstd(samples, axis=0)


# --------------------------------------------------------------------------- #
# I/O: read result files
# --------------------------------------------------------------------------- #
def _decode_json_attr(attr):
    """h5py may hand back a numpy array of chars for a JSON string attr; coerce."""
    if isinstance(attr, (bytes, bytearray)):
        attr = attr.decode()
    if isinstance(attr, np.ndarray):
        attr = "".join(c.decode() if isinstance(c, bytes) else str(c) for c in attr.ravel())
    return json.loads(attr)


def load_result_file(path: Path) -> FileGains:
    with h5py.File(path, "r") as f:
        gr = f["gain_matrix_real"][:]
        gi = f["gain_matrix_imag"][:]
        gain = gr + 1j * gi
        dof_order = _decode_json_attr(f.attrs["dof_order"])
        electrodes = _decode_json_attr(f.attrs["electrodes"])
        f0, Q, residual, coherence = {}, {}, {}, {}
        for d in dof_order:
            f0[d] = float(f.attrs.get(f"peak_frequency_hz_{d}", float("nan")))
            Q[d] = float(f.attrs.get(f"Q_{d}", float("nan")))
            residual[d] = float(f.attrs.get(f"residual_norm_{d}", float("nan")))
            ckey = f"coherence_{d}"
            coherence[d] = f[ckey][:] if ckey in f else np.array([])
    return FileGains(path=path, electrodes=list(electrodes), dof_order=list(dof_order),
                     gain=gain, f0=f0, Q=Q, residual=residual, coherence=coherence)


def resolve_path(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p))))


# --------------------------------------------------------------------------- #
# Planning (pure given loaded files) -- the write-planner
# --------------------------------------------------------------------------- #
def plan_columns(cfg: dict, A_pinv: np.ndarray, dof_order: list[str],
                 elec_order: list[str], cell_sigma: dict,
                 strict_subspace: bool, a_field: np.ndarray | None = None,
                 a_sigma: np.ndarray | None = None) -> list[ColumnPlan]:
    """Build the per-column write plan (no EPICS calls)."""
    plans: list[ColumnPlan] = []
    for col in cfg["columns"]:
        idx = int(col["index"])
        label = col.get("label", f"col{idx}")
        coupled = bool(col.get("coupled", False))
        clear = bool(col.get("clear", False))
        gain = float(col.get("gain", 1.0))
        plan = ColumnPlan(index=idx, label=label, coupled=coupled, clear=clear,
                          gain=gain, direction=None, realized_field=None)
        if not coupled:
            plans.append(plan)
            continue
        u3 = ucoord.direction_unit_vector(col)
        plan.direction = u3
        rem = ucoord.out_of_subspace_fraction(u3, dof_order)
        if rem > 1e-6:
            msg = (f"direction has {rem*100:.1f}% outside measured DOFs {dof_order}; "
                   f"projecting onto the measured subspace")
            if strict_subspace:
                raise ValueError(f"column {label}: {msg} (strict mode)")
            plan.notes.append(msg)
        u = ucoord.select_dofs(u3, dof_order)
        v = column_electrode_values(A_pinv, u, gain)
        for ei, e in enumerate(elec_order):
            plan.electrode_values[e] = float(v[ei])
        # Uncertainty on the electrode counts (filled if A_sigma provided).
        if a_field is not None and a_sigma is not None:
            vs = propagate_column_sigma(a_field, a_sigma, u, gain)
            for ei, e in enumerate(elec_order):
                plan.electrode_sigma[e] = float(vs[ei])
            plan.realized_field = a_field @ v
        plans.append(plan)
    return plans


# --------------------------------------------------------------------------- #
# Output / reporting
# --------------------------------------------------------------------------- #
def print_diagnostics(file_gains, A, A_field, A_sigma, A_pinv, dof_order, elec_order,
                      s_lookup, cells, cfg, plans, res):
    print("=" * 74)
    print("  ACTS actuation-matrix inversion")
    print("=" * 74)
    print(f"  DOFs (rows):       {dof_order}")
    print(f"  Electrodes (cols): {elec_order}")
    print(f"  field_normalize:   {cfg.get('field_normalize', 'common_gamma')}")
    amode = res["anchor_mode"]
    if amode == "none":
        print("  field_anchor:      none (gain=1 NOT reproducible across particles)")
    else:
        print(f"  field_anchor:      {amode} ({res['anchor_functional']})  "
              f"N(A_field)={res['anchor_N']:.4g}  P={res['anchor_P']:g}  "
              f"-> scale x{res['anchor']:.4g}")
        print("  unit note:         gain=1 -> reproducible field of magnitude "
              "P/||B||_F per unit direction")
        print("                     (P=1 => arbitrary-but-fixed electrode unit; "
              "set P=||B||_F for 1 V/m)")
    print()
    print("  Source files (fitted plant + pooled damping rate):")
    spread_warn = float(cfg.get("gamma_spread_warn", 0.20))
    for fg in file_gains:
        gmodes = {d: fg.f0[d] / fg.Q[d] for d in fg.dof_order if fg.Q.get(d, 0) > 0}
        spread = 0.0
        if len(gmodes) > 1:
            gv = np.array(list(gmodes.values()))
            spread = float((gv.max() - gv.min()) / np.mean(gv))
        flag = "  <-- gamma spread WARNING" if spread > spread_warn else ""
        print(f"    {fg.path.parent.name}/{fg.path.name}")
        print(f"      electrodes={fg.electrodes}  f0={ {d: round(fg.f0[d],2) for d in fg.dof_order} }"
              f"  Q={ {d: round(fg.Q[d],1) for d in fg.dof_order} }")
        per = "  ".join(f"gamma_{d}={gmodes[d]:.3f}" for d in gmodes)
        print(f"      {per}  ->  gamma_file={fg.gamma:.3f} +/- {fg.gamma_sigma:.3f} Hz"
              f"  (spread {spread*100:.0f}%){flag}")
        for d in fg.dof_order:
            if d == "x" and not (40.0 <= fg.f0[d] <= 42.0):
                print(f"      !! fitted x f0={fg.f0[d]:.2f} Hz is NOT ~41 Hz -- check the fit/data")
    print()
    np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
    print("  Forward matrix A (signed-magnitude, rows=DOF, cols=electrode):")
    print("   ", str(A).replace("\n", "\n    "))
    print("  Field-normalized A_field:")
    print("   ", str(A_field).replace("\n", "\n    "))
    print("  A_field 1-sigma (gain coherence + gamma uncertainty):")
    print("   ", str(A_sigma).replace("\n", "\n    "))
    print(f"  cond(A_field) = {np.linalg.cond(A_field):.3f}")
    if np.linalg.cond(A_field) > 50:
        print("    !! large condition number -- inversion is ill-conditioned")
    print()


def print_column_plan(plans, dof_order, A_field, elec_order):
    print("  Per-column write plan (electrode rows only):")
    for p in plans:
        if not p.coupled:
            action = "CLEAR (zero E-rows)" if p.clear else "leave untouched"
            print(f"    col {p.index} {p.label:<6} uncoupled -> {action}")
            continue
        # realized field = A_field @ v
        v = np.array([p.electrode_values[e] for e in elec_order])
        realized = A_field @ v
        u = ucoord.select_dofs(p.direction, dof_order)
        print(f"    col {p.index} {p.label:<6} coupled  dir(xyz)={np.round(p.direction,3)} "
              f"gain={p.gain:g}")
        counts = "  ".join(
            f"{e}={p.electrode_values[e]:+.4f}+/-{p.electrode_sigma.get(e, 0.0):.4f}"
            for e in elec_order)
        print(f"        electrode counts: {counts}")
        print(f"        target field {np.round(u,3)} -> realized {np.round(realized,3)}")
        for note in p.notes:
            print(f"        note: {note}")
    print()


# --------------------------------------------------------------------------- #
# EPICS write path
# --------------------------------------------------------------------------- #
def acts_base(prefix: str, row: int, col: int) -> str:
    return f"{prefix}-ACTS_{row}_{col}"


def write_plans(plans, cfg, elec_order, tramp, dry_run, enable_switches,
                clear_uncoupled):
    prefix = cfg["prefix"]
    elec_row = {e: int(r) for e, r in cfg["electrode_row"].items()}

    # Collect the bases we will write so we can batch-read switch states.
    write_bases = []
    for p in plans:
        if p.coupled or (p.clear or clear_uncoupled):
            for e in elec_order:
                write_bases.append(acts_base(prefix, elec_row[e], p.index))

    sw_states = {}
    if not dry_run:
        pvs = []
        for b in write_bases:
            pvs += [f"{b}_SW1R", f"{b}_SW2R"]
        raw = subprocess.run(["caget"] + pvs, capture_output=True, text=True,
                             timeout=30).stdout if pvs else ""
        parsed = {}
        for line in raw.strip().splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                try:
                    parsed[parts[0]] = int(float(parts[1]))
                except ValueError:
                    parsed[parts[0]] = None
        for b in write_bases:
            s1 = parsed.get(f"{b}_SW1R")
            s2 = parsed.get(f"{b}_SW2R")
            sw_states[b] = {
                "input_on": (s1 is not None) and bool(s1 & usm._SW1_INPUT_ON_BIT),
                "output_on": (s2 is not None) and bool(s2 & usm._SW2_OUTPUT_ON_BIT),
            }

    n_ok = n_fail = 0
    for p in plans:
        if not p.coupled:
            if not (p.clear or clear_uncoupled):
                continue
            # Zero electrode rows of this uncoupled column.
            for e in elec_order:
                b = acts_base(prefix, elec_row[e], p.index)
                if dry_run:
                    print(f"    DRY: {b}_GAIN <- 0.0  (clear uncoupled)")
                    continue
                ok = usm._caput(f"{b}_TRAMP", tramp) and usm._caput(f"{b}_GAIN", 0.0)
                n_ok += ok
                n_fail += (not ok)
            continue
        for e in elec_order:
            b = acts_base(prefix, elec_row[e], p.index)
            val = p.electrode_values[e]
            if dry_run:
                print(f"    DRY: {b}_GAIN <- {val:+.6f}")
                continue
            ok = usm._caput(f"{b}_TRAMP", tramp)
            ok = usm._caput(f"{b}_GAIN", val) and ok
            st = sw_states.get(b, {"input_on": None, "output_on": None})
            if enable_switches:
                if not st["input_on"]:
                    ok = usm._caput(f"{b}_SW1", usm._SW1_INPUT_ON_BIT) and ok
                if not st["output_on"]:
                    ok = usm._caput(f"{b}_SW2", usm._SW2_OUTPUT_ON_BIT) and ok
            else:
                if st["input_on"] is False or st["output_on"] is False:
                    print(f"    WARN: {b} switch(es) OFF "
                          f"(in={st['input_on']}, out={st['output_on']}); "
                          f"GAIN written but column is not live. Use --enable-switches.")
            n_ok += ok
            n_fail += (not ok)
    return n_ok, n_fail


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def assemble(cfg: dict):
    """Load files, build A, normalize, invert, and plan columns. Returns a dict."""
    dof_order = list(cfg["dofs"])
    elec_order = list(cfg["electrode_row"].keys()) if "electrodes" not in cfg \
        else list(cfg["electrodes"])

    data_entries = cfg["data"]
    file_gains: list[FileGains] = []
    for entry in data_entries:
        path = resolve_path(entry["path"] if isinstance(entry, dict) else entry)
        if not path.exists():
            raise FileNotFoundError(f"result file not found: {path}")
        fg = load_result_file(path)
        # Optional cross-check of declared electrode(s).
        if isinstance(entry, dict) and entry.get("electrode"):
            declared = entry["electrode"]
            decl = [declared] if isinstance(declared, str) else list(declared)
            if set(decl) != set(fg.electrodes):
                raise ValueError(
                    f"{path.name}: config declares electrode(s) {decl} but file "
                    f"contains {fg.electrodes}")
        file_gains.append(fg)

    mode = cfg.get("field_normalize", "common_gamma")
    for fg in file_gains:
        g, gs, _ = pool_gamma(fg.f0, fg.Q, fg.residual, fg.dof_order)
        fg.gamma, fg.gamma_sigma = g, gs

    A, cells = build_forward_matrix(file_gains, dof_order, elec_order,
                                    duplicate=cfg.get("duplicate", "error"))
    # Lossy-phase + low-coherence warnings.
    coh_warn = float(cfg.get("low_coherence_warn", 0.3))
    for (d, e), c in cells.items():
        if lossy_phase(c.g_complex):
            print(f"  WARN: (dof={d}, elec={e}) phase={c.phase_deg:.0f} deg near +-90 -- "
                  f"signed-magnitude reduction is lossy")
        if np.isfinite(c.coherence) and c.coherence < coh_warn:
            print(f"  WARN: (dof={d}, elec={e}) low coherence {c.coherence:.2f} -- "
                  f"this gain is noisy and corrupts the inversion")

    A_field, s_lookup = apply_field_normalization(A, dof_order, elec_order, cells,
                                                  file_gains, mode)

    # Global field anchor: divide out the single particle-dependent global scalar
    # c (= cal*q/4pi^2 m) left over after susceptibility removal, so gain=1 is a
    # fixed, reproducible field whenever the electrodes are unchanged. See
    # anchor_scale() for the math. physical_scale P (default 1.0) is the future
    # V/m hook: set P = ||B||_F (from a geometry-matched COMSOL model) and gain=1
    # -> 1 V/m; realized |F| = gain * ||B||_F / P.
    anchor_cfg = cfg.get("field_anchor", {}) or {}
    anchor_mode = anchor_cfg.get("mode", "self_norm")     # none | self_norm | physical
    functional = anchor_cfg.get("functional", "frobenius")
    P = float(anchor_cfg.get("physical_scale", 1.0))
    if anchor_mode == "none":
        anchor_N, anchor = 1.0, 1.0
    elif anchor_mode in ("self_norm", "physical"):
        ref_label = anchor_cfg.get("reference_electrode")
        ref_idx = elec_order.index(ref_label) if ref_label else 0
        anchor_N = anchor_scale(A_field, functional, ref_idx)
        anchor = P / anchor_N
    else:
        raise ValueError(f"unknown field_anchor mode {anchor_mode!r}")
    A_field = A_field * anchor

    A_pinv = np.linalg.pinv(A_field)

    # Cell sigma (relative) for uncertainty propagation.
    n_avg = int(cfg.get("n_averages_for_error", 30))
    cell_sigma = {}
    for (d, e), c in cells.items():
        rel = gain_rel_error(c.coherence, n_avg) if np.isfinite(c.coherence) else 0.0
        cell_sigma[(d, e)] = rel

    # A_sigma from the anchored A_field: relative errors are scale-invariant, so
    # only the absolute sigma rescales with the anchor (consistent with A_field).
    A_sigma = afield_abs_sigma(A_field, dof_order, elec_order, cells, file_gains,
                               cell_sigma, mode)

    strict = bool(cfg.get("strict_subspace", False))
    plans = plan_columns(cfg, A_pinv, dof_order, elec_order, cell_sigma, strict,
                         a_field=A_field, a_sigma=A_sigma)

    return {
        "dof_order": dof_order, "elec_order": elec_order, "file_gains": file_gains,
        "A": A, "A_field": A_field, "A_sigma": A_sigma, "A_pinv": A_pinv,
        "s_lookup": s_lookup, "cells": cells, "cell_sigma": cell_sigma,
        "plans": plans, "mode": mode,
        "anchor_mode": anchor_mode, "anchor_functional": functional,
        "anchor_N": anchor_N, "anchor": anchor, "anchor_P": P,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config",
                    default=str(Path(__file__).parent / "upload_actuation_matrix_config.yml"))
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print everything; make no caput calls")
    ap.add_argument("--enable-switches", action="store_true",
                    help="auto-enable input+output switches on written elements "
                         "(signal goes LIVE on the trap electrodes)")
    ap.add_argument("--clear-uncoupled", action="store_true",
                    help="zero the electrode rows of every uncoupled column")
    ap.add_argument("--tramp", type=float, default=None,
                    help="ramp time (s) for each written element (overrides config)")
    ap.add_argument("--plot-naive-comparison", action="store_true",
                    help="also write a naive-vs-measurement-based ACTS E-field locus "
                         "plot (to today's data folder; see plot_naive_vs_measured_acts.py)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    tramp = args.tramp if args.tramp is not None else float(cfg.get("tramp_s", 5.0))

    try:
        res = assemble(cfg)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print_diagnostics(res["file_gains"], res["A"], res["A_field"], res["A_sigma"],
                      res["A_pinv"], res["dof_order"], res["elec_order"],
                      res["s_lookup"], res["cells"], cfg, res["plans"], res)
    print_column_plan(res["plans"], res["dof_order"], res["A_field"], res["elec_order"])

    if args.plot_naive_comparison:
        import plot_naive_vs_measured_acts as pnm
        pnm.make_comparison_plot(cfg, pnm._default_out_dir(), res=res)

    prefix = "DRY RUN -- " if args.dry_run else ""
    print(f"  {prefix}writing ACTS elements (tramp={tramp:.1f}s, "
          f"enable_switches={args.enable_switches}):")
    n_ok, n_fail = write_plans(res["plans"], cfg, res["elec_order"], tramp,
                               args.dry_run, args.enable_switches, args.clear_uncoupled)
    if args.dry_run:
        print("\n  Dry run complete -- no channels written.")
        return
    print(f"\n  Done: {n_ok} written, {n_fail} failed.")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
