#!/var/lib/cds-conda/base/envs/cds-testing/bin/python3
# Run with: /var/lib/cds-conda/base/envs/cds-testing/bin/python3 plot_naive_vs_measured_acts.py
"""Illustrate naive-diagonalized vs measurement-based ACTS as an E-field locus.

A *naive* electrode diagonalization assumes a symmetric four-electrode geometry
and drives the two field quadratures with the sign patterns (indexed E1..E4):

    cosine / x quadrature :  c_x = [+1, -1, -1, +1]
    sine   / y quadrature :  c_y = [+1, +1, -1, -1]

A rotating field is then commanded by phasing the two quadratures:

    v(theta) = cos(theta) * c_x + sin(theta) * c_y ,   theta in [0, 2*pi)

If the electrodes really were symmetric and orthogonal, propagating v(theta)
through the true actuator coupling would trace a perfect circle of constant
magnitude whose direction equals theta. In reality the measured forward matrix
``A_field`` (DOF x electrode, field-per-count after susceptibility removal — the
same matrix ``upload_actuation_matrix.py`` inverts) is neither symmetric nor
orthogonal, so the naive commands trace a distorted, mis-pointed locus:

    E_naive(theta) = A_field @ v(theta)            (a 2-vector in the x-y plane)

The MEASUREMENT-BASED ACTS instead writes the columns A+ @ x_hat and A+ @ y_hat,
so the same cos/sin phasing reproduces the commanded unit circle exactly
(A_field @ A+ = I): that circle is the reference this script overlays.

This makes the cost of skipping the measurement visible: magnitude ripple
(anisotropy) and a theta-dependent pointing error.

USAGE
    plot_naive_vs_measured_acts.py [--config CONFIG] [--out-dir DIR]
                                   [--n-points N] [--no-show-paths]

    --config       ACTS inversion config (default: upload_actuation_matrix_config.yml).
                   Its ``data:`` entries select the actuator-gain result files; edit
                   that config to choose which measurement to illustrate.
    --out-dir      Where to write the PNG (default: today's data folder under
                   $MQG_DROPBOX_PATH/worker1/data/YYMMDD/).
    --n-points     Number of theta samples around the circle (default 361).

It can also be invoked by ``upload_actuation_matrix.py --plot-naive-comparison``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Agg backend: no display required; must be set before importing pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — must follow matplotlib.use()

# Reuse the inversion machinery (forward-matrix assembly + field normalization).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import upload_actuation_matrix as uam  # noqa: E402
# Step 2 of this workflow: propagate the naive E-field distortion to the dipole
# libration sideband and the d-inference error. All that physics lives in
# dipole_sideband_model.py — see its module docstring.
import dipole_sideband_model as dsm  # noqa: E402

# Naive diagonalization sign patterns, indexed by electrode label.
NAIVE_CX = {"E1": +1.0, "E2": -1.0, "E3": -1.0, "E4": +1.0}  # cosine / x quadrature
NAIVE_CY = {"E1": +1.0, "E2": +1.0, "E3": -1.0, "E4": -1.0}  # sine   / y quadrature

DEFAULT_CONFIG = Path(__file__).resolve().parent / "upload_actuation_matrix_config.yml"


# --------------------------------------------------------------------------- #
# Pure math (no I/O) — unit-testable
# --------------------------------------------------------------------------- #
def naive_command_vectors(elec_order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (c_x, c_y) naive command vectors ordered to match ``elec_order``."""
    missing = [e for e in elec_order if e not in NAIVE_CX]
    if missing:
        raise ValueError(
            f"naive sign pattern is defined only for E1..E4; got electrode(s) "
            f"{missing}. Update NAIVE_CX / NAIVE_CY for a different electrode set.")
    c_x = np.array([NAIVE_CX[e] for e in elec_order], dtype=float)
    c_y = np.array([NAIVE_CY[e] for e in elec_order], dtype=float)
    return c_x, c_y


def naive_field_locus(A_field: np.ndarray, elec_order: list[str], dof_order: list[str],
                      n_points: int = 361) -> dict:
    """Propagate the naive cos/sin command through the measured forward matrix.

    Returns a dict with theta (rad), the 2-D field locus, its magnitude/angle,
    and the mean-radius normalization factor. Only the in-plane x,y DOFs are used
    (a z row, if present, is dropped — the field locus is an x-y-plane object).
    """
    xy_idx = [dof_order.index(d) for d in ("x", "y") if d in dof_order]
    if len(xy_idx) != 2:
        raise ValueError(
            f"need both 'x' and 'y' in the measured DOFs to draw an x-y field "
            f"locus; got dof_order={dof_order}")
    A_xy = np.asarray(A_field, dtype=float)[xy_idx, :]   # (2, n_elec)

    c_x, c_y = naive_command_vectors(elec_order)
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    # v(theta) = cos(theta) * c_x + sin(theta) * c_y  -> (n_elec, n_points)
    v = np.outer(c_x, np.cos(theta)) + np.outer(c_y, np.sin(theta))
    field = A_xy @ v                                     # (2, n_points)

    mag = np.hypot(field[0], field[1])
    ang = np.arctan2(field[1], field[0])
    mean_r = float(np.mean(mag)) if np.mean(mag) > 0 else 1.0
    return {
        "theta": theta,
        "field": field,                 # (2, n_points), absolute A_field units
        "field_norm": field / mean_r,   # mean radius -> 1
        "mag": mag,
        "mag_norm": mag / mean_r,
        "ang": ang,
        "mean_radius": mean_r,
        "A_xy": A_xy,
        "c_x": c_x,
        "c_y": c_y,
    }


def direction_error_deg(theta: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """Achieved field angle minus commanded angle, wrapped to (-180, 180] deg."""
    err = np.degrees(ang - theta)
    return (err + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def build_figure(locus: dict, title: str) -> plt.Figure:
    """4-panel figure: polar locus, Cartesian locus, |E| vs theta, dir-error vs theta."""
    theta = locus["theta"]
    fx, fy = locus["field_norm"]
    mag = locus["mag_norm"]
    ang = locus["ang"]
    err = direction_error_deg(theta, ang)

    ripple = (mag.max() - mag.min()) / np.mean(mag) * 100.0
    max_err = float(np.max(np.abs(err)))

    fig = plt.figure(figsize=(12, 10))

    # --- polar locus ---
    ax_p = fig.add_subplot(2, 2, 1, projection="polar")
    ax_p.plot(theta, np.ones_like(theta), color="#2ca02c", lw=2.0,
              label="measurement-based (reference)")
    ax_p.plot(np.arctan2(fy, fx), np.hypot(fx, fy), color="#d62728", lw=1.8,
              label="naive diagonalization")
    ax_p.set_title("E-field locus (polar, mean radius = 1)", fontsize=10, pad=14)
    ax_p.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=8)

    # --- Cartesian locus ---
    ax_c = fig.add_subplot(2, 2, 2)
    circ = np.linspace(0, 2 * np.pi, 361)
    ax_c.plot(np.cos(circ), np.sin(circ), color="#2ca02c", lw=2.0,
              label="measurement-based (reference)")
    ax_c.plot(fx, fy, color="#d62728", lw=1.8, label="naive diagonalization")
    # Mark theta=0 on the naive locus to show the pointing offset.
    ax_c.plot([0, fx[0]], [0, fy[0]], color="#d62728", ls=":", lw=1.0)
    ax_c.plot([0, 1], [0, 0], color="#2ca02c", ls=":", lw=1.0)
    ax_c.scatter([fx[0]], [fy[0]], color="#d62728", s=25, zorder=5,
                 label=r"naive $\theta=0$")
    ax_c.set_aspect("equal", "box")
    ax_c.axhline(0, color="gray", lw=0.5)
    ax_c.axvline(0, color="gray", lw=0.5)
    ax_c.set_xlabel(r"$E_x$ (mean radius = 1)")
    ax_c.set_ylabel(r"$E_y$ (mean radius = 1)")
    ax_c.set_title("E-field locus (Cartesian x-y)", fontsize=10)
    ax_c.legend(fontsize=8, loc="upper right")
    ax_c.grid(True, alpha=0.3)

    # --- magnitude vs theta ---
    ax_m = fig.add_subplot(2, 2, 3)
    ax_m.axhline(1.0, color="#2ca02c", lw=2.0, label="measurement-based")
    ax_m.plot(np.degrees(theta), mag, color="#d62728", lw=1.8,
              label="naive diagonalization")
    ax_m.set_xlabel(r"commanded angle $\theta$ (deg)")
    ax_m.set_ylabel("|E| (mean radius = 1)")
    ax_m.set_title(f"Field magnitude vs angle  (anisotropy ripple {ripple:.1f}%)",
                   fontsize=10)
    ax_m.set_xlim(0, 360)
    ax_m.set_xticks(np.arange(0, 361, 45))
    ax_m.grid(True, alpha=0.3)
    ax_m.legend(fontsize=8, loc="best")

    # --- direction error vs theta ---
    ax_e = fig.add_subplot(2, 2, 4)
    ax_e.axhline(0.0, color="#2ca02c", lw=2.0, label="measurement-based")
    ax_e.plot(np.degrees(theta), err, color="#d62728", lw=1.8,
              label="naive diagonalization")
    ax_e.set_xlabel(r"commanded angle $\theta$ (deg)")
    ax_e.set_ylabel("pointing error (deg)")
    ax_e.set_title(f"Direction error vs angle  (max {max_err:.1f} deg)", fontsize=10)
    ax_e.set_xlim(0, 360)
    ax_e.set_xticks(np.arange(0, 361, 45))
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(fontsize=8, loc="best")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# --------------------------------------------------------------------------- #
# Orchestration (importable by upload_actuation_matrix.py)
# --------------------------------------------------------------------------- #
def make_comparison_plot(cfg: dict, out_dir: Path, n_points: int = 361,
                         res: dict | None = None,
                         cfg_text: str | None = None,
                         sideband_model: bool = True,
                         sim_opts: dict | None = None) -> tuple[Path, dict]:
    """Assemble A_field from the config (or reuse a prior assemble() result),
    compute the naive locus, and write the figure. Returns (png_path, locus).

    ``cfg_text``: if given, the config YAML used for this run is saved alongside
    the figure as ``config_used.yml`` (so a solo run is self-documenting).

    ``sideband_model``: if True (default) also run **step 2** — hand the assembled
    A_field to ``dipole_sideband_model.run_model`` to compute the libration
    sideband, the d-inference error table, and (unless disabled in ``sim_opts``)
    the numeric EOM spectrum. ``sim_opts`` overrides the dipole/sim defaults (see
    ``dipole_sideband_model.run_model``); the ``dipole_sim:`` block of the config
    supplies defaults."""
    if res is None:
        res = uam.assemble(cfg)
    A_field = res["A_field"]
    elec_order = res["elec_order"]
    dof_order = res["dof_order"]

    locus = naive_field_locus(A_field, elec_order, dof_order, n_points=n_points)

    files = ", ".join(Path(fg.path).parent.name for fg in res["file_gains"])
    title = ("Naive vs measurement-based ACTS — E-field in the x-y plane\n"
             f"forward matrix from: {files}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    png = out_dir / f"{ts}_naive_vs_measured_acts.png"
    fig = build_figure(locus, title)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    if cfg_text is not None:
        (out_dir / "config_used.yml").write_text(cfg_text)
        print(f"    config: {out_dir / 'config_used.yml'}")

    # Console summary.
    ripple = (locus["mag_norm"].max() - locus["mag_norm"].min()) / np.mean(locus["mag_norm"])
    err = direction_error_deg(locus["theta"], locus["ang"])
    print("  Naive vs measurement-based ACTS comparison")
    print(f"    forward matrix A_field (rows {dof_order}, cols {elec_order}):")
    print("     ", str(np.round(A_field, 4)).replace("\n", "\n      "))
    print(f"    naive |E| anisotropy ripple : {ripple * 100:.1f}%  "
          f"(max/min = {locus['mag_norm'].max() / locus['mag_norm'].min():.3f})")
    print(f"    naive max pointing error    : {np.max(np.abs(err)):.1f} deg")
    print(f"    wrote: {png}")

    # --- Step 2: dipole libration sideband + d-inference error ---------------
    if sideband_model:
        opts = dict(cfg.get("dipole_sim", {}) or {})
        if sim_opts:
            opts.update({k: v for k, v in sim_opts.items() if v is not None})
        files = ", ".join(Path(fg.path).parent.name for fg in res["file_gains"])
        M = dsm.effective_command_matrix(A_field, elec_order, dof_order)
        print("\n  Step 2: dipole libration-sideband model "
              "(see dipole_sideband_model.py)")
        dsm.run_model(M, out_dir,
                      title=f"Dipole libration-sideband model -- {files}", **opts)

    return png, locus


def _default_out_dir() -> Path:
    import os
    root = Path(os.path.expandvars("$MQG_DROPBOX_PATH/worker1/data"))
    return root / time.strftime("%y%m%d")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="ACTS inversion config (selects the actuator-gain data files)")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: today's data folder)")
    ap.add_argument("--n-points", type=int, default=361,
                    help="number of theta samples around the circle (default 361)")
    # Step 2 (dipole sideband model) toggles. Dipole/sim parameters default from
    # the config's `dipole_sim:` block; these flags override the common ones.
    ap.add_argument("--no-sideband-model", dest="sideband_model",
                    action="store_false", default=True,
                    help="skip step 2 (the dipole libration-sideband model)")
    ap.add_argument("--no-simulate", dest="simulate", action="store_false",
                    default=None, help="skip the numeric EOM->spectrum in step 2")
    ap.add_argument("--drag", dest="drag", action="store_true", default=None,
                    help="enable gas drag in step 2 (default off; for lock-loss)")
    ap.add_argument("--d-emum", type=float, default=None)
    ap.add_argument("--observed-sideband-hz", type=float, default=None)
    ap.add_argument("--f0-hz", type=float, default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    cfg_text = cfg_path.read_text()
    cfg = yaml.safe_load(cfg_text)

    # A solo run gets its own timestamped folder (with the config it used);
    # --out-dir overrides the parent. The flag-driven path from
    # upload_actuation_matrix.py calls make_comparison_plot directly instead.
    parent = Path(args.out_dir) if args.out_dir else _default_out_dir()
    run_dir = parent / f"{time.strftime('%Y%m%d_%H%M%S')}_naive_vs_measured_acts"
    sim_opts = {"simulate": args.simulate, "drag": args.drag,
                "d_emum": args.d_emum,
                "observed_sideband_hz": args.observed_sideband_hz,
                "f0_hz": args.f0_hz}
    try:
        make_comparison_plot(cfg, run_dir, n_points=args.n_points, cfg_text=cfg_text,
                             sideband_model=args.sideband_model, sim_opts=sim_opts)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
