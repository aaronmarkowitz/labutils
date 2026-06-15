#!/var/lib/cds-conda/base/envs/cds-testing/bin/python3
"""Bode-style diagnostic plots for actuator gain measurement results.

Callable as a standalone script (reads HDF5 output from measure_actuator_gain.py)
OR importable by measure_actuator_gain.py for inline plotting after a run.

Public API
----------
plot_measurement(records, dof_fits, electrodes, cfg, plots_dir, is_trim=False)
    Generate and save all Bode plots.  Returns list of Path objects written.

CLI usage
---------
    plot_actuator_gain.py <run_dir>
        Reads <run_dir>/actuator_gain_results.h5 and writes PNGs to
        <run_dir>/plots/.

    plot_actuator_gain.py --records-json records.json <run_dir>
        Use a pre-built JSON records list instead of the HDF5 file (for testing).
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
import h5py

# Agg backend: no display required; must be set before importing pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — must follow matplotlib.use()


# --------------------------------------------------------------------------- #
# Lorentzian curve helper (mirrors measure_actuator_gain.plant_lorentzian)
# Duplicated here so plot_actuator_gain is standalone (no circular import).
# --------------------------------------------------------------------------- #
def _plant_lorentzian(f, f0, Q):
    """Peak-magnitude-normalized complex Lorentzian (|H(f0)| = 1)."""
    f = np.asarray(f, dtype=float)
    h_raw = 1.0 / (f0 ** 2 - f ** 2 + 1j * f * f0 / Q)
    peak = Q / (f0 ** 2)
    return h_raw / peak


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
_DOF_COLORS = {"x": "#1f77b4", "y": "#ff7f0e", "z": "#2ca02c"}
_ELEC_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p"]

# One color per electrode (cycles if more than 8)
_ELEC_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
]


def _elec_color(idx: int) -> str:
    return _ELEC_PALETTE[idx % len(_ELEC_PALETTE)]


def _elec_marker(idx: int) -> str:
    return _ELEC_MARKERS[idx % len(_ELEC_MARKERS)]


def _freq_curve(records_for_plot: list) -> np.ndarray:
    """Return a dense frequency array spanning the data in records_for_plot."""
    freqs = [r["freq"] for r in records_for_plot]
    if not freqs:
        return np.linspace(1.0, 100.0, 500)
    lo = min(freqs) * 0.7
    hi = max(freqs) * 1.3
    return np.linspace(max(lo, 0.1), hi, 500)


def _should_log_xaxis(records: list) -> bool:
    freqs = [r["freq"] for r in records]
    if len(freqs) < 2:
        return False
    return max(freqs) / max(min(freqs), 1e-9) > 2.0


# --------------------------------------------------------------------------- #
# Core figure builder
# --------------------------------------------------------------------------- #
def _build_bode_figure(
    title: str,
    records_subset: list,          # records to plot as scatter
    all_records: list,             # all records (for fit curve frequency range)
    dof_fits: dict,
    electrodes: list,
    is_trim: bool,
    show_all_elec_colors: bool,    # combined plot → colour by electrode
) -> plt.Figure:
    """Build a 3-panel Bode figure (amplitude, phase, coherence).

    Parameters
    ----------
    title:                  Figure suptitle.
    records_subset:         Records to scatter-plot.
    all_records:            Full set (used to compute the curve frequency range).
    dof_fits:               Dict of DofFit-like objects keyed by 'x'/'y'/'z'.
    electrodes:             Ordered list of electrode names.
    is_trim:                If True, scatter alpha = 1.0 (ignore coherence).
    show_all_elec_colors:   If True, colour points by electrode index.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    ax_amp, ax_phase, ax_coh = axes

    f_curve = _freq_curve(all_records)

    # ---- amplitude + phase: one curve per DOF ----
    for dof in ("x", "y", "z"):
        fit = dof_fits.get(dof)
        if fit is None:
            continue
        color = _DOF_COLORS[dof]

        # Fitted curve — use the first electrode's gain for illustration
        # (the curve shape is shared; amplitude is scaled by |G[0]|)
        G = fit.gains
        # pick the electrode with largest |G| to anchor the curve magnitude
        anchor_idx = int(np.argmax(np.abs(G)))
        G_anchor = G[anchor_idx]
        H_curve = _plant_lorentzian(f_curve, fit.f0, fit.Q)
        curve_vals = G_anchor * H_curve

        label = f"{dof.upper()} fit  f₀={fit.f0:.2f} Hz  Q={fit.Q:.1f}"
        ax_amp.plot(f_curve, np.abs(curve_vals), color=color, lw=1.5,
                    label=label, zorder=2)
        ax_phase.plot(f_curve, np.degrees(np.angle(curve_vals)), color=color,
                      lw=1.5, zorder=2)

    # ---- scatter: one point per (record, dof) — colored by measurement channel ----
    for r in records_subset:
        e_idx = electrodes.index(r["electrode"]) if r["electrode"] in electrodes else 0
        marker = _elec_marker(e_idx)

        for dof in ("x", "y", "z"):
            color = _DOF_COLORS[dof]
            alpha = 1.0 if is_trim else float(np.clip(r["coh"][dof], 0.05, 1.0))
            tf_val = r["tf"][dof]
            ax_amp.scatter(r["freq"], abs(tf_val),
                           color=color, marker=marker, s=30, alpha=alpha, zorder=3)
            ax_phase.scatter(r["freq"], np.degrees(np.angle(tf_val)),
                             color=color, marker=marker, s=30, alpha=alpha, zorder=3)
            ax_coh.scatter(r["freq"], r["coh"][dof],
                           color=color, marker=marker, s=25, zorder=3)

    ax_coh.axhline(0.9, color="gray", ls="--", lw=1.0, label="coh = 0.9")

    # ---- legends ----
    ax_amp.legend(fontsize=7, loc="upper right")
    if show_all_elec_colors:
        elec_handles = [
            plt.Line2D([0], [0], marker=_elec_marker(i), color="w",
                       markerfacecolor="gray", markersize=7,
                       label=electrodes[i] if i < len(electrodes) else f"E{i}")
            for i in range(len(electrodes))
        ]
        dof_handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=_DOF_COLORS[d], markersize=7,
                       label=d.upper())
            for d in ("x", "y", "z")
        ]
        ax_coh.legend(handles=elec_handles + dof_handles, fontsize=7,
                      loc="lower right", ncol=2)

    # ---- axes formatting ----
    if _should_log_xaxis(all_records):
        for ax in axes:
            ax.set_xscale("log")

    ax_amp.set_yscale("log")
    ax_amp.set_ylabel("|TF| (counts/counts)")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.set_ylim(-200, 200)
    ax_coh.set_ylabel("Coherence")
    ax_coh.set_ylim(0, 1.05)
    ax_coh.set_xlabel("Frequency (Hz)")
    if not show_all_elec_colors:
        ax_coh.legend(fontsize=7, loc="lower right")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Gain matrix heatmap
# --------------------------------------------------------------------------- #
def _build_gain_matrix_figure(dof_fits: dict, electrodes: list) -> plt.Figure:
    """2D heatmap of |G| with complex-phase angle annotations."""
    dofs = ["x", "y", "z"]
    n_dof = len(dofs)
    n_elec = len(electrodes)

    gain_abs = np.zeros((n_dof, n_elec))
    gain_angle = np.zeros((n_dof, n_elec))
    for i, d in enumerate(dofs):
        fit = dof_fits.get(d)
        if fit is not None and len(fit.gains) == n_elec:
            gain_abs[i] = np.abs(fit.gains)
            gain_angle[i] = np.degrees(np.angle(fit.gains))

    fig, ax = plt.subplots(figsize=(max(5, 1.5 * n_elec), 4))
    im = ax.imshow(gain_abs, aspect="auto", origin="upper",
                   cmap="viridis", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="|G| (counts/counts)")

    # annotate cells
    for i in range(n_dof):
        for j in range(n_elec):
            text = f"|G|={gain_abs[i, j]:.3g}\n∠{gain_angle[i, j]:+.0f}°"
            text_color = "white" if gain_abs[i, j] < gain_abs.max() * 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color=text_color)

    ax.set_xticks(range(n_elec))
    ax.set_xticklabels(electrodes)
    ax.set_yticks(range(n_dof))
    ax.set_yticklabels([d.upper() for d in dofs])
    ax.set_xlabel("Electrode")
    ax.set_ylabel("DOF")
    ax.set_title("Actuator gain matrix |G|")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def plot_measurement(
    records: list,
    dof_fits: dict,
    electrodes: list,
    cfg: dict,
    plots_dir: Path,
    is_trim: bool = False,
) -> list:
    """Generate and save all Bode plots.

    Parameters
    ----------
    records:    List of per-tone dicts from compute_tfs / loaded from HDF5.
                Each dict: {"electrode": str, "dof_intended": str, "freq": float,
                            "tf": {"x": complex, "y": complex, "z": complex},
                            "coh": {"x": float, "y": float, "z": float}}
    dof_fits:   Dict keyed "x"/"y"/"z", values have attributes .f0, .Q, .gains,
                .fit_plant, .per_electrode_coherence (DofFit or SimpleNamespace).
    electrodes: Ordered list of electrode names, e.g. ["E1","E2","E3","E4"].
    cfg:        Full config dict (used for f0/Q seed fallback if dof_fits missing).
    plots_dir:  Directory in which to write PNGs.
    is_trim:    If True, scatter alpha = 1.0 and the gain matrix figure is skipped.

    Returns
    -------
    List of Path objects for all files written.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    written: list = []

    # ---- per-electrode Bode figures ----
    for e in electrodes:
        records_e = [r for r in records if r["electrode"] == e]
        title = f"Electrode {e}"
        fig = _build_bode_figure(
            title=title,
            records_subset=records_e,
            all_records=records,
            dof_fits=dof_fits,
            electrodes=electrodes,
            is_trim=is_trim,
            show_all_elec_colors=False,
        )
        out = plots_dir / f"bode_{e}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)

    # ---- combined Bode figure (all electrodes, coloured by electrode) ----
    fig = _build_bode_figure(
        title="All electrodes (combined)",
        records_subset=records,
        all_records=records,
        dof_fits=dof_fits,
        electrodes=electrodes,
        is_trim=is_trim,
        show_all_elec_colors=True,
    )
    out = plots_dir / "bode_combined.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    written.append(out)

    # ---- gain matrix heatmap (final measurement only, not trim steps) ----
    if not is_trim:
        fig = _build_gain_matrix_figure(dof_fits, electrodes)
        out = plots_dir / "gain_matrix.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)

    return written


# --------------------------------------------------------------------------- #
# HDF5 loader (standalone mode)
# --------------------------------------------------------------------------- #
def _load_results_h5(h5_path: Path):
    """Reconstruct records + DofFit-like objects from actuator_gain_results.h5.

    Returns (records, dof_fits, electrodes, cfg_text_or_none).
    """
    with h5py.File(h5_path, "r") as f:
        tone_freqs = f["tone_freqs"][:]
        tone_electrode = json.loads(f.attrs["tone_electrode"])
        electrodes = json.loads(f.attrs["electrodes"])

        tf = {}
        coh = {}
        for d in ("x", "y", "z"):
            tf[d] = f[f"tf_{d}_real"][:] + 1j * f[f"tf_{d}_imag"][:]
            coh[d] = f[f"coherence_{d}"][:]

        # Reconstruct dof_intended from the electrode assignment and tone order;
        # it is not stored in the HDF5 — use an empty string as fallback.
        # (The records are used for plotting; dof_intended is only needed for
        #  alpha-by-coherence, which falls back to the coh value of each DOF.)
        # If the file was written with tone_electrode attr we can infer dof_intended
        # approximately from which cluster of tones each electrode belongs to.
        # For simplicity, default to "x" (only affects scatter colour in trim plots).
        n_tones = len(tone_freqs)
        records = []
        for i in range(n_tones):
            rec = {
                "electrode": tone_electrode[i],
                "dof_intended": "x",   # best-effort; not stored in h5
                "freq": float(tone_freqs[i]),
                "tf": {d: complex(tf[d][i]) for d in ("x", "y", "z")},
                "coh": {d: float(coh[d][i]) for d in ("x", "y", "z")},
            }
            records.append(rec)

        # Reconstruct DofFit-like objects from attrs
        gain_real = f["gain_matrix_real"][:]   # shape (3, n_elec)
        gain_imag = f["gain_matrix_imag"][:]
        gain_matrix = gain_real + 1j * gain_imag

        dof_fits = {}
        for i, d in enumerate(("x", "y", "z")):
            dof_fits[d] = SimpleNamespace(
                dof=d,
                f0=float(f.attrs[f"peak_frequency_hz_{d}"]),
                Q=float(f.attrs[f"Q_{d}"]),
                gains=gain_matrix[i],
                fit_plant=bool(f.attrs[f"fit_plant_{d}"]),
                residual_norm=float(f.attrs[f"residual_norm_{d}"]),
                per_electrode_coherence={},
            )

        cfg_text = f.attrs.get("params_yaml", None)

    return records, dof_fits, electrodes, cfg_text


# --------------------------------------------------------------------------- #
# CLI main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="Run directory containing actuator_gain_results.h5")
    ap.add_argument("--records-json", default=None,
                    help="JSON file with records list (overrides HDF5; for testing)")
    ap.add_argument("--config", default=None,
                    help="Config YAML to use (falls back to embedded params_yaml attr)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    h5_path = run_dir / "actuator_gain_results.h5"
    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found", file=sys.stderr)
        sys.exit(1)

    records, dof_fits, electrodes, cfg_text = _load_results_h5(h5_path)

    if args.records_json:
        with open(args.records_json) as fj:
            raw = json.load(fj)
        records = []
        for r in raw:
            records.append({
                "electrode": r["electrode"],
                "dof_intended": r.get("dof_intended", "x"),
                "freq": float(r["freq"]),
                "tf": {d: complex(r["tf"][d]) for d in ("x", "y", "z")},
                "coh": {d: float(r["coh"][d]) for d in ("x", "y", "z")},
            })

    # Build a minimal cfg for fallback f0/Q display
    cfg = {}
    if args.config:
        with open(args.config) as fc:
            cfg = yaml.safe_load(fc)
    elif cfg_text:
        cfg = yaml.safe_load(cfg_text) or {}

    plots_dir = run_dir / "plots"
    written = plot_measurement(records, dof_fits, electrodes, cfg, plots_dir)
    for p in written:
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
