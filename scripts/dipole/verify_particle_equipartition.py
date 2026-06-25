#!/usr/bin/env python3
"""
Verify equipartition on the RECORDED PARTICLE_X/Y/Z channels of a diagonalized CSD.

Unlike ``particle_xyz_from_W.py`` (which APPLIES a step-01 W matrix to the raw sensor
CSD), this script reads the ``Y1:DMD-PARTICLE_{X,Y,Z}_IN1`` auto-spectra that the live
SENSE matrix already produced — i.e. it checks the diagonalization *as actually realized
on the experiment*.

Equipartition
-------------
Each motional mode is a thermal harmonic oscillator: ½ k_i <q_i^2> = ½ k_B T_i, with
spring constant k_i = m_i (2 pi f0_i)^2. The displacement variance under each resonance
is the integral of its one-sided displacement PSD over the peak:

    Var_i = ∫ PSD_i(f) df   over [f0_i - W*Gamma_i, f0_i + W*Gamma_i]

(diaggui stores auto-spectra as ASD, so PSD = ASD^2.) The mode's effective temperature
is then proportional to

    T_i ∝ k_i <q_i^2> = m_i (2 pi f0_i)^2 Var_i.

With the dipole-pipeline COMMON displacement calibration (PARTICLE_X/Y/Z share one
length unit) and EQUAL effective mass across DOFs, equipartition predicts

    (2 pi f0_i)^2 * Var_i  EQUAL across x, y, z   (all at the bath temperature T).

We report this proxy normalized to its cross-DOF median ("relative T"); deviations from
1 flag either a mis-scaled DOF (calibration) or residual cross-coupling leakage (a DOF
reading another mode's energy). A DOF whose band variance is dominated by a peak that is
NOT its own resonance is leakage, not heat — so we also report, per PARTICLE channel,
which mode actually dominates its own band, and the inter-DOF coherence at each peak.

Var is computed two ways and shown side by side:
  * empirical  — band integral of (PSD − local floor), mains-masked (robust but still
    carries floor-subtraction noise and truncates the Lorentzian tails at ±band_width·Γ);
  * DHO model  — fit PSD = A·DHO(f0,Γ) + (c0 + c1/f) over the band and integrate the DHO
    term ALONE: Var = A·∫DHO = A·πΓ/2. This rejects the additive sensing/floor noise that
    biases the raw integral and captures the full wings analytically. When the two agree,
    the residual cross-DOF spread is physical (state/calibration), not floor bias.

Output goes to a per-run subdirectory  <data folder>/equipartition_<xml stem>/  so repeated
runs don't clog the data folder (re-running the same file overwrites; --out overrides parent).

Apples-to-apples re-validation (--baseline)
-------------------------------------------
The relT recovered here is NOT 1.0/1.0/1.0 even on the CSD W was fit from: the estimator
re-fits the W-applied data, so relT is a fixed (W, estimator, dataset) shape. Comparing it
to an idealized 1.0 is apples-to-oranges. Instead pass ``--baseline <step01_results.h5>``:
the script reconstructs the in-sample baseline relT by applying that run's W to its fit CSD
(``source_csd_path``) and reducing with the SAME current estimator, then reports the per-DOF
ratio ``relT_new / relT_baseline`` (geomean-normalized) and its spread as the FOM. The
estimator-induced in-sample shape divides out, so ratio ≈ 1 means equipartition is
reproduced. (A baseline *CSD xml* is accepted as a fallback but is ONLY valid if recorded
AFTER the W upload — a fit CSD's own PARTICLE_* channels are stale. Prefer the h5.)

Usage:
    python3 verify_particle_equipartition.py <csd.xml> \
        [--baseline <step01_results.h5>] \
        [--out <parent dir>] [--f0 x=40.5,y=54.8,z=6.1] [--gamma x=1.9,y=2.1,z=1.9] \
        [--band-width 4.0] [--mains 50,60]

If --f0 / --gamma are omitted they are estimated from the PARTICLE auto-spectra by a local
peak find within default search bands (or adopted from the --baseline h5 so both legs share
one band). --disagreement is DEPRECATED (it measures only the equip-vs-white_force anchor
gap, not equipartition drift) — use --baseline.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dttxml

# Single source of truth for the equipartition / DHO-variance estimator lives in the
# analysis repo (dipole_pipeline/diagnostics/equipartition.py) so this validator and the
# step-01 self-check use IDENTICAL math. That repo isn't importable as a package in the
# cds-testing env, so add its path directly; fall back to a vendored copy if absent.
_EQUIP_PATH = Path("/home/controls/analysis/mastqg/dipole_pipeline/diagnostics")
if _EQUIP_PATH.is_dir() and str(_EQUIP_PATH) not in sys.path:
    sys.path.insert(0, str(_EQUIP_PATH))
try:
    from equipartition import (dho, dho_model_variance, band_variance,  # noqa: E402
                               relative_temperature, relt_ratio)
except ImportError as _e:
    raise ImportError(
        f"Could not import the shared equipartition estimator from {_EQUIP_PATH}. "
        f"It is the single source of truth shared with the step-01 self-check. "
        f"Original error: {_e}")

# The W⊗CSD→relT reduction (used to RECONSTRUCT the baseline for --baseline) lives in the
# analysis repo's particle_xyz_from_W diagnostic, imported via the package root. Optional:
# if it can't be imported in this env, --baseline degrades gracefully (warn + vs-1.0).
_ANALYSIS_ROOT = Path("/home/controls/analysis")
if _ANALYSIS_ROOT.is_dir() and str(_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_ROOT))
try:
    from mastqg.dipole_pipeline.diagnostics.particle_xyz_from_W import (  # noqa: E402
        relT_from_W_on_csd)
except Exception as _e:  # noqa: BLE001 — optional; baseline reconstruction only
    relT_from_W_on_csd = None
    _RELT_IMPORT_ERR = _e

K_B = 1.380649e-23
DOFS = ["x", "y", "z"]
DEFAULT_SEARCH = {"x": (36.0, 45.0), "y": (51.0, 58.0), "z": (4.0, 9.0)}


def _spec(o):
    """Return (freqs, asd) from a dttxml PSD bunch (diaggui stores ASD)."""
    arr = np.atleast_1d(np.asarray(o.PSD).squeeze()).astype(float)
    fhz = np.asarray(o.FHz).squeeze()
    if fhz.shape != arr.shape:
        fhz = o.f0 + np.arange(arr.size) * o.df
    return fhz, np.abs(arr)


def _cross(csd, aname, bname):
    """CSD(aname, bname) from a dttxml CSD result, trying both orderings."""
    if aname in csd:
        Bs = list(csd[aname].channelB)
        if bname in Bs:
            return np.array(np.asarray(csd[aname].CSD)[Bs.index(bname)], dtype=complex)
    if bname in csd:
        Bs = list(csd[bname].channelB)
        if aname in Bs:
            return np.conj(np.array(np.asarray(csd[bname].CSD)[Bs.index(aname)], dtype=complex))
    return None


def parse_kv(s):
    out = {}
    if s:
        for tok in s.split(","):
            k, v = tok.split("=")
            out[k.strip()] = float(v)
    return out


def load_baseline(path, band_width):
    """Reconstruct the in-sample baseline relT from a --baseline argument.

    Primary: a step-01 results h5 — read W, channel_names, dofs, peak_frequency_hz_* and
    mode_gamma_hz_*, and source_csd_path, then apply W to that fit CSD and reduce with the
    CURRENT estimator (relT_from_W_on_csd). The baseline is RECONSTRUCTED, not frozen, so it
    stays comparable as the estimator evolves, and uses W⊗raw-sensors (the fit CSD's own
    PARTICLE_* channels are stale — recorded under the previous SENSE matrix).

    Returns (baseline_relT_dict, f0_dict, gamma_dict, rel_err_dict) so the caller can adopt
    the baseline's f0/Γ for the new CSD too (identical band on both legs = apples-to-apples)
    and propagate the baseline's finite-averaging error into the ratio error bar. Any failure
    returns (None, {}, {}, {}) with a warning so the caller falls back to vs-1.0 reporting.
    """
    p = os.path.expandvars(path)
    if not os.path.exists(p):
        print(f"  [warn] --baseline path not found: {p}; falling back to vs-1.0.",
              file=sys.stderr)
        return None, {}, {}, {}

    # CSD-xml fallback (only valid if recorded AFTER the W upload — loud warning).
    if not (p.endswith(".h5") or p.endswith(".hdf5")):
        print("  [warn] --baseline is not a step-01 .h5; treating it as a baseline CSD "
              "xml. This is ONLY valid if recorded AFTER the W upload (a fit CSD's own "
              "PARTICLE_* are stale). Prefer the step-01 results.h5.", file=sys.stderr)
        try:
            bd = dttxml.dtt_read(p)
            bpsd = bd.results["PSD"]
            n_avg_b = getattr(bpsd[f"Y1:DMD-PARTICLE_X_IN1"], "averages", None) \
                if "Y1:DMD-PARTICLE_X_IN1" in bpsd else None
            f0b, gb, varb, eb = {}, {}, {}, {}
            for dof in DOFS:
                ch = f"Y1:DMD-PARTICLE_{dof.upper()}_IN1"
                ch = ch if ch in bpsd else ch + "_DQ"
                fb, ab = _spec(bpsd[ch])
                lo, hi = DEFAULT_SEARCH[dof]
                msk = (fb >= lo) & (fb <= hi)
                f0b[dof] = float(fb[msk][np.argmax(ab[msk])])
                gb[dof] = max(0.05 * f0b[dof], 5 * (fb[1] - fb[0]))
                dfit = dho_model_variance(fb, ab ** 2, f0b[dof], gb[dof], band_width,
                                          n_avg=n_avg_b)
                varb[dof] = dfit["var"] if dfit.get("ok") else band_variance(
                    fb, ab ** 2, f0b[dof], gb[dof], band_width)
                verr = dfit.get("var_err", np.nan) if dfit.get("ok") else np.nan
                eb[dof] = (verr / varb[dof]) if (varb[dof] and np.isfinite(verr)) else np.nan
            return relative_temperature(f0b, varb), f0b, gb, eb
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] baseline CSD reduction failed: {e}; vs-1.0 fallback.",
                  file=sys.stderr)
            return None, {}, {}, {}

    if relT_from_W_on_csd is None:
        print(f"  [warn] could not import relT_from_W_on_csd ({_RELT_IMPORT_ERR}); "
              f"cannot reconstruct baseline from h5. Falling back to vs-1.0.",
              file=sys.stderr)
        return None, {}, {}, {}
    try:
        import h5py
        import json as _json
        with h5py.File(p, "r") as f:
            W = f["W"][:]
            a = dict(f.attrs)
            channels = _json.loads(a["channel_names"])
            dofs = _json.loads(a["dofs"])
            f0b = {d: float(a[f"peak_frequency_hz_{d}"]) for d in dofs}
            gb = {d: float(a[f"mode_gamma_hz_{d}"]) for d in dofs}
            src = a.get("source_csd_path", None)
        if not src:
            print("  [warn] --baseline h5 has no source_csd_path (older step-01 run); "
                  "cannot reconstruct baseline. Falling back to vs-1.0.", file=sys.stderr)
            return None, {}, {}, {}
        src = os.path.expandvars(src)
        if not os.path.exists(src):
            print(f"  [warn] fit CSD referenced by the h5 no longer exists: {src}; "
                  f"falling back to vs-1.0.", file=sys.stderr)
            return None, {}, {}, {}
        relT_base, eb = relT_from_W_on_csd(W, channels, dofs, f0b, gb, src,
                                           band_width=band_width, return_rel_err=True)
        return relT_base, f0b, gb, eb
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] baseline reconstruction from h5 failed: {e}; vs-1.0 fallback.",
              file=sys.stderr)
        return None, {}, {}, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("--out", default=None, help="output dir (default: alongside xml)")
    ap.add_argument("--f0", default=None, help="comma list x=..,y=..,z=.. (Hz); else auto")
    ap.add_argument("--gamma", default=None, help="comma list x=..,y=..,z=.. (Hz); else auto")
    ap.add_argument("--band-width", type=float, default=4.0,
                    help="band half-width in units of Gamma for the variance integral")
    ap.add_argument("--temp-k", type=float, default=295.0,
                    help="assumed bath temperature, for the absolute-scale note")
    ap.add_argument("--mains", default="50,60",
                    help="comma list of mains fundamentals to mask [Hz] (default 50,60)")
    ap.add_argument("--mains-guard", type=float, default=1.0,
                    help="half-width to mask around each mains harmonic [Hz]")
    ap.add_argument("--baseline", default=None,
                    help="PRIMARY apples-to-apples FOM. A step-01 results .h5: reconstructs "
                         "the in-sample baseline relT by applying its W to the fit CSD "
                         "(source_csd_path) and reducing with the CURRENT estimator, then "
                         "reports per-DOF ratio relT_new/relT_baseline (geomean-norm) and "
                         "its spread as the FOM. relT alone is NOT expected to be 1.0 (the "
                         "estimator re-fits data). A baseline *CSD xml* is accepted as a "
                         "fallback but is ONLY valid if recorded AFTER the W upload.")
    ap.add_argument("--disagreement", type=float, default=None,
                    help="DEPRECATED / NOT the yardstick: step-01 mode_scale_disagreement "
                         "(max|ln ratio| of white vs equip anchor). Measures only the "
                         "equip-vs-white_force anchor gap, irrelevant once committed to "
                         "equipartition. Use --baseline instead. If given (and no "
                         "--baseline), draws the old ±anchor band, labeled not-the-yardstick.")
    args = ap.parse_args()
    mains = [float(x) for x in args.mains.split(",") if x.strip()]

    xml = os.path.expandvars(args.xml)
    # Put results in a per-run subdirectory so repeated runs don't clog the data folder.
    # Default: <data folder>/equipartition_<xml stem>/ (re-running the same file overwrites;
    # different input files get their own subdir). Override the parent with --out.
    parent = Path(args.out) if args.out else Path(xml).parent
    out_dir = parent / f"equipartition_{Path(xml).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = dttxml.dtt_read(xml)
    psd = d.results["PSD"]
    csd = d.results.get("CSD", {})

    pchan = {dof: f"Y1:DMD-PARTICLE_{dof.upper()}_IN1" for dof in DOFS}
    for dof, ch in pchan.items():
        if ch not in psd:
            print(f"ERROR: {ch} not found in {xml}", file=sys.stderr)
            print(f"Available: {list(psd.keys())}", file=sys.stderr)
            sys.exit(1)

    n_avg = getattr(psd[pchan["x"]], "averages", None)
    freqs, asd = {}, {}
    for dof in DOFS:
        f, a = _spec(psd[pchan[dof]])
        freqs[dof] = f
        asd[dof] = a
    fcom = freqs["x"]   # common frequency grid

    # --- resolve f0 / gamma ---
    f0_in = parse_kv(args.f0)
    g_in = parse_kv(args.gamma)

    # --- reconstruct the apples-to-apples baseline (if requested) ---
    baseline_relT = None
    baseline_err = {}
    if args.baseline:
        baseline_relT, f0_base, g_base, baseline_err = load_baseline(
            args.baseline, args.band_width)
        # Adopt the baseline's f0/Γ for the NEW CSD too, so BOTH legs use an identical band
        # (apples-to-apples). Explicit user --f0/--gamma still win over the baseline.
        for dof in DOFS:
            if dof in f0_base and dof not in f0_in:
                f0_in[dof] = f0_base[dof]
            if dof in g_base and dof not in g_in:
                g_in[dof] = g_base[dof]

    f0, gamma = {}, {}
    for dof in DOFS:
        if dof in f0_in:
            f0[dof] = f0_in[dof]
        else:
            lo, hi = DEFAULT_SEARCH[dof]
            m = (fcom >= lo) & (fcom <= hi)
            f0[dof] = float(fcom[m][np.argmax(asd[dof][m])])
        if dof in g_in:
            gamma[dof] = g_in[dof]
        else:
            # crude HWHM->FWHM from the half-power points around the peak
            ip = int(np.argmin(np.abs(fcom - f0[dof])))
            pk = asd[dof][ip] ** 2
            half = pk / 2.0
            lo_i = ip
            while lo_i > 0 and asd[dof][lo_i] ** 2 > half:
                lo_i -= 1
            hi_i = ip
            while hi_i < len(fcom) - 1 and asd[dof][hi_i] ** 2 > half:
                hi_i += 1
            gamma[dof] = max(float(fcom[hi_i] - fcom[lo_i]), 5 * (fcom[1] - fcom[0]))

    # --- equipartition proxy ---
    # n_avg feeds the χ² per-bin weights so the DHO fit returns a physical var_err (1σ on
    # Var from finite averaging), which becomes the per-DOF relT error bar below.
    equip = {}
    for dof in DOFS:
        var, floor = band_variance(fcom, asd[dof] ** 2, f0[dof], gamma[dof],
                                   args.band_width, mains=mains,
                                   mains_guard=args.mains_guard, return_floor=True)
        dfit = dho_model_variance(fcom, asd[dof] ** 2, f0[dof], gamma[dof],
                                  args.band_width, mains=mains,
                                  mains_guard=args.mains_guard, n_avg=n_avg)
        w0 = 2 * np.pi * f0[dof]
        vdho = dfit["var"]
        verr = dfit.get("var_err", np.nan)
        equip[dof] = {"var": var, "floor": floor, "kq2": (w0 ** 2) * var,
                      "dho": dfit, "kq2_dho": (w0 ** 2) * vdho,
                      # fractional 1σ on relT_DHO (constants cancel under normalization)
                      "rel_err": (verr / vdho) if (vdho and np.isfinite(verr)) else np.nan}
    # Median-normalized relT via the shared helper (identical math to the baseline producer).
    relT = relative_temperature({d: f0[d] for d in DOFS},
                                {d: equip[d]["var"] for d in DOFS})
    relT_m = relative_temperature(
        {d: f0[d] for d in DOFS},
        {d: (equip[d]["dho"]["var"] if equip[d]["dho"].get("ok") else np.nan)
         for d in DOFS})
    relT = np.array([relT[d] for d in DOFS])
    relT_m = np.array([relT_m[d] for d in DOFS])
    new_rel_err = np.array([equip[d]["rel_err"] for d in DOFS])  # fractional, DHO leg

    # Apples-to-apples drift: per-DOF ratio of the new CSD's relT to the baseline's,
    # geomean-normalized. The DHO-model relT is the rigorous estimator used for the ratio.
    # Error bar = finite-averaging (χ²) error of BOTH legs in quadrature; the per-mode W
    # normalization cancels in the ratio (same W both legs), so it does not enter.
    ratio = spread_ratio = ratio_err = None
    if baseline_relT is not None:
        base_arr = np.array([baseline_relT.get(d, np.nan) for d in DOFS])
        base_err_arr = np.array([baseline_err.get(d, np.nan) for d in DOFS])
        ratio, spread_ratio, ratio_err = relt_ratio(
            relT_m, base_arr, rel_err_new=new_rel_err, rel_err_baseline=base_err_arr)

    # --- which mode dominates each PARTICLE band; self vs other ---
    dominance = {}
    for dof in DOFS:
        floor = np.median(asd[dof][(fcom > 25) & (fcom < 35)])
        heights = {p: asd[dof][np.argmin(np.abs(fcom - f0[p]))] / floor for p in DOFS}
        dominance[dof] = {"heights": heights, "self": dof,
                          "best": max(heights, key=heights.get)}

    # --- inter-DOF coherence at each peak (leakage) ---
    coh = {}
    for a, b in [("x", "y"), ("x", "z"), ("y", "z")]:
        c = _cross(csd, pchan[a], pchan[b])
        if c is None:
            coh[(a, b)] = None
            continue
        # coherence^2 = |Sab|^2 / (Saa Sbb)
        Saa = asd[a] ** 2
        Sbb = asd[b] ** 2
        c = c[:len(fcom)]
        coh[(a, b)] = np.abs(c) ** 2 / np.maximum(Saa * Sbb, 1e-300)

    # ---------------- Plot ----------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"x": "C0", "y": "C1", "z": "C2"}

    ax = axes[0, 0]
    for dof in DOFS:
        ax.loglog(fcom, asd[dof], color=colors[dof], lw=1.3,
                  label=f"PARTICLE_{dof.upper()}  (f0={f0[dof]:.2f}, Γ={gamma[dof]:.2f} Hz)")
        ax.axvline(f0[dof], color=colors[dof], ls=":", alpha=0.4)
        lo, hi = f0[dof] - args.band_width * gamma[dof], f0[dof] + args.band_width * gamma[dof]
        ax.axvspan(lo, hi, color=colors[dof], alpha=0.07)
        # overlay the DHO+floor model fit (ASD) over the fitted band. Draw in black so it
        # is visible against the same-colored data line (a same-color dashed overlay sits
        # invisibly on top of the solid trace).
        df = equip[dof]["dho"]
        if df.get("ok"):
            fb = np.linspace(*df["band"], 400)
            model_psd = df["A"] * dho(fb, df["f0"], df["gamma"]) + df["c0"] + df["c1"] / fb
            ax.loglog(fb, np.sqrt(model_psd), color="k", lw=1.4, ls="--", alpha=0.9,
                      label="DHO+floor fit" if dof == DOFS[0] else None)
    ax.set_xlim(2, 70)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Displacement ASD [common unit / √Hz]")
    ax.set_title("Recorded PARTICLE_XYZ auto-spectra (dashed = DHO+floor fit)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)

    # equipartition bar chart
    ax = axes[0, 1]
    xs = np.arange(3)
    if baseline_relT is not None:
        # Apples-to-apples mode: plot the per-DOF ratio relT_new/relT_baseline with a
        # PHYSICAL ±1σ error bar (finite-averaging / χ² error of both legs in quadrature).
        # The FOM is whether the ratio is consistent with 1 within that error — no arbitrary
        # tolerance band. The estimator-induced in-sample shape divides out in the ratio.
        w = 0.32
        ax.bar(xs - w / 2, relT_m, w, color=[colors[d] for d in DOFS], alpha=0.35,
               label="relT_new (DHO)")
        rerr = [ratio_err[i] if (ratio_err is not None and np.isfinite(ratio_err[i]))
                else 0.0 for i in range(3)]
        ax.bar(xs + w / 2, [ratio[i] for i in range(3)], w,
               yerr=rerr, capsize=4, ecolor="k",
               color=[colors[d] for d in DOFS], alpha=1.0, edgecolor="k", linewidth=0.6,
               hatch="//", label="ratio new/baseline (±1σ stat)")
        ax.axhline(1.0, color="k", ls="--", lw=1)
        for i, d in enumerate(DOFS):
            ax.text(i - w / 2, relT_m[i], f"{relT_m[i]:.2f}", ha="center", va="bottom", fontsize=8)
            if np.isfinite(ratio[i]):
                ax.text(i + w / 2, ratio[i] + rerr[i], f"{ratio[i]:.2f}", ha="center",
                        va="bottom", fontsize=8)
        ax.set_ylabel("relT (÷median)  &  ratio new/baseline")
        ax.set_title("Equipartition drift vs baseline (ratio → 1 within ±1σ = reproduced)")
    else:
        w = 0.38
        # ±1σ finite-averaging error on the DHO-model relT bars (the rigorous estimator).
        rmerr = [relT_m[i] * new_rel_err[i] if np.isfinite(new_rel_err[i]) else 0.0
                 for i in range(3)]
        ax.bar(xs - w / 2, relT, w, color=[colors[d] for d in DOFS], alpha=0.55,
               label="empirical (band ∫, floor-sub)")
        ax.bar(xs + w / 2, relT_m, w, yerr=rmerr, capsize=4, ecolor="k",
               color=[colors[d] for d in DOFS], alpha=1.0,
               edgecolor="k", linewidth=0.6, hatch="//", label="DHO model (±1σ stat)")
        ax.axhline(1.0, color="k", ls="--", lw=1)
        if args.disagreement is not None:
            # NOT the yardstick — measures only the white-vs-equip anchor mismatch.
            tol = np.exp(2 * args.disagreement)
            ax.axhspan(1.0 / tol, tol, color="k", alpha=0.08,
                       label=f"±anchor tol (NOT the yardstick — use --baseline)")
        for i, d in enumerate(DOFS):
            ax.text(i - w / 2, relT[i], f"{relT[i]:.2f}", ha="center", va="bottom", fontsize=8)
            if np.isfinite(relT_m[i]):
                ax.text(i + w / 2, relT_m[i] + rmerr[i], f"{relT_m[i]:.2f}", ha="center",
                        va="bottom", fontsize=8)
        ax.set_ylabel("relative effective T  [(2πf0)²·Var, ÷median]")
        ax.set_title("Equipartition: empirical vs DHO-model (in-sample shape)")
    ax.set_xticks(xs)
    ax.set_xticklabels([d.upper() for d in DOFS])
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.2)

    # inter-DOF coherence (leakage)
    ax = axes[1, 0]
    for (a, b), cc in coh.items():
        if cc is None:
            continue
        ax.semilogx(fcom, cc, lw=1.0, label=f"coh²({a.upper()},{b.upper()})")
    for dof in DOFS:
        ax.axvline(f0[dof], color=colors[dof], ls=":", alpha=0.4)
    ax.set_xlim(2, 70)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("magnitude-squared coherence")
    ax.set_title("Inter-DOF coherence (leakage; want →0 at each peak)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)

    # summary text
    ax = axes[1, 1]
    ax.axis("off")
    L = ["Equipartition summary  (empirical band-∫ vs DHO model)", ""]
    if baseline_relT is not None:
        L[0] = "Equipartition drift vs baseline  (relT_new / relT_base, geomean-norm)"
        L.append(f"  {'DOF':4s} {'f0[Hz]':>7s} {'relT_new':>8s} {'relT_base':>9s} {'ratio±1σ':>12s}")
        for i, d in enumerate(DOFS):
            re = ratio_err[i] if (ratio_err is not None and np.isfinite(ratio_err[i])) else float('nan')
            L.append(f"  {d.upper():4s} {f0[d]:7.2f} {relT_m[i]:8.2f} "
                     f"{baseline_relT.get(d, float('nan')):9.2f} {ratio[i]:6.2f}±{re:4.2f}")
        L.append("")
        # Per-DOF consistency with 1 within the physical (finite-averaging) error.
        worst = max((abs(ratio[i] - 1.0) / ratio_err[i]
                     for i in range(3)
                     if ratio_err is not None and np.isfinite(ratio_err[i]) and ratio_err[i] > 0),
                    default=float('nan'))
        L.append(f"  ratio spread max/min = {spread_ratio:.2f};  worst |ratio−1|/σ = {worst:.1f}")
        L.append("  ±1σ = finite-averaging (χ²) error, both legs in quadrature; the W")
        L.append("  normalization cancels in the ratio. Consistent if each ratio≈1 within ±1σ.")
        L.append("  THIS ratio is the FOM — NOT relT vs 1.0, NOT the --disagreement band.")
    else:
        L.append(f"  {'DOF':4s} {'f0[Hz]':>7s} {'Γfit[Hz]':>8s} {'relT_emp':>8s} {'relT_DHO±1σ':>13s}")
        for i, d in enumerate(DOFS):
            gfit = equip[d]["dho"].get("gamma", float("nan")) if equip[d]["dho"].get("ok") else float("nan")
            e = relT_m[i] * new_rel_err[i] if np.isfinite(new_rel_err[i]) else float('nan')
            L.append(f"  {d.upper():4s} {f0[d]:7.2f} {gfit:8.2f} {relT[i]:8.2f} {relT_m[i]:7.2f}±{e:4.2f}")
        L.append("")
        L.append(f"  DHO model: fit A·DHO(f0,Γ)+(c0+c1/f) per band, integrate the DHO term")
        L.append(f"  only (Var=πAΓ/2). ±1σ = finite-averaging (χ²) error, n_avg={n_avg}.")
        sp_e = np.nanmax(relT) / np.nanmin(relT)
        sp_m = np.nanmax(relT_m) / np.nanmin(relT_m)
        L.append(f"  spread max/min:  empirical={sp_e:.2f}   DHO-model={sp_m:.2f}")
        L.append("  relT is the in-sample shape; compare to baseline via --baseline,")
        L.append("  NOT to 1.0 (the estimator re-fits data — it is not 1.0/1.0/1.0).")
    L.append("")
    L.append("Self-dominance of each PARTICLE band (peak/floor):")
    for d in DOFS:
        h = dominance[d]["heights"]
        tag = "OK" if dominance[d]["best"] == d else f"LEAK→{dominance[d]['best'].upper()}"
        L.append(f"  PARTICLE_{d.upper()}: " +
                 ", ".join(f"{p}={h[p]:.1f}x" for p in DOFS) + f"   [{tag}]")
    L.append("")
    L.append("On-peak inter-DOF coherence² (leakage):")
    for (a, b), cc in coh.items():
        if cc is None:
            L.append(f"  ({a.upper()},{b.upper()}): n/a")
            continue
        vals = []
        for p in DOFS:
            ip = int(np.argmin(np.abs(fcom - f0[p])))
            vals.append(f"@{p}={cc[ip]:.2f}")
        L.append(f"  ({a.upper()},{b.upper()}): " + " ".join(vals))
    ax.text(0.0, 1.0, "\n".join(L), family="monospace", fontsize=9,
            va="top", ha="left", transform=ax.transAxes)

    fig.suptitle(f"PARTICLE_XYZ equipartition  ({Path(xml).name}, n_avg={n_avg})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = out_dir / "particle_equipartition.png"
    pdf = out_dir / "particle_equipartition.pdf"
    fig.savefig(png, dpi=130)
    fig.savefig(pdf)
    print(f"  wrote {png}")
    print(f"  wrote {pdf}")

    # --- per-DOF DHO fit cross-check figure (one panel per mode, zoomed to its band) ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))
    for k, dof in enumerate(DOFS):
        ax = axes2[k]
        df = equip[dof]["dho"]
        lo, hi = (df["band"] if df.get("ok")
                  else (max(f0[dof] - args.band_width * gamma[dof], 2.0),
                        f0[dof] + args.band_width * gamma[dof]))
        # widen the view a little beyond the fit band so the wings/floor are visible
        span = hi - lo
        vlo, vhi = max(lo - 0.5 * span, 1.0), hi + 0.5 * span
        m = (fcom >= vlo) & (fcom <= vhi)
        ax.semilogy(fcom[m], asd[dof][m] ** 2, color=colors[dof], lw=0.9, alpha=0.6,
                    label="data PSD")
        # mark mains-masked bins inside the fit band (excluded from the fit)
        for mf in mains:
            j = 1
            while j * mf <= vhi:
                if lo <= j * mf <= hi:
                    mm = m & (np.abs(fcom - j * mf) <= args.mains_guard)
                    if mm.any():
                        ax.semilogy(fcom[mm], asd[dof][mm] ** 2, "x", color="0.5",
                                    ms=4, label="masked (mains)" if j == 1 else None)
                j += 1
        if df.get("ok"):
            ff = np.linspace(vlo, vhi, 600)
            dho_term = df["A"] * dho(ff, df["f0"], df["gamma"])
            floor_term = df["c0"] + df["c1"] / ff
            ax.semilogy(ff, dho_term + floor_term, color="k", lw=1.6, label="model (DHO+floor)")
            ax.semilogy(ff, dho_term, color=colors[dof], lw=1.4, ls="--",
                        label="DHO term (∫=πAΓ/2)")
            ax.semilogy(ff, floor_term, color="0.4", lw=1.0, ls=":", label="floor (c0+c1/f)")
            ax.axvspan(lo, hi, color=colors[dof], alpha=0.06)
            ax.axvline(df["f0"], color=colors[dof], ls=":", alpha=0.5)
            ax.set_title(f"PARTICLE_{dof.upper()}: f0={df['f0']:.2f} Hz, Γ={df['gamma']:.2f} Hz\n"
                         f"relT_DHO={relT_m[k]:.2f}  (emp {relT[k]:.2f})", fontsize=9)
        else:
            ax.set_title(f"PARTICLE_{dof.upper()}: DHO fit FAILED", fontsize=9)
        ax.set_xlim(vlo, vhi)
        ax.set_xlabel("Frequency [Hz]")
        if k == 0:
            ax.set_ylabel("PSD [common unit² / Hz]")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.2)
    fig2.suptitle(f"DHO fit cross-check per DOF  ({Path(xml).name}, n_avg={n_avg})", fontsize=11)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    png2 = out_dir / "particle_equipartition_fits.png"
    pdf2 = out_dir / "particle_equipartition_fits.pdf"
    fig2.savefig(png2, dpi=130)
    fig2.savefig(pdf2)
    print(f"  wrote {png2}")
    print(f"  wrote {pdf2}")

    if baseline_relT is not None:
        print("\n  Equipartition drift vs baseline (ratio relT_new/relT_base ± 1σ stat, want ~1):")
        for i, d in enumerate(DOFS):
            re = ratio_err[i] if (ratio_err is not None and np.isfinite(ratio_err[i])) else float('nan')
            print(f"    {d.upper()}: f0={f0[d]:6.2f} Hz  relT_new={relT_m[i]:.2f}  "
                  f"base={baseline_relT.get(d, float('nan')):.2f}  ratio={ratio[i]:.2f} ± {re:.2f}")
        worst = max((abs(ratio[i] - 1.0) / ratio_err[i]
                     for i in range(3)
                     if ratio_err is not None and np.isfinite(ratio_err[i]) and ratio_err[i] > 0),
                    default=float('nan'))
        print(f"    ratio spread max/min = {spread_ratio:.2f};  worst |ratio−1|/σ = {worst:.1f}"
              f"  (consistent with equipartition if each ratio≈1 within ±1σ)")
    else:
        print("\n  Equipartition (relative T; in-sample shape, compare via --baseline NOT 1.0):")
        for i, d in enumerate(DOFS):
            print(f"    {d.upper()}: f0={f0[d]:6.2f} Hz  relT_emp={relT[i]:.2f}  relT_DHO={relT_m[i]:.2f}")
        print(f"    spread max/min:  empirical={np.nanmax(relT)/np.nanmin(relT):.2f}  "
              f"DHO-model={np.nanmax(relT_m)/np.nanmin(relT_m):.2f}")
    print("\n  Self-dominance:")
    for d in DOFS:
        tag = "OK" if dominance[d]["best"] == d else f"LEAK→{dominance[d]['best'].upper()}"
        print(f"    PARTICLE_{d.upper()}: dominant={dominance[d]['best'].upper()}  [{tag}]")


if __name__ == "__main__":
    main()
