#!/var/lib/cds-conda/base/envs/cds-testing/bin/python3
# Run with: /var/lib/cds-conda/base/envs/cds-testing/bin/python3 dipole_sideband_model.py
"""Dipole libration-sideband model: how the naive rotating-field drive biases d.

This is **step 2** of the naive-vs-measured ACTS workflow. Step 1
(``plot_naive_vs_measured_acts.py``) shows that the naive electrode
diagonalization (top-bottom ``++--``, left-right ``+--+``, phased cos/sin)
produces a distorted, *elliptical* E-field locus when propagated through the
measured forward matrix ``A_field``. This module answers the physics question
that distortion raises:

    PRIMARY QUESTION
    ----------------
    "If I believe I am delivering the rotating field my naive COMSOL model
     predicts (a clean unit circle from the two electrode-gap length scales),
     how wrong is the libration frequency I infer -- and therefore the dipole
     moment d -- once the real, asymmetric electrodes turn that command into an
     ellipse?"

PHYSICS BACKGROUND (Rider et al., PRA 99 041802; Afek et al., PRA 104 053512)
-----------------------------------------------------------------------------
A permanent dipole d in a field E rotating at omega_0 librates about the
instantaneous field direction at omega_phi = sqrt(d*E/I). The cross-polarized
readout P_perp ~ sin^2(phi_MS) carries a carrier at 2*omega_0 with libration
sidebands at 2*omega_0 +/- omega_phi. The sideband offset omega_phi is the
experimental handle on d/I and on the field magnitude.

CO-ROTATING DECOMPOSITION (the heart of this module)
----------------------------------------------------
The naive command through the measured matrix gives a complex in-plane field

    E(theta) = E_x(theta) + i E_y(theta) = A cos(theta) + B sin(theta),

where A, B are the (complex-as-2-vector) responses to the cosine and sine
quadratures. Decompose into co- and counter-rotating phasors:

    E(theta) = Eplus e^{+i theta} + Eminus e^{-i theta},
    Eplus  = (A - iB)/2,   Eminus = (A + iB)/2.

A perfect circle has Eminus = 0. In the frame rotating at omega_0 the field is
E'(t) = Eplus + Eminus e^{-2 i omega_0 t}: the **static** trapping field the
dipole sees is the co-rotating projection Eplus; Eminus spins at 2*omega_0.

Therefore the libration frequency is set by |Eplus|, NOT by the average
magnitude <|E|> around the locus:

    omega_phi = sqrt(d |Eplus| / I),     |Eplus| <= <|E|>    (eq. iff circle).

This resolves the three sub-questions:
  Q1  Is the sideband at the average-magnitude frequency? NO -- it is |Eplus|,
      which is <= <|E|>. (For our data they differ by only ~2%, because the
      large magnitude ripple lands mostly in the non-trapping Eminus.)
  Q2  Does the 2*omega_0 magnitude modulation split/broaden the sideband? In the
      fast-rotation regime (2*omega_0 >> 2*omega_phi) NO: Eminus drives a damped
      Mathieu equation far off its 2*omega_phi parametric resonance, so the only
      effects are a negligible Kapitza shift O((omega_phi/2 omega_0)^2 * eps^2)
      and tiny micro-sidebands at carrier +/- 2*omega_0 (well separated from the
      libration sidebands). It would split/broaden only near omega_0 ~ omega_phi.
  Q3  Does the field *phase* deviation (not just magnitude) matter? YES -- Eplus
      is a *complex* average, so the 2*omega_0 phase wobble enters it directly.

THE d-INFERENCE ERROR (headline deliverable)
--------------------------------------------
Inferring d from the measured sideband via omega_phi^2 = d E / I, but using a
*believed* field E_believed instead of the true |Eplus|, gives

    d_inferred / d_true = |Eplus|_true / E_believed.

This is a pure ratio of the measured effective command matrix M -- scale-free,
needing no V/m calibration. We report it under four reductions of the naive
COMSOL belief (M assumed diagonal: x-quadrature -> pure x-field M_xx from the
L-R gap, y-quadrature -> pure y-field M_yy from the T-B gap):

    1. x only        E_believed = |M_xx|
    2. y only        E_believed = |M_yy|
    3. average       E_believed = (|M_xx| + |M_yy|)/2   (and <|E|>_naive)
    4. best naive    E_believed = |Eplus|_naive = |M_xx + M_yy|/2

Way 4 is the apples-to-apples |Eplus|_naive vs |Eplus|_true. The antisymmetric
cross term (M_yx - M_xy) raises |Eplus|_true above |Eplus|_naive (you
*over*-estimate d), while symmetric distortion bleeds into the non-trapping
Eminus. Ways 3 and 4 coincide when M_xx, M_yy share a sign.

USAGE
-----
Two input modes:

    # (a) from a measured ACTS config (assembles A_field, builds M):
    dipole_sideband_model.py --a-field-from-config upload_actuation_matrix_config.yml

    # (b) from an explicit assumed ellipse (no measurement needed):
    dipole_sideband_model.py --semi-major 1.45 --semi-minor 0.85 --tilt-deg 20
    dipole_sideband_model.py --eplus 1.15 --eminus 0.30

Particle / regime options (sensible microdiamond defaults):
    --d-emum 13           permanent dipole moment in e*um
    --radius-um 1.5       sphere radius (-> I = (2/5) m r^2)
    --density 3510        diamond density kg/m^3 (override I directly with --I)
    --observed-sideband-hz 100   anchors the Hz axis to recorded data
    --f0-hz 7500          field rotation frequency (fast regime: 2 f0 >> omega_phi)
    --drag / --no-drag    gas drag in the numeric EOM (default OFF -- the
                          libration measurement is drag-negligible; --drag is for
                          the future lock-loss measurement)
    --pressure-mbar 1e-6  used only with --drag
    --simulate / --no-simulate   run the numeric EOM->PSD confirmation (default ON)

It is normally invoked automatically by
``plot_naive_vs_measured_acts.py`` (which passes the assembled A_field straight
into ``run_model``), but runs fine standalone for either input mode.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np

# ---- physical constants -------------------------------------------------- #
E_CHARGE = 1.602176634e-19   # C
MICRON = 1.0e-6              # m

# Naive diagonalization sign patterns, indexed by electrode label (match
# plot_naive_vs_measured_acts.py).
NAIVE_CX = {"E1": +1.0, "E2": -1.0, "E3": -1.0, "E4": +1.0}  # cosine / x quadrature
NAIVE_CY = {"E1": +1.0, "E2": +1.0, "E3": -1.0, "E4": -1.0}  # sine   / y quadrature


# --------------------------------------------------------------------------- #
# Pure math: effective command matrix + co-rotating decomposition
# --------------------------------------------------------------------------- #
def naive_command_vectors(elec_order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (c_x, c_y) naive command vectors ordered to match ``elec_order``."""
    missing = [e for e in elec_order if e not in NAIVE_CX]
    if missing:
        raise ValueError(
            f"naive sign pattern is defined only for E1..E4; got {missing}.")
    c_x = np.array([NAIVE_CX[e] for e in elec_order], dtype=float)
    c_y = np.array([NAIVE_CY[e] for e in elec_order], dtype=float)
    return c_x, c_y


def effective_command_matrix(A_field: np.ndarray, elec_order: list[str],
                             dof_order: list[str]) -> np.ndarray:
    """Build the 2x2 command->field matrix ``M = A_xy @ [c_x | c_y]``.

    Column 0 is the field produced by the cosine (x) quadrature command, column 1
    by the sine (y) quadrature. M is what the experiment *actually* delivers; the
    naive belief is that M is diagonal.
    """
    xy_idx = [dof_order.index(d) for d in ("x", "y") if d in dof_order]
    if len(xy_idx) != 2:
        raise ValueError(
            f"need both 'x' and 'y' DOFs to form the x-y field; got {dof_order}")
    A_xy = np.asarray(A_field, dtype=float)[xy_idx, :]      # (2, n_elec)
    c_x, c_y = naive_command_vectors(elec_order)
    return np.column_stack([A_xy @ c_x, A_xy @ c_y])        # (2, 2)


def corotating_decomposition(M: np.ndarray, n_points: int = 3601) -> dict:
    """Decompose the elliptical field locus E(theta)=A cos+B sin into Eplus/Eminus.

    ``M`` is the 2x2 [A | B] command matrix (columns = cos/sin quadrature fields).
    Returns Eplus, Eminus (complex), their magnitudes, ellipticity eps=|E-|/|E+|,
    semi-axes, tilt, the numeric mean radius <|E|>, and the calibration-robust
    shape factor <|E|>/|E+|.
    """
    M = np.asarray(M, dtype=float)
    A = M[0, 0] + 1j * M[1, 0]      # response to cos quadrature, as complex E
    B = M[0, 1] + 1j * M[1, 1]      # response to sin quadrature
    Eplus = (A - 1j * B) / 2.0
    Eminus = (A + 1j * B) / 2.0
    aEp, aEm = abs(Eplus), abs(Eminus)

    semi_major = aEp + aEm
    semi_minor = abs(aEp - aEm)
    # Ellipse orientation: the major axis points along (arg(E+) + arg(E-))/2.
    tilt = 0.5 * (np.angle(Eplus) + np.angle(Eminus))

    # Numeric mean radius <|E|> around the locus (for the average-vs-|E+| gap).
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    Eth = A * np.cos(theta) + B * np.sin(theta)
    mag = np.abs(Eth)
    mean_radius = float(np.mean(mag))
    ripple = float((mag.max() - mag.min()) / mean_radius) if mean_radius > 0 else 0.0

    return {
        "M": M, "A": A, "B": B,
        "Eplus": Eplus, "Eminus": Eminus,
        "abs_Eplus": aEp, "abs_Eminus": aEm,
        "eps": (aEm / aEp) if aEp > 0 else np.inf,
        "semi_major": semi_major, "semi_minor": semi_minor,
        "axis_ratio": (semi_major / semi_minor) if semi_minor > 0 else np.inf,
        "tilt_rad": float(tilt),
        "mean_radius": mean_radius,
        "ripple": ripple,
        "shape_factor": (mean_radius / aEp) if aEp > 0 else np.inf,  # <|E|>/|E+|
    }


def ellipse_to_M(semi_major: float, semi_minor: float, tilt_rad: float = 0.0,
                 handedness: int = +1) -> np.ndarray:
    """Build an equivalent 2x2 command matrix M for an assumed ellipse.

    Lets the model run with no measurement: given semi-axes (and optional tilt /
    rotation handedness) produce an M whose co-rotating decomposition reproduces
    the requested ellipse. Uses |E+|=(a+b)/2, |E-|=(a-b)/2 with a real-axis
    parameterization rotated by ``tilt_rad``.
    """
    aEp = 0.5 * (semi_major + semi_minor)
    aEm = 0.5 * (semi_major - semi_minor)
    # Put the major axis along `tilt`: arg(E+)+arg(E-) = 2*tilt. Simplest split:
    Eplus = aEp * np.exp(1j * tilt_rad)
    Eminus = aEm * np.exp(1j * tilt_rad) * (1 if handedness >= 0 else 1)
    # Invert Eplus=(A-iB)/2, Eminus=(A+iB)/2  ->  A=E++E-, B=i(E+-E-)... careful:
    # A - iB = 2E+, A + iB = 2E-  =>  A = E+ + E-,  iB = E- - E+  => B = -i(E- - E+)
    A = Eplus + Eminus
    B = -1j * (Eminus - Eplus)
    M = np.array([[A.real, B.real], [A.imag, B.imag]], dtype=float)
    return M


# --------------------------------------------------------------------------- #
# d-inference error table
# --------------------------------------------------------------------------- #
@dataclass
class BeliefWay:
    key: str
    label: str
    E_believed: float
    d_ratio: float           # d_inferred / d_true = |E+|_true / E_believed
    omega_phi_frac_err: float  # fractional error in omega_phi you'd predict


def d_inference_error(decomp: dict, comsol_alpha: tuple[float, float] | None = None
                      ) -> list[BeliefWay]:
    """Four reductions of E_believed -> d_inferred/d_true = |E+|_true/E_believed.

    The naive belief takes the diagonal of M (the on-axis couplings COMSOL fixes
    from the electrode gaps). ``comsol_alpha`` overrides (|alpha_x|,|alpha_y|) if
    you have explicit COMSOL per-axis field predictions instead of diag(M).

    For each way d_inferred/d_true = |E+|_true / E_believed. Since at a fixed
    *measured* sideband omega_phi ~ sqrt(d), the fractional error you'd make in a
    *predicted* omega_phi from the believed field is sqrt(E_believed/|E+|) - 1 =
    1/sqrt(d_ratio) - 1.
    """
    M = decomp["M"]
    aEp_true = decomp["abs_Eplus"]
    if comsol_alpha is not None:
        ax, ay = abs(comsol_alpha[0]), abs(comsol_alpha[1])
    else:
        ax, ay = abs(M[0, 0]), abs(M[1, 1])     # diag(M)
    aEp_naive = abs(M[0, 0] + M[1, 1]) / 2.0 if comsol_alpha is None \
        else abs(ax + ay) / 2.0

    believed = [
        ("x_only", "x quadrature only  (E=|M_xx|)", ax),
        ("y_only", "y quadrature only  (E=|M_yy|)", ay),
        ("average", "average of x,y      (E=(|M_xx|+|M_yy|)/2)", 0.5 * (ax + ay)),
        ("best_naive", "best naive (project naive ellipse, E=|E+|_naive)", aEp_naive),
    ]
    ways = []
    for key, label, Eb in believed:
        d_ratio = aEp_true / Eb if Eb > 0 else np.inf
        frac = (1.0 / np.sqrt(d_ratio) - 1.0) if d_ratio > 0 else np.inf
        ways.append(BeliefWay(key, label, Eb, d_ratio, frac))
    return ways


# --------------------------------------------------------------------------- #
# Particle properties + absolute anchor
# --------------------------------------------------------------------------- #
def moment_of_inertia(radius_um: float, density: float) -> float:
    """Solid-sphere moment of inertia I = (2/5) m r^2."""
    r = radius_um * MICRON
    m = density * (4.0 / 3.0) * np.pi * r ** 3
    return 0.4 * m * r ** 2


def field_from_sideband(omega_phi: float, d_SI: float, I: float) -> float:
    """Invert omega_phi = sqrt(d E / I) for the absolute field E (V/m)."""
    return I * omega_phi ** 2 / d_SI


@dataclass
class SidebandSummary:
    omega_phi_true_hz: float        # measured/true sideband offset (anchored)
    E_true_SI: float                # absolute |E+|_true in V/m at the anchor
    field_scale_SI_per_unit: float  # V/m per A_field unit (from the anchor)
    d_SI: float
    I: float
    f0_hz: float
    micro_sideband_suppression: float  # (omega_phi / 2 omega_0)^2
    ways: list[BeliefWay] = dc_field(default_factory=list)


def sideband_summary(decomp: dict, d_emum: float, I: float,
                     observed_sideband_hz: float, f0_hz: float,
                     comsol_alpha: tuple[float, float] | None = None
                     ) -> SidebandSummary:
    """Anchor the Hz axis to the observed sideband and assemble the d-error table.

    The observed libration sideband is taken to be the TRUE one (set by
    |E+|_true). That fixes the V/m-per-A_field-unit scale, from which the absolute
    |E+|_true in V/m follows. The d-error ways are scale-free (pure ratios), so
    the anchor only affects reported Hz / V/m, never d_inferred/d_true.
    """
    d_SI = d_emum * E_CHARGE * MICRON
    omega_phi = 2.0 * np.pi * observed_sideband_hz
    E_true_SI = field_from_sideband(omega_phi, d_SI, I)         # V/m for |E+|_true
    scale = E_true_SI / decomp["abs_Eplus"] if decomp["abs_Eplus"] > 0 else np.nan
    omega0 = 2.0 * np.pi * f0_hz
    suppression = (omega_phi / (2.0 * omega0)) ** 2
    ways = d_inference_error(decomp, comsol_alpha)
    return SidebandSummary(
        omega_phi_true_hz=observed_sideband_hz, E_true_SI=E_true_SI,
        field_scale_SI_per_unit=scale, d_SI=d_SI, I=I, f0_hz=f0_hz,
        micro_sideband_suppression=suppression, ways=ways)


# --------------------------------------------------------------------------- #
# Numeric EOM -> spectrum (confirmation of the analytic |E+| result)
# --------------------------------------------------------------------------- #
def _field_timeseries(M: np.ndarray, t: np.ndarray, omega0: float,
                      scale_SI: float) -> tuple[np.ndarray, np.ndarray]:
    """E_x(t), E_y(t) in V/m for the locus described by M, sampled at theta=omega0 t."""
    theta = omega0 * t
    cs, sn = np.cos(theta), np.sin(theta)
    Ex = scale_SI * (M[0, 0] * cs + M[0, 1] * sn)
    Ey = scale_SI * (M[1, 0] * cs + M[1, 1] * sn)
    return Ex, Ey


def simulate_sideband_spectrum(M: np.ndarray, d_SI: float, I: float,
                               scale_SI: float, f0_hz: float,
                               beta: float = 0.0,
                               oversample: int = 8, freq_resolution_hz: float = 2.0,
                               settle_frac: float = 0.3) -> dict:
    """Integrate the lab-frame planar-rotor EOM and return the P_perp PSD.

        I * phi'' = d * (E_x sin phi - E_y cos phi) - beta * phi'

    (z-component of torque d x E, with the dipole along phi; beta = I/tau gas
    drag, zero by default). The rotor is launched near the rotating field and
    librates; P_perp = sin^2(phi) carries the carrier at 2*omega_0 with sidebands
    at 2*omega_0 +/- omega_phi. We return the PSD, the detected carrier and
    sideband, and the extracted omega_phi for comparison with sqrt(d|E+|/I).

    ``oversample`` sets fs = oversample * (2 f0) (>=4 needed to resolve the
    carrier). ``freq_resolution_hz`` sets the record length T (kept, post-settle)
    so the FFT bin spacing ~ freq_resolution_hz: resolving a ~100 Hz sideband
    needs a long *single-segment* FFT, not Welch averaging. Sub-bin peak
    interpolation then pins omega_phi finely. Uses scipy if available; otherwise a
    fixed-step RK4 + windowed rFFT. beta=0 is the drag-negligible libration
    regime; beta>0 is the lock-loss study.
    """
    omega0 = 2.0 * np.pi * f0_hz
    f_carrier = 2.0 * f0_hz
    fs = oversample * f_carrier                       # sample rate
    # Record length so that the *kept* (post-settle) segment gives the requested
    # bin spacing for a single-segment FFT.
    T_kept = 1.0 / max(freq_resolution_hz, 1e-6)
    T = T_kept / (1.0 - settle_frac)
    n = int(T * fs)
    t = np.arange(n) / fs

    Ex, Ey = _field_timeseries(M, t, omega0, scale_SI)
    # interpolation helpers for the ODE RHS (continuous time)
    def E_at(tt):
        th = omega0 * tt
        cs, sn = np.cos(th), np.sin(th)
        ex = scale_SI * (M[0, 0] * cs + M[0, 1] * sn)
        ey = scale_SI * (M[1, 0] * cs + M[1, 1] * sn)
        return ex, ey

    def rhs(tt, y):
        phi, w = y
        ex, ey = E_at(tt)
        # Torque = z-component of (d_vec x E_vec) with d_vec = d(cos phi, sin phi):
        #   tau_z = d (cos phi * Ey - sin phi * Ex) = d|E| sin(theta_E - phi),
        # which restores phi toward the instantaneous field direction.
        torque = d_SI * (ey * np.cos(phi) - ex * np.sin(phi))
        return [w, (torque - beta * w) / I]

    # Launch co-rotating and tracking the field (theta=0 -> +x at t=0), but kicked
    # a small angle off equilibrium so the libration mode is excited and shows up
    # as the sideband. The kick is small enough to stay in the linear regime.
    libration_kick_rad = 0.05
    y0 = [libration_kick_rad, omega0]

    try:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(rhs, (t[0], t[-1]), y0, t_eval=t, method="RK45",
                        rtol=1e-7, atol=1e-9, max_step=1.0 / fs)
        phi = sol.y[0]
    except Exception:
        # Fixed-step RK4 fallback.
        phi = np.empty(n)
        y = np.array(y0, dtype=float)
        dt = 1.0 / fs
        for i in range(n):
            phi[i] = y[0]
            k1 = np.array(rhs(t[i], y))
            k2 = np.array(rhs(t[i] + dt / 2, y + dt / 2 * k1))
            k3 = np.array(rhs(t[i] + dt / 2, y + dt / 2 * k2))
            k4 = np.array(rhs(t[i] + dt, y + dt * k3))
            y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    Pperp = np.sin(phi) ** 2
    # Drop the settling transient, then take a single long windowed FFT (a
    # Hann-windowed periodogram) so the bin spacing stays fine enough to resolve
    # the libration sideband. Welch averaging would coarsen the bins and smear it.
    i0 = int(settle_frac * n)
    sig = Pperp[i0:] - np.mean(Pperp[i0:])
    win = np.hanning(len(sig))
    sp = np.fft.rfft(sig * win)
    psd = (np.abs(sp) ** 2) / (fs * np.sum(win ** 2))
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)

    # Locate carrier (near 2 f0) and the nearest sideband peak.
    omega_phi_hz = _extract_sideband(freqs, psd, f_carrier)
    return {
        "t": t, "phi": phi, "Pperp": Pperp,
        "freqs": freqs, "psd": psd, "fs": fs,
        "f_carrier": f_carrier, "omega_phi_hz": omega_phi_hz,
    }


def _extract_sideband(freqs: np.ndarray, psd: np.ndarray, f_carrier: float,
                      search_hz: float = 2000.0) -> float:
    """Find the libration sideband offset from the carrier in a PSD."""
    # Carrier bin: largest peak within +/-5% of f_carrier.
    cmask = np.abs(freqs - f_carrier) < 0.05 * f_carrier
    if not np.any(cmask):
        return np.nan
    c_idx = np.where(cmask)[0][np.argmax(psd[cmask])]
    fc = freqs[c_idx]
    # Sideband: largest peak in (fc, fc+search_hz], excluding the carrier skirt.
    df = freqs[1] - freqs[0]
    skirt = max(3.0 * df, 5.0)               # keep clear of the carrier leakage
    smask = (freqs > fc + skirt) & (freqs <= fc + search_hz)
    if not np.any(smask):
        return np.nan
    rel = np.where(smask)[0]
    s_idx = rel[np.argmax(psd[smask])]
    # Parabolic sub-bin interpolation on the log-PSD for a finer peak estimate.
    if 0 < s_idx < len(psd) - 1:
        y0, y1, y2 = np.log(psd[s_idx - 1] + 1e-300), np.log(psd[s_idx] + 1e-300), \
            np.log(psd[s_idx + 1] + 1e-300)
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        f_peak = freqs[s_idx] + delta * df
    else:
        f_peak = freqs[s_idx]
    return float(f_peak - fc)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def analytic_omega_phi_hz(abs_Eplus: float, scale_SI: float, d_SI: float,
                          I: float) -> float:
    """omega_phi/2pi in Hz from the analytic sqrt(d |E+| / I)."""
    E_SI = abs_Eplus * scale_SI
    return float(np.sqrt(d_SI * E_SI / I) / (2.0 * np.pi))


def build_results_markdown(decomp: dict, summary: SidebandSummary,
                           sim: dict | None, title: str,
                           drag_beta: float = 0.0) -> str:
    """Compose the human-readable results report (also echoed to console)."""
    d = decomp
    lines = []
    L = lines.append
    L(f"# {title}\n")
    L("## Effective command matrix M = A_xy @ [c_x | c_y]\n")
    L("Columns: field from the cosine (x) and sine (y) quadrature commands.\n")
    L("```")
    L(f"M = [[{d['M'][0,0]:+.4f}, {d['M'][0,1]:+.4f}],")
    L(f"     [{d['M'][1,0]:+.4f}, {d['M'][1,1]:+.4f}]]")
    L("```\n")
    L("## Co-rotating decomposition  E(theta) = E+ e^{+i th} + E- e^{-i th}\n")
    L(f"- |E+| (co-rotating, the trapping field) = {d['abs_Eplus']:.4f}")
    L(f"- |E-| (counter-rotating, non-trapping)  = {d['abs_Eminus']:.4f}")
    L(f"- ellipticity eps = |E-|/|E+|            = {d['eps']:.4f}")
    L(f"- semi-major / semi-minor                = {d['semi_major']:.4f} / "
      f"{d['semi_minor']:.4f}  (axis ratio {d['axis_ratio']:.3f})")
    L(f"- ellipse tilt                           = {np.degrees(d['tilt_rad']):.1f} deg")
    L(f"- magnitude ripple (max-min)/mean        = {d['ripple']*100:.1f}%")
    L(f"- mean radius <|E|>                       = {d['mean_radius']:.4f}")
    L(f"- **shape factor <|E|>/|E+|**             = {d['shape_factor']:.4f}  "
      f"(calibration-robust; the average-vs-|E+| gap)\n")
    L("## PRIMARY: d_inferred / d_true under the naive belief\n")
    L("d_inferred/d_true = |E+|_true / E_believed  (scale-free).\n")
    L("| Way | E_believed | d_inf/d_true | err in d | omega_phi pred. err |")
    L("|---|---|---|---|---|")
    for w in summary.ways:
        L(f"| {w.label} | {w.E_believed:.4f} | {w.d_ratio:.4f} | "
          f"{(w.d_ratio-1)*100:+.1f}% | {w.omega_phi_frac_err*100:+.1f}% |")
    L("")
    best = next(w for w in summary.ways if w.key == "best_naive")
    L(f"**Best-case (way 4): believing the naive COMSOL field over-/under-estimates "
      f"d by {(best.d_ratio-1)*100:+.1f}%** "
      f"(predicted omega_phi off by {best.omega_phi_frac_err*100:+.1f}%).\n")
    L("## Absolute scale (anchored to the observed sideband)\n")
    L(f"- observed/true sideband omega_phi/2pi   = {summary.omega_phi_true_hz:.2f} Hz")
    L(f"- => |E+|_true                            = {summary.E_true_SI:.1f} V/m "
      f"({summary.E_true_SI/1e3:.2f} kV/m)")
    L(f"- dipole d                                = {summary.d_SI/(E_CHARGE*MICRON):.1f} e*um")
    L(f"- moment of inertia I                     = {summary.I:.3e} kg m^2")
    L(f"- rotation f0                             = {summary.f0_hz:.0f} Hz "
      f"(2 f0 = {2*summary.f0_hz:.0f} Hz)\n")
    L("## Fast-regime verdict (Q2/Q3)\n")
    L(f"- 2 f0 / (2 omega_phi/2pi) = {2*summary.f0_hz/(2*summary.omega_phi_true_hz):.0f} "
      f">> 1 -> no parametric splitting/broadening of the libration sideband.")
    L(f"- the 2*omega_0 ripple seeds micro-sidebands at carrier +/- 2 f0, suppressed "
      f"by ~(omega_phi/2 omega_0)^2 = {summary.micro_sideband_suppression:.2e}.")
    L(f"- the magnitude ripple ({d['ripple']*100:.0f}%) lands mostly in the "
      f"non-trapping E-, so the libration frequency moves only with |E+|.\n")
    if sim is not None and np.isfinite(sim.get("omega_phi_hz", np.nan)):
        ana = analytic_omega_phi_hz(d["abs_Eplus"], summary.field_scale_SI_per_unit,
                                    summary.d_SI, summary.I)
        L("## Numeric EOM confirmation\n")
        L(f"- numeric sideband offset  = {sim['omega_phi_hz']:.2f} Hz")
        L(f"- analytic sqrt(d|E+|/I)   = {ana:.2f} Hz")
        if ana > 0:
            L(f"- agreement                = {sim['omega_phi_hz']/ana:.3f} x analytic")
        L(f"- gas drag beta            = {drag_beta:.3e} "
          f"({'OFF' if drag_beta == 0 else 'ON'})\n")
    return "\n".join(lines)


def build_spectrum_figure(sims: dict[str, dict], summary: SidebandSummary,
                          out_png: Path, title: str) -> None:
    """PSD of P_perp around the carrier: naive vs circular, sidebands marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"naive": "#d62728", "circular": "#2ca02c"}
    for name, sim in sims.items():
        if sim is None:
            continue
        f = sim["freqs"] - sim["f_carrier"]
        ax.semilogy(f, sim["psd"], color=colors.get(name, None), lw=1.3,
                    label=f"{name}")
        wp = sim.get("omega_phi_hz", np.nan)
        if np.isfinite(wp):
            ax.axvline(wp, color=colors.get(name, None), ls=":", lw=1.0)
            ax.axvline(-wp, color=colors.get(name, None), ls=":", lw=1.0)
    ax.set_xlim(-3 * summary.omega_phi_true_hz, 3 * summary.omega_phi_true_hz)
    ax.set_xlabel(r"$f - 2 f_0$ (Hz)")
    ax.set_ylabel(r"$P_\perp$ PSD (arb.)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Library entry point (used by plot_naive_vs_measured_acts.py and main())
# --------------------------------------------------------------------------- #
def run_model(M: np.ndarray, out_dir: Path, *, d_emum: float = 13.0,
              radius_um: float = 1.5, density: float = 3510.0,
              I: float | None = None, observed_sideband_hz: float = 100.0,
              f0_hz: float = 7500.0, drag: bool = False,
              pressure_mbar: float = 1e-6, simulate: bool = True,
              comsol_alpha: tuple[float, float] | None = None,
              title: str = "Dipole libration-sideband model",
              write_files: bool = True) -> dict:
    """Run the full model for a given effective command matrix M.

    Returns a dict with the decomposition, the SidebandSummary, the (optional)
    simulation results, and the paths written. ``plot_naive_vs_measured_acts.py``
    calls this directly with the assembled A_field's M.
    """
    decomp = corotating_decomposition(M)
    I = I if I is not None else moment_of_inertia(radius_um, density)
    summary = sideband_summary(decomp, d_emum, I, observed_sideband_hz, f0_hz,
                               comsol_alpha)

    # Gas drag: beta = I/tau. For the lock-loss study we set tau from pressure via
    # the molecular-flow k (Rider et al.: k ~ 4e-25 m^3 s for a 2.35um sphere).
    beta = 0.0
    if drag:
        k = 4.1e-25                          # m^3 s (paper value; order-of-mag)
        P_Pa = pressure_mbar * 100.0
        beta = k * P_Pa

    sims = {}
    if simulate:
        scale = summary.field_scale_SI_per_unit
        sims["naive"] = simulate_sideband_spectrum(
            M, summary.d_SI, I, scale, f0_hz, beta=beta)
        # Ideal circular field with |E+|_true radius (the corrected basis).
        M_circ = ellipse_to_M(decomp["abs_Eplus"], decomp["abs_Eplus"])
        sims["circular"] = simulate_sideband_spectrum(
            M_circ, summary.d_SI, I, scale, f0_hz, beta=beta)

    md = build_results_markdown(decomp, summary, sims.get("naive"), title, beta)
    print(md)

    written = {}
    if write_files:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / "naive_vs_measured_acts_results.md"
        md_path.write_text(md + "\n")
        written["markdown"] = md_path
        print(f"\n  wrote: {md_path}")
        if simulate and sims:
            ts = time.strftime("%Y%m%d_%H%M%S")
            png = out_dir / f"{ts}_sideband_spectrum.png"
            build_spectrum_figure(sims, summary, png, title)
            written["spectrum_png"] = png
            print(f"  wrote: {png}")

    return {"decomp": decomp, "summary": summary, "sims": sims, "written": written}


# --------------------------------------------------------------------------- #
# Standalone CLI
# --------------------------------------------------------------------------- #
def _M_from_config(config_path: str) -> tuple[np.ndarray, str]:
    """Assemble A_field from an ACTS config and build the naive command matrix M."""
    import yaml
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import upload_actuation_matrix as uam
    cfg = yaml.safe_load(Path(config_path).read_text())
    res = uam.assemble(cfg)
    M = effective_command_matrix(res["A_field"], res["elec_order"], res["dof_order"])
    files = ", ".join(Path(fg.path).parent.name for fg in res["file_gains"])
    return M, files


def _default_out_dir() -> Path:
    import os
    root = Path(os.path.expandvars("$MQG_DROPBOX_PATH/worker1/data"))
    return root / time.strftime("%y%m%d") / f"{time.strftime('%Y%m%d_%H%M%S')}_dipole_sideband_model"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("field source (choose one)")
    src.add_argument("--a-field-from-config",
                     help="ACTS inversion config; assemble A_field and build M")
    src.add_argument("--semi-major", type=float,
                     help="assumed-ellipse semi-major axis (no measurement)")
    src.add_argument("--semi-minor", type=float, help="assumed-ellipse semi-minor")
    src.add_argument("--tilt-deg", type=float, default=0.0,
                     help="assumed-ellipse tilt (deg, default 0)")
    src.add_argument("--eplus", type=float, help="give |E+| directly")
    src.add_argument("--eminus", type=float, help="give |E-| directly")

    par = ap.add_argument_group("particle / regime")
    par.add_argument("--d-emum", type=float, default=13.0)
    par.add_argument("--radius-um", type=float, default=1.5)
    par.add_argument("--density", type=float, default=3510.0)
    par.add_argument("--I", type=float, default=None,
                     help="moment of inertia (overrides radius/density)")
    par.add_argument("--observed-sideband-hz", type=float, default=100.0)
    par.add_argument("--f0-hz", type=float, default=7500.0)
    par.add_argument("--comsol-alpha", type=float, nargs=2, default=None,
                     metavar=("AX", "AY"),
                     help="explicit COMSOL per-axis field beliefs (override diag(M))")

    sim = ap.add_argument_group("simulation / drag")
    sim.add_argument("--simulate", dest="simulate", action="store_true", default=True)
    sim.add_argument("--no-simulate", dest="simulate", action="store_false")
    sim.add_argument("--drag", dest="drag", action="store_true", default=False)
    sim.add_argument("--no-drag", dest="drag", action="store_false")
    sim.add_argument("--pressure-mbar", type=float, default=1e-6)

    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    # Resolve the effective command matrix M.
    title = "Dipole libration-sideband model"
    if args.a_field_from_config:
        M, files = _M_from_config(args.a_field_from_config)
        title += f" -- {files}"
    elif args.eplus is not None:
        em = args.eminus if args.eminus is not None else 0.0
        M = ellipse_to_M(args.eplus + em, args.eplus - em, np.radians(args.tilt_deg))
        title += f" -- assumed |E+|={args.eplus}, |E-|={em}"
    elif args.semi_major is not None and args.semi_minor is not None:
        M = ellipse_to_M(args.semi_major, args.semi_minor, np.radians(args.tilt_deg))
        title += f" -- assumed ellipse {args.semi_major}x{args.semi_minor}"
    else:
        ap.error("provide --a-field-from-config, --semi-major/--semi-minor, or "
                 "--eplus[/--eminus]")

    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()
    run_model(M, out_dir, d_emum=args.d_emum, radius_um=args.radius_um,
              density=args.density, I=args.I,
              observed_sideband_hz=args.observed_sideband_hz, f0_hz=args.f0_hz,
              drag=args.drag, pressure_mbar=args.pressure_mbar,
              simulate=args.simulate,
              comsol_alpha=tuple(args.comsol_alpha) if args.comsol_alpha else None,
              title=title)


if __name__ == "__main__":
    main()
