"""Tests for the dipole libration-sideband model (dipole_sideband_model.py).

Pure logic + a couple of short numeric EOM integrations (no hardware). Covers:
co-rotating decomposition (E-=0 for a circle, semi-axes, shape factor), the four
d-inference reductions, the absolute anchor, and numeric-vs-analytic omega_phi.
"""
import math

import numpy as np
import pytest

import dipole_sideband_model as dsm


# --------------------------------------------------------------------------- #
# co-rotating decomposition
# --------------------------------------------------------------------------- #
def test_circle_has_no_counter_rotating_part():
    # An ideal isotropic diagonal M is a perfect circle: E- = 0, |E+| = radius.
    M = np.eye(2)
    d = dsm.corotating_decomposition(M)
    assert d["abs_Eminus"] == pytest.approx(0.0, abs=1e-12)
    assert d["abs_Eplus"] == pytest.approx(1.0, rel=1e-9)
    assert d["eps"] == pytest.approx(0.0, abs=1e-12)
    # Shape factor exactly 1 for a circle (mean radius == |E+|).
    assert d["shape_factor"] == pytest.approx(1.0, rel=1e-6)
    assert d["ripple"] == pytest.approx(0.0, abs=1e-6)


def test_semi_axes_and_meanE_geq_eplus_for_ellipse():
    # Axis-aligned ellipse a=1.5 (x), b=0.5 (y): |E+|=(a+b)/2, |E-|=(a-b)/2.
    M = np.diag([1.5, 0.5])
    d = dsm.corotating_decomposition(M)
    assert d["abs_Eplus"] == pytest.approx(1.0, rel=1e-9)
    assert d["abs_Eminus"] == pytest.approx(0.5, rel=1e-9)
    assert d["semi_major"] == pytest.approx(1.5, rel=1e-9)
    assert d["semi_minor"] == pytest.approx(0.5, rel=1e-9)
    # <|E|> >= |E+| always, with equality only for a circle.
    assert d["mean_radius"] >= d["abs_Eplus"] - 1e-9
    assert d["shape_factor"] >= 1.0 - 1e-9


def test_ellipse_to_M_roundtrips():
    # ellipse_to_M then decompose recovers the requested semi-axes.
    for a, b, tilt in [(1.45, 0.85, 0.0), (2.0, 0.5, 0.3), (1.0, 1.0, 0.0)]:
        M = dsm.ellipse_to_M(a, b, tilt)
        d = dsm.corotating_decomposition(M)
        assert d["semi_major"] == pytest.approx(max(a, b), rel=1e-6)
        assert d["semi_minor"] == pytest.approx(min(a, b), rel=1e-6)


def test_effective_command_matrix_from_A_field():
    # A_xy chosen so M = A_xy @ [c_x|c_y] is computable by hand. With electrodes
    # E1..E4 and the naive sign patterns, M columns are A_xy @ c_x and A_xy @ c_y.
    A_xy = np.array([[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0]])
    M = dsm.effective_command_matrix(A_xy, ["E1", "E2", "E3", "E4"], ["x", "y"])
    cx, cy = dsm.naive_command_vectors(["E1", "E2", "E3", "E4"])
    assert M[:, 0] == pytest.approx(A_xy @ cx)
    assert M[:, 1] == pytest.approx(A_xy @ cy)


# --------------------------------------------------------------------------- #
# d-inference error table
# --------------------------------------------------------------------------- #
def test_d_error_unity_for_circle():
    # Perfect circle: every belief reduction gives d_inferred/d_true = 1.
    M = np.eye(2)
    d = dsm.corotating_decomposition(M)
    ways = dsm.d_inference_error(d)
    for w in ways:
        assert w.d_ratio == pytest.approx(1.0, rel=1e-9)
        assert w.omega_phi_frac_err == pytest.approx(0.0, abs=1e-9)


def test_axis_aligned_ellipse_best_naive_is_exact():
    # Axis-aligned (no cross terms): |E+|_naive = |E+|_true, so way-4 ratio = 1
    # even for a strongly elliptical locus. The error then comes ONLY from
    # off-diagonal/curl coupling (covered below).
    M = np.diag([1.5, 0.5])
    d = dsm.corotating_decomposition(M)
    ways = {w.key: w for w in dsm.d_inference_error(d)}
    assert ways["best_naive"].d_ratio == pytest.approx(1.0, rel=1e-9)
    # ways 3 and 4 coincide when M_xx, M_yy share a sign.
    assert ways["average"].d_ratio == pytest.approx(ways["best_naive"].d_ratio, rel=1e-9)
    # x-only / y-only bracket it.
    assert ways["x_only"].d_ratio < 1.0 < ways["y_only"].d_ratio


def test_antisymmetric_cross_term_raises_eplus_true():
    # A curl-like antisymmetric M (M_yx = -M_xy) raises |E+|_true above the
    # diagonal-belief |E+|_naive -> best-naive over-estimates d (ratio > 1).
    M = np.array([[1.0, -0.4],
                  [0.4, 1.0]])
    d = dsm.corotating_decomposition(M)
    ways = {w.key: w for w in dsm.d_inference_error(d)}
    aEp_naive = abs(M[0, 0] + M[1, 1]) / 2.0
    assert d["abs_Eplus"] > aEp_naive
    assert ways["best_naive"].d_ratio > 1.0


def test_comsol_alpha_override():
    # Explicit COMSOL per-axis beliefs override diag(M).
    M = np.diag([1.5, 0.5])
    d = dsm.corotating_decomposition(M)
    ways = {w.key: w for w in dsm.d_inference_error(d, comsol_alpha=(2.0, 2.0))}
    assert ways["x_only"].E_believed == pytest.approx(2.0)
    assert ways["y_only"].E_believed == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# particle properties + anchor
# --------------------------------------------------------------------------- #
def test_moment_of_inertia_solid_sphere():
    I = dsm.moment_of_inertia(1.5, 3510.0)
    # (2/5) m r^2 with m = rho (4/3) pi r^3.
    r = 1.5e-6
    m = 3510.0 * (4 / 3) * math.pi * r ** 3
    assert I == pytest.approx(0.4 * m * r ** 2, rel=1e-12)


def test_anchor_sets_absolute_field_and_is_scale_free_for_d_ratio():
    M = np.array([[1.3745, -0.4675], [0.2661, 0.8022]])  # the real measured M
    d = dsm.corotating_decomposition(M)
    I = dsm.moment_of_inertia(1.5, 3510.0)
    s = dsm.sideband_summary(d, d_emum=13.0, I=I, observed_sideband_hz=100.0,
                             f0_hz=7500.0)
    # The anchor reproduces |E+|_true ~ 8.5 kV/m for 100 Hz with these params.
    assert s.E_true_SI == pytest.approx(8464.8, rel=1e-2)
    # d_ratio table is scale-free: scaling M by k leaves every ratio unchanged.
    d2 = dsm.corotating_decomposition(5.0 * M)
    s2 = dsm.sideband_summary(d2, d_emum=13.0, I=I, observed_sideband_hz=100.0,
                              f0_hz=7500.0)
    for w1, w2 in zip(s.ways, s2.ways):
        assert w1.d_ratio == pytest.approx(w2.d_ratio, rel=1e-9)


# --------------------------------------------------------------------------- #
# numeric EOM <-> analytic
# --------------------------------------------------------------------------- #
def test_numeric_omega_phi_matches_analytic_circle():
    # The numeric EOM sideband must reproduce sqrt(d|E+|/I) on a circular field.
    I = dsm.moment_of_inertia(1.5, 3510.0)
    d_SI = 13.0 * dsm.E_CHARGE * dsm.MICRON
    E_target = I * (2 * np.pi * 100.0) ** 2 / d_SI    # tuned for 100 Hz
    sim = dsm.simulate_sideband_spectrum(dsm.ellipse_to_M(1.0, 1.0), d_SI, I,
                                         E_target, f0_hz=7500.0, beta=0.0,
                                         freq_resolution_hz=1.0)
    ana = dsm.analytic_omega_phi_hz(1.0, E_target, d_SI, I)
    assert ana == pytest.approx(100.0, rel=1e-6)
    assert sim["omega_phi_hz"] == pytest.approx(ana, rel=0.05)


def test_numeric_sideband_governed_by_eplus_not_meanE():
    # For the real elliptical locus, the numeric sideband must track |E+| (the
    # circular-field value), NOT the larger <|E|>. We compare the naive-locus
    # sideband against an analytic |E+| prediction within a few percent.
    M = np.array([[1.3745, -0.4675], [0.2661, 0.8022]])
    d = dsm.corotating_decomposition(M)
    I = dsm.moment_of_inertia(1.5, 3510.0)
    s = dsm.sideband_summary(d, d_emum=13.0, I=I, observed_sideband_hz=100.0,
                             f0_hz=7500.0)
    sim = dsm.simulate_sideband_spectrum(M, s.d_SI, I, s.field_scale_SI_per_unit,
                                         f0_hz=7500.0, beta=0.0,
                                         freq_resolution_hz=1.0)
    ana_eplus = dsm.analytic_omega_phi_hz(d["abs_Eplus"], s.field_scale_SI_per_unit,
                                          s.d_SI, I)
    ana_meanE = dsm.analytic_omega_phi_hz(d["mean_radius"], s.field_scale_SI_per_unit,
                                          s.d_SI, I)
    # |E+| prediction is ~100 Hz; <|E|> prediction is ~1% higher. Numeric tracks E+.
    assert sim["omega_phi_hz"] == pytest.approx(ana_eplus, rel=0.05)
    assert abs(sim["omega_phi_hz"] - ana_eplus) < abs(sim["omega_phi_hz"] - ana_meanE) + 1.0
