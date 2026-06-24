"""Tests for the ACTS actuation-matrix inversion (upload_actuation_matrix.py).

All math is exercised via the pure functions; EPICS writes are covered through
the write-planner / dry-run path (no hardware).
"""
import json
import math

import h5py
import numpy as np
import pytest

import upload_actuation_matrix as uam


# --------------------------------------------------------------------------- #
# signed_magnitude / lossy_phase
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("z,expected_sign", [
    (1.0 + 0j, +1),
    (-1.0 + 0j, -1),
    (1.0 * np.exp(1j * math.radians(10)), +1),
    (1.0 * np.exp(1j * math.radians(170)), -1),
    (1.0 * np.exp(1j * math.radians(-175)), -1),
])
def test_signed_magnitude_sign(z, expected_sign):
    r = uam.signed_magnitude(z)
    assert math.isclose(abs(r), abs(z), rel_tol=1e-9)
    assert (r > 0) == (expected_sign > 0)


def test_lossy_phase_flags_near_90():
    assert uam.lossy_phase(1.0 * np.exp(1j * math.radians(80)))
    assert not uam.lossy_phase(1.0 * np.exp(1j * math.radians(10)))
    assert not uam.lossy_phase(1.0 * np.exp(1j * math.radians(175)))


# --------------------------------------------------------------------------- #
# gamma pooling
# --------------------------------------------------------------------------- #
def test_pool_gamma_weighted_mean():
    f0 = {"x": 40.0, "y": 80.0}
    Q = {"x": 20.0, "y": 20.0}   # gamma_x = 2.0, gamma_y = 4.0
    res = {"x": 1.0, "y": 1.0}   # equal weights -> mean 3.0
    g, sig, per = uam.pool_gamma(f0, Q, res, ["x", "y"])
    assert math.isclose(g, 3.0, rel_tol=1e-9)
    assert math.isclose(per["x"], 2.0) and math.isclose(per["y"], 4.0)
    assert sig > 0  # the two modes disagree


def test_pool_gamma_residual_weighting():
    f0 = {"x": 40.0, "y": 80.0}
    Q = {"x": 20.0, "y": 20.0}   # gamma_x=2, gamma_y=4
    res = {"x": 0.01, "y": 1.0}  # x far better -> pooled near 2.0
    g, sig, per = uam.pool_gamma(f0, Q, res, ["x", "y"])
    assert g < 2.5


def test_pool_gamma_single_mode_zero_sigma():
    g, sig, per = uam.pool_gamma({"x": 40.0}, {"x": 20.0}, {"x": 1.0}, ["x"])
    assert math.isclose(g, 2.0) and sig == 0.0


# --------------------------------------------------------------------------- #
# forward matrix assembly (N files, any layout)
# --------------------------------------------------------------------------- #
def _fg(electrodes, dof_order, gains, f0=None, Q=None, res=None, coh=0.99, path="f"):
    """Construct a FileGains for tests. gains shape (n_dof, n_elec) complex."""
    from pathlib import Path
    f0 = f0 or {d: 40.0 + 10 * i for i, d in enumerate(dof_order)}
    Q = Q or {d: 20.0 for d in dof_order}
    res = res or {d: 1.0 for d in dof_order}
    coherence = {d: np.full(4, coh) for d in dof_order}
    return uam.FileGains(path=Path(path), electrodes=list(electrodes),
                         dof_order=list(dof_order), gain=np.asarray(gains, complex),
                         f0=f0, Q=Q, residual=res, coherence=coherence)


# Known forward matrix used across layout tests (rows x,y; cols E1..E4).
A_TRUE = np.array([
    [1.0, -0.5, -0.8, 0.9],
    [0.7, 0.6, -0.4, -0.3],
])


def test_forward_matrix_four_single_electrode_files():
    files = [_fg([f"E{j+1}"], ["x", "y"], A_TRUE[:, j:j+1]) for j in range(4)]
    A, cells = uam.build_forward_matrix(files, ["x", "y"], ["E1", "E2", "E3", "E4"])
    assert np.allclose(A, A_TRUE)


def test_forward_matrix_one_four_electrode_file():
    files = [_fg(["E1", "E2", "E3", "E4"], ["x", "y"], A_TRUE)]
    A, _ = uam.build_forward_matrix(files, ["x", "y"], ["E1", "E2", "E3", "E4"])
    assert np.allclose(A, A_TRUE)


def test_forward_matrix_file_order_irrelevant():
    files = [_fg([f"E{j+1}"], ["x", "y"], A_TRUE[:, j:j+1]) for j in (2, 0, 3, 1)]
    A, _ = uam.build_forward_matrix(files, ["x", "y"], ["E1", "E2", "E3", "E4"])
    assert np.allclose(A, A_TRUE)


def test_forward_matrix_one_dof_per_file():
    # x for all electrodes in one file, y in another (single-DOF files).
    fx = _fg(["E1", "E2", "E3", "E4"], ["x"], A_TRUE[0:1, :])
    fy = _fg(["E1", "E2", "E3", "E4"], ["y"], A_TRUE[1:2, :])
    A, _ = uam.build_forward_matrix([fx, fy], ["x", "y"], ["E1", "E2", "E3", "E4"])
    assert np.allclose(A, A_TRUE)


def test_forward_matrix_missing_cell_raises():
    files = [_fg([f"E{j+1}"], ["x", "y"], A_TRUE[:, j:j+1]) for j in range(3)]
    with pytest.raises(ValueError, match="missing measurements"):
        uam.build_forward_matrix(files, ["x", "y"], ["E1", "E2", "E3", "E4"])


def test_forward_matrix_duplicate_error():
    files = [_fg(["E1"], ["x", "y"], A_TRUE[:, 0:1]),
             _fg(["E1"], ["x", "y"], A_TRUE[:, 0:1] * 1.1)]
    with pytest.raises(ValueError, match="duplicate"):
        uam.build_forward_matrix(files, ["x", "y"], ["E1"], duplicate="error")


def test_forward_matrix_duplicate_average():
    files = [_fg(["E1"], ["x"], np.array([[2.0]])),
             _fg(["E1"], ["x"], np.array([[4.0]]))]
    A, _ = uam.build_forward_matrix(files, ["x"], ["E1"], duplicate="average")
    assert math.isclose(A[0, 0], 3.0)


# --------------------------------------------------------------------------- #
# field normalization
# --------------------------------------------------------------------------- #
def test_field_scale_common_gamma():
    fg = _fg(["E1"], ["x", "y"], A_TRUE[:, 0:1],
             f0={"x": 40.0, "y": 56.0}, Q={"x": 20.0, "y": 20.0})
    fg.gamma = 2.0
    s = uam.field_scale_factors(fg, "common_gamma")
    assert math.isclose(s["x"], 40.0 * 2.0) and math.isclose(s["y"], 56.0 * 2.0)


def test_field_scale_per_mode_q():
    fg = _fg(["E1"], ["x", "y"], A_TRUE[:, 0:1],
             f0={"x": 40.0, "y": 80.0}, Q={"x": 20.0, "y": 40.0})
    s = uam.field_scale_factors(fg, "per_mode_q")
    assert math.isclose(s["x"], 40.0 ** 2 / 20.0)
    assert math.isclose(s["y"], 80.0 ** 2 / 40.0)


def test_field_scale_none():
    fg = _fg(["E1"], ["x", "y"], A_TRUE[:, 0:1])
    s = uam.field_scale_factors(fg, "none")
    assert s == {"x": 1.0, "y": 1.0}


def test_field_normalize_ratio_within_file():
    # Within one file, common_gamma x:y scale ratio is f0_y/f0_x (gamma cancels).
    fg = _fg(["E1", "E2", "E3", "E4"], ["x", "y"], A_TRUE,
             f0={"x": 40.0, "y": 56.0}, Q={"x": 20.0, "y": 20.0})
    fg.gamma = 2.5
    cells = {}
    A, cells = uam.build_forward_matrix([fg], ["x", "y"], ["E1", "E2", "E3", "E4"])
    # pool gamma so apply_field_normalization sees it
    g, gs, _ = uam.pool_gamma(fg.f0, fg.Q, fg.residual, fg.dof_order)
    fg.gamma, fg.gamma_sigma = g, gs
    A_field, s = uam.apply_field_normalization(A, ["x", "y"], ["E1", "E2", "E3", "E4"],
                                               cells, [fg], "common_gamma")
    ratio = s[("y", "E1")] / s[("x", "E1")]
    assert math.isclose(ratio, 56.0 / 40.0, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# global field anchor (fixes the magnitude unit of gain=1)
# --------------------------------------------------------------------------- #
def test_anchor_scale_frobenius():
    assert math.isclose(uam.anchor_scale(A_TRUE, "frobenius"),
                        float(np.linalg.norm(A_TRUE)))


def test_anchor_scale_sigma_max():
    assert math.isclose(uam.anchor_scale(A_TRUE, "sigma_max"),
                        float(np.linalg.svd(A_TRUE, compute_uv=False)[0]))


def test_anchor_scale_reference_column():
    assert math.isclose(uam.anchor_scale(A_TRUE, "reference_column", ref_col=1),
                        float(np.linalg.norm(A_TRUE[:, 1])))
    with pytest.raises(ValueError, match="out of range"):
        uam.anchor_scale(A_TRUE, "reference_column", ref_col=9)


def test_anchor_scale_unknown_functional_raises():
    with pytest.raises(ValueError, match="unknown anchor functional"):
        uam.anchor_scale(A_TRUE, "bogus")


def test_anchor_scale_degenerate_returns_one():
    assert uam.anchor_scale(np.zeros((2, 4)), "frobenius") == 1.0
    assert uam.anchor_scale(np.zeros((2, 4)), "sigma_max") == 1.0


@pytest.mark.parametrize("functional", ["frobenius", "sigma_max"])
def test_anchor_invariance_under_global_scalar(functional):
    # THE core property: A_field and k*A_field (different particle's global c)
    # give the SAME anchored matrix, so written GAINs are particle-independent.
    k = 7.3
    a1 = A_TRUE / uam.anchor_scale(A_TRUE, functional)
    a2 = (k * A_TRUE) / uam.anchor_scale(k * A_TRUE, functional)
    assert np.allclose(a1, a2)


def test_anchor_preserves_direction_scales_magnitude():
    # Anchoring scales electrode counts but not the realized field direction.
    anchored = A_TRUE / uam.anchor_scale(A_TRUE, "frobenius")
    u = np.array([1.0, 0.0])
    v_raw = uam.column_electrode_values(np.linalg.pinv(A_TRUE), u, 1.0)
    v_anc = uam.column_electrode_values(np.linalg.pinv(anchored), u, 1.0)
    # same direction in electrode space (parallel count vectors)
    assert np.allclose(v_anc / np.linalg.norm(v_anc),
                       v_raw / np.linalg.norm(v_raw), atol=1e-9)
    # realized field still points along u for both
    assert np.allclose(anchored @ v_anc, u, atol=1e-9)


# --------------------------------------------------------------------------- #
# inversion / column values
# --------------------------------------------------------------------------- #
def test_unit_response_inversion():
    A_pinv = np.linalg.pinv(A_TRUE)
    for ang in range(0, 360, 15):
        u_hat = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
        v = uam.column_electrode_values(A_pinv, u_hat, 1.0)
        assert np.allclose(A_TRUE @ v, u_hat, atol=1e-9)


def test_per_column_gain_scales_linearly():
    A_pinv = np.linalg.pinv(A_TRUE)
    u_hat = np.array([1.0, 0.0])
    v1 = uam.column_electrode_values(A_pinv, u_hat, 1.0)
    v2 = uam.column_electrode_values(A_pinv, u_hat, 2.0)
    assert np.allclose(v2, 2 * v1)
    v0 = uam.column_electrode_values(A_pinv, u_hat, 0.0)
    assert np.allclose(v0, 0.0)


def test_rotating_field_orthogonal_equal_magnitude():
    A_pinv = np.linalg.pinv(A_TRUE)
    loc = uam.column_electrode_values(A_pinv, np.array([1.0, 0.0]), 1.0)
    los = uam.column_electrode_values(A_pinv, np.array([0.0, 1.0]), 1.0)
    fx = A_TRUE @ loc
    fy = A_TRUE @ los
    assert np.allclose(fx, [1, 0], atol=1e-9)
    assert np.allclose(fy, [0, 1], atol=1e-9)
    assert math.isclose(np.linalg.norm(fx), np.linalg.norm(fy))


# --------------------------------------------------------------------------- #
# uncertainty propagation
# --------------------------------------------------------------------------- #
def test_zero_uncertainty_gives_zero_sigma():
    A_field = A_TRUE.copy()
    A_sigma = np.zeros_like(A_field)
    s = uam.propagate_column_sigma(A_field, A_sigma, np.array([1.0, 0.0]), 1.0)
    assert np.allclose(s, 0.0)


def test_uncertainty_grows_with_input_sigma():
    A_field = A_TRUE.copy()
    small = uam.propagate_column_sigma(A_field, 0.01 * np.abs(A_field),
                                       np.array([1.0, 0.0]), 1.0)
    big = uam.propagate_column_sigma(A_field, 0.1 * np.abs(A_field),
                                     np.array([1.0, 0.0]), 1.0)
    assert np.all(big >= small)
    assert np.any(big > 0)


def test_gain_rel_error_monotonic():
    hi = uam.gain_rel_error(0.99, 30)
    lo = uam.gain_rel_error(0.5, 30)
    assert lo > hi  # lower coherence -> larger relative error


# --------------------------------------------------------------------------- #
# planning: coupled/uncoupled, channel strings
# --------------------------------------------------------------------------- #
def _plan_cfg(columns):
    return {
        "prefix": "Y1:DMD",
        "electrode_row": {"E1": 1, "E2": 2, "E3": 3, "E4": 4},
        "columns": columns,
    }


def test_plan_uncoupled_writes_nothing():
    cfg = _plan_cfg([{"index": 1, "label": "XCTL", "coupled": False}])
    A_pinv = np.linalg.pinv(A_TRUE)
    plans = uam.plan_columns(cfg, A_pinv, ["x", "y"], ["E1", "E2", "E3", "E4"],
                             {}, strict_subspace=False)
    assert plans[0].coupled is False
    assert plans[0].electrode_values == {}


def test_plan_coupled_channel_strings():
    cfg = _plan_cfg([{"index": 4, "label": "LOC", "coupled": True, "angle_deg": 0}])
    A_pinv = np.linalg.pinv(A_TRUE)
    plans = uam.plan_columns(cfg, A_pinv, ["x", "y"], ["E1", "E2", "E3", "E4"],
                             {}, strict_subspace=False)
    p = plans[0]
    assert set(p.electrode_values) == {"E1", "E2", "E3", "E4"}
    # exact ACTS channel for E3 row in column 4
    assert uam.acts_base("Y1:DMD", 3, p.index) == "Y1:DMD-ACTS_3_4"


def test_plan_z_direction_projected_with_note():
    cfg = _plan_cfg([{"index": 4, "label": "LOC", "coupled": True,
                      "elevation_deg": 45}])
    A_pinv = np.linalg.pinv(A_TRUE)
    plans = uam.plan_columns(cfg, A_pinv, ["x", "y"], ["E1", "E2", "E3", "E4"],
                             {}, strict_subspace=False)
    assert any("outside measured DOFs" in n for n in plans[0].notes)


def test_plan_z_direction_strict_raises():
    cfg = _plan_cfg([{"index": 4, "label": "LOC", "coupled": True,
                      "elevation_deg": 45}])
    A_pinv = np.linalg.pinv(A_TRUE)
    with pytest.raises(ValueError, match="strict"):
        uam.plan_columns(cfg, A_pinv, ["x", "y"], ["E1", "E2", "E3", "E4"],
                         {}, strict_subspace=True)


# --------------------------------------------------------------------------- #
# end-to-end via HDF5 fixtures + dry-run write planner
# --------------------------------------------------------------------------- #
def _write_result_h5(path, electrode, dof_order, gains, f0, Q):
    with h5py.File(path, "w") as f:
        g = np.asarray(gains, complex)
        f.create_dataset("gain_matrix_real", data=g.real)
        f.create_dataset("gain_matrix_imag", data=g.imag)
        f.attrs["dof_order"] = json.dumps(dof_order)
        f.attrs["electrodes"] = json.dumps([electrode])
        for d in dof_order:
            f.attrs[f"peak_frequency_hz_{d}"] = f0[d]
            f.attrs[f"Q_{d}"] = Q[d]
            f.attrs[f"residual_norm_{d}"] = 0.01
            f.create_dataset(f"coherence_{d}", data=np.full(4, 0.95))


def test_assemble_end_to_end(tmp_path):
    f0 = {"x": 41.0, "y": 56.0}
    Q = {"x": 18.0, "y": 18.0}
    data = []
    for j in range(4):
        p = tmp_path / f"E{j+1}.h5"
        _write_result_h5(p, f"E{j+1}", ["x", "y"], A_TRUE[:, j:j+1], f0, Q)
        data.append({"electrode": f"E{j+1}", "path": str(p)})
    cfg = {
        "prefix": "Y1:DMD",
        "electrode_row": {"E1": 1, "E2": 2, "E3": 3, "E4": 4},
        "electrodes": ["E1", "E2", "E3", "E4"],
        "dofs": ["x", "y"],
        "data": data,
        "field_normalize": "common_gamma",
        "columns": [
            {"index": 1, "label": "XCTL", "coupled": True, "angle_deg": 0},
            {"index": 2, "label": "YCTL", "coupled": True, "angle_deg": 90},
            {"index": 3, "label": "ZCTL", "coupled": False},
        ],
    }
    res = uam.assemble(cfg)
    assert np.allclose(res["A"], A_TRUE)
    # XCTL realizes [1,0], YCTL realizes [0,1] (field-normalized space)
    xctl = next(p for p in res["plans"] if p.label == "XCTL")
    v = np.array([xctl.electrode_values[e] for e in ["E1", "E2", "E3", "E4"]])
    assert np.allclose(res["A_field"] @ v, [1, 0], atol=1e-9)
    # anchor metadata present; default self_norm/frobenius applied
    assert res["anchor_mode"] == "self_norm"
    assert res["anchor_functional"] == "frobenius"
    assert res["anchor"] > 0


def _assemble_cfg(tmp_path, gains_by_elec, f0, Q, field_anchor=None):
    data = []
    for j in range(4):
        e = f"E{j+1}"
        p = tmp_path / f"{e}.h5"
        _write_result_h5(p, e, ["x", "y"], gains_by_elec[:, j:j+1], f0, Q)
        data.append({"electrode": e, "path": str(p)})
    cfg = {
        "prefix": "Y1:DMD",
        "electrode_row": {"E1": 1, "E2": 2, "E3": 3, "E4": 4},
        "electrodes": ["E1", "E2", "E3", "E4"],
        "dofs": ["x", "y"],
        "data": data,
        "field_normalize": "common_gamma",
        "columns": [{"index": 1, "label": "XCTL", "coupled": True, "angle_deg": 0}],
    }
    if field_anchor is not None:
        cfg["field_anchor"] = field_anchor
    return cfg


def test_assemble_anchor_makes_gains_particle_independent(tmp_path):
    # Two "particles": identical electrodes (same B) but a global scalar k between
    # them (different cal*q/m). With self_norm, the written GAINs must match.
    f0 = {"x": 41.0, "y": 56.0}
    Q = {"x": 18.0, "y": 18.0}
    k = 4.2
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    cfg_a = _assemble_cfg(tmp_path / "a", A_TRUE, f0, Q)
    cfg_b = _assemble_cfg(tmp_path / "b", k * A_TRUE, f0, Q)
    res_a = uam.assemble(cfg_a)
    res_b = uam.assemble(cfg_b)
    va = res_a["plans"][0].electrode_values
    vb = res_b["plans"][0].electrode_values
    for e in ["E1", "E2", "E3", "E4"]:
        assert math.isclose(va[e], vb[e], rel_tol=1e-9, abs_tol=1e-12)


def test_assemble_anchor_none_is_drift_prone(tmp_path):
    # mode: none reproduces the old behaviour -> a global scalar k DOES change
    # the written GAINs (regression guard for the new default's effect).
    f0 = {"x": 41.0, "y": 56.0}
    Q = {"x": 18.0, "y": 18.0}
    k = 4.2
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    cfg_a = _assemble_cfg(tmp_path / "a", A_TRUE, f0, Q, field_anchor={"mode": "none"})
    cfg_b = _assemble_cfg(tmp_path / "b", k * A_TRUE, f0, Q, field_anchor={"mode": "none"})
    va = uam.assemble(cfg_a)["plans"][0].electrode_values
    vb = uam.assemble(cfg_b)["plans"][0].electrode_values
    # counts scale ~1/k between the two -> NOT equal
    assert not math.isclose(va["E1"], vb["E1"], rel_tol=1e-6)


def test_assemble_physical_scale_scales_counts(tmp_path):
    # physical_scale P scales electrode counts linearly vs self_norm (P=1).
    f0 = {"x": 41.0, "y": 56.0}
    Q = {"x": 18.0, "y": 18.0}
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    cfg1 = _assemble_cfg(tmp_path / "a", A_TRUE, f0, Q,
                         field_anchor={"mode": "self_norm", "physical_scale": 1.0})
    cfg2 = _assemble_cfg(tmp_path / "b", A_TRUE, f0, Q,
                         field_anchor={"mode": "physical", "physical_scale": 10.0})
    v1 = uam.assemble(cfg1)["plans"][0].electrode_values
    v2 = uam.assemble(cfg2)["plans"][0].electrode_values
    # A_field scales by P -> pinv scales by 1/P -> counts scale by 1/P
    for e in ["E1", "E2", "E3", "E4"]:
        assert math.isclose(v2[e], v1[e] / 10.0, rel_tol=1e-9, abs_tol=1e-12)


def test_electrode_cross_check_mismatch_raises(tmp_path):
    p = tmp_path / "E1.h5"
    _write_result_h5(p, "E1", ["x", "y"], A_TRUE[:, 0:1],
                     {"x": 41.0, "y": 56.0}, {"x": 18.0, "y": 18.0})
    cfg = {
        "prefix": "Y1:DMD",
        "electrode_row": {"E1": 1},
        "electrodes": ["E1"],
        "dofs": ["x", "y"],
        "data": [{"electrode": "E2", "path": str(p)}],  # wrong label
        "columns": [{"index": 1, "label": "XCTL", "coupled": False}],
    }
    with pytest.raises(ValueError, match="declares electrode"):
        uam.assemble(cfg)
