"""Tests for the generalized (directional) SENSE uploader.

Covers the n.W row math, axis back-compatibility, full-sphere directions, and
DOF-subspace validation. Uses small synthetic W matrices plus a regression
check against the real step-01 file if present.
"""
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest

import upload_sense_matrix as usm


def _hdf5(W, channel_names, dofs):
    return {"W": np.asarray(W, float), "channel_names": list(channel_names),
            "dofs": list(dofs), "peak_hz": {}, "eigenratio": {}}


def _cfg(rows, cols=None):
    if cols is None:
        cols = [
            {"index": 1, "label": "C0", "channel_suffix": "CH0"},
            {"index": 2, "label": "C1", "channel_suffix": "CH1"},
        ]
    return {"prefix": "Y1:DMD", "matrix_name": "SENSE", "rows": rows, "cols": cols}


W_XY = [[1.0, 2.0],    # x row
        [3.0, 4.0]]    # y row
CH = ["Y1:DMD-CH0_IN1", "Y1:DMD-CH1_IN1"]


def test_pure_axis_reproduces_w_row():
    hdf5 = _hdf5(W_XY, CH, ["x", "y"])
    cfg = _cfg([{"index": 1, "label": "XERR", "mode": "x"}])
    entries, skipped = usm.build_mapping(hdf5, cfg)
    vals = {e["col_label"]: e["value"] for e in entries}
    assert math.isclose(vals["C0"], 1.0) and math.isclose(vals["C1"], 2.0)
    assert skipped == []


def test_y_axis_row():
    hdf5 = _hdf5(W_XY, CH, ["x", "y"])
    cfg = _cfg([{"index": 2, "label": "YERR", "mode": "y"}])
    entries, _ = usm.build_mapping(hdf5, cfg)
    vals = {e["col_label"]: e["value"] for e in entries}
    assert math.isclose(vals["C0"], 3.0) and math.isclose(vals["C1"], 4.0)


def test_angle_45_is_combination():
    hdf5 = _hdf5(W_XY, CH, ["x", "y"])
    cfg = _cfg([{"index": 4, "label": "D45", "angle_deg": 45}])
    entries, _ = usm.build_mapping(hdf5, cfg)
    vals = {e["col_label"]: e["value"] for e in entries}
    s = math.sqrt(0.5)
    assert math.isclose(vals["C0"], s * 1.0 + s * 3.0)
    assert math.isclose(vals["C1"], s * 2.0 + s * 4.0)


def test_angle_225_flips_sign():
    hdf5 = _hdf5(W_XY, CH, ["x", "y"])
    e45, _ = usm.build_mapping(hdf5, _cfg([{"index": 4, "label": "D", "angle_deg": 45}]))
    e225, _ = usm.build_mapping(hdf5, _cfg([{"index": 4, "label": "D", "angle_deg": 225}]))
    for a, b in zip(e45, e225):
        assert math.isclose(a["value"], -b["value"], abs_tol=1e-12)


def test_full_sphere_combines_three_rows():
    W = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]  # x, y, z rows
    ch = ["Y1:DMD-CH0_IN1", "Y1:DMD-CH1_IN1"]
    hdf5 = _hdf5(W, ch, ["x", "y", "z"])
    cfg = _cfg([{"index": 5, "label": "SKEW", "elevation_deg": 45, "azimuth_deg": 60}])
    entries, _ = usm.build_mapping(hdf5, cfg)
    el, az = math.radians(45), math.radians(60)
    n = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    Wn = np.asarray(W)
    expected = n @ Wn
    vals = {e["col_label"]: e["value"] for e in entries}
    assert math.isclose(vals["C0"], expected[0])
    assert math.isclose(vals["C1"], expected[1])


def test_pure_axis_absent_dof_skipped():
    # z axis row on an x,y-only diagonalization -> skipped, not error.
    hdf5 = _hdf5(W_XY, CH, ["x", "y"])
    cfg = _cfg([{"index": 1, "label": "XERR", "mode": "x"},
                {"index": 3, "label": "ZERR", "mode": "z"}])
    entries, skipped = usm.build_mapping(hdf5, cfg)
    assert len(skipped) == 1 and skipped[0]["label"] == "ZERR"
    assert all(e["row_label"] != "ZERR" for e in entries)


def test_explicit_z_direction_absent_dof_raises():
    # explicit elevation pointing out of x,y plane -> hard error.
    hdf5 = _hdf5(W_XY, CH, ["x", "y"])
    cfg = _cfg([{"index": 4, "label": "TILT", "elevation_deg": 30}])
    with pytest.raises(ValueError, match="outside the measured DOFs"):
        usm.build_mapping(hdf5, cfg)


def test_unused_column_is_zero():
    hdf5 = _hdf5(W_XY, CH, ["x", "y"])
    cols = [
        {"index": 1, "label": "C0", "channel_suffix": "CH0"},
        {"index": 2, "label": "C1", "channel_suffix": "CH1"},
        {"index": 3, "label": "SPARE", "channel_suffix": "NOPE"},
    ]
    cfg = _cfg([{"index": 1, "label": "XERR", "mode": "x"}], cols=cols)
    entries, _ = usm.build_mapping(hdf5, cfg)
    spare = next(e for e in entries if e["col_label"] == "SPARE")
    assert spare["value"] == 0.0 and spare["w_col"] is None


# --------------------------------------------------------------------------- #
# Regression against the real step-01 file (skipped if unavailable)
# --------------------------------------------------------------------------- #
REAL_H5 = Path("/home/controls/Dropbox/Microspheres/MAST-QG/worker1/data/260622/"
               "step01_4ch_265avg/01_SensorDiagonalization/"
               "step_01_sensor_diagonalization_results.h5")


@pytest.mark.skipif(not REAL_H5.exists(), reason="real step-01 file not present")
def test_regression_real_w_axis_rows():
    hdf5 = usm.load_hdf5(REAL_H5)
    W = hdf5["W"]
    cfg = usm.load_config(Path(usm.__file__).parent / "sense_matrix_config.yml")
    entries, _ = usm.build_mapping(hdf5, cfg)
    # For XERR (mode x), each written column value must equal W[0, w_col].
    for e in entries:
        if e["row_label"] == "XERR" and e["w_col"] is not None:
            assert math.isclose(e["value"], float(W[0, e["w_col"]]), rel_tol=1e-9)
