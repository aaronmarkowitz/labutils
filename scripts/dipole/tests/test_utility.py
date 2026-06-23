"""Tests for the shared coordinate-system parser (utility.py).

This single parser is what makes the ACTS and SENSE uploaders provably use an
identical direction convention.
"""
import math

import numpy as np
import pytest

import utility as u


def test_axis_unit_vectors():
    assert np.allclose(u.axis_unit_vector("x"), [1, 0, 0])
    assert np.allclose(u.axis_unit_vector("y"), [0, 1, 0])
    assert np.allclose(u.axis_unit_vector("Z"), [0, 0, 1])


def test_axis_invalid():
    with pytest.raises(ValueError):
        u.axis_unit_vector("q")


@pytest.mark.parametrize("ang,expected", [
    (0, [1, 0, 0]),
    (45, [math.sqrt(0.5), math.sqrt(0.5), 0]),
    (90, [0, 1, 0]),
    (180, [-1, 0, 0]),
    (270, [0, -1, 0]),
])
def test_angle_deg_in_plane(ang, expected):
    n = u.direction_unit_vector({"angle_deg": ang})
    assert np.allclose(n, expected, atol=1e-12)


def test_spherical_elevation_azimuth():
    n = u.direction_unit_vector({"elevation_deg": 45, "azimuth_deg": 60})
    el, az = math.radians(45), math.radians(60)
    assert np.allclose(n, [math.cos(el) * math.cos(az),
                           math.cos(el) * math.sin(az),
                           math.sin(el)])
    assert abs(np.linalg.norm(n) - 1.0) < 1e-12


def test_elevation_90_is_z():
    n = u.direction_unit_vector({"elevation_deg": 90})
    assert np.allclose(n, [0, 0, 1], atol=1e-12)


def test_vector_is_normalized():
    n = u.direction_unit_vector({"vector": [3, 4, 0]})
    assert np.allclose(n, [0.6, 0.8, 0.0])
    assert abs(np.linalg.norm(n) - 1.0) < 1e-12


def test_vector_two_component_pads_z():
    n = u.direction_unit_vector({"vector": [1, 0]})
    assert np.allclose(n, [1, 0, 0])


def test_azimuth_180_flips_sign():
    a = u.direction_unit_vector({"angle_deg": 45})
    b = u.direction_unit_vector({"angle_deg": 225})
    assert np.allclose(a, -b, atol=1e-12)


def test_mode_shorthand_matches_axis():
    assert np.allclose(u.direction_unit_vector({"mode": "y"}), [0, 1, 0])
    assert np.allclose(u.direction_unit_vector({"axis": "x"}), [1, 0, 0])


def test_overspecified_raises():
    with pytest.raises(ValueError, match="over-specified"):
        u.direction_unit_vector({"angle_deg": 0, "vector": [1, 0, 0]})
    with pytest.raises(ValueError, match="over-specified"):
        u.direction_unit_vector({"mode": "x", "angle_deg": 0})


def test_underspecified_raises():
    with pytest.raises(ValueError, match="must provide"):
        u.direction_unit_vector({"gain": 1.0})


def test_zero_vector_raises():
    with pytest.raises(ValueError, match="zero length"):
        u.direction_unit_vector({"vector": [0, 0, 0]})


def test_select_dofs_orders_components():
    n3 = np.array([0.1, 0.2, 0.3])
    assert np.allclose(u.select_dofs(n3, ["x", "y"]), [0.1, 0.2])
    assert np.allclose(u.select_dofs(n3, ["y", "x", "z"]), [0.2, 0.1, 0.3])


def test_out_of_subspace_fraction():
    # In-plane direction has ~0 outside x,y
    n = u.direction_unit_vector({"angle_deg": 30})
    assert u.out_of_subspace_fraction(n, ["x", "y"]) < 1e-9
    # Pure z has ~all of itself outside x,y
    nz = u.axis_unit_vector("z")
    assert abs(u.out_of_subspace_fraction(nz, ["x", "y"]) - 1.0) < 1e-9
    # 45 deg elevation -> sin(45) outside the x,y plane
    ne = u.direction_unit_vector({"elevation_deg": 45})
    assert abs(u.out_of_subspace_fraction(ne, ["x", "y"]) - math.sin(math.radians(45))) < 1e-9
