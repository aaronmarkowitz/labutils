#!/usr/bin/env python3
"""Shared coordinate-system utilities for the dipole ACTS / SENSE upload scripts.

Both ``upload_actuation_matrix.py`` (writes the ACTS actuation matrix) and
``upload_sense_matrix.py`` (writes the SENSE readout matrix) need to turn a
human-friendly *direction* specification into a unit vector in the particle
DOF basis (x, y, z). Keeping that parsing in one place guarantees the two
scripts use an *identical*, unit-tested convention — a 45 deg field driven by
ACTS points the same way as a 45 deg readout configured in SENSE.

Coordinate convention
----------------------
Right-handed (x, y, z). A direction is a 3-D unit vector ``n = [nx, ny, nz]``.

  * azimuth  (``azimuth_deg``)   : angle in the x-y plane, measured from +x
                                   toward +y. 0 deg -> +x, 90 deg -> +y.
  * elevation(``elevation_deg``) : angle above the x-y plane toward +z.
                                   0 deg -> in plane, +90 deg -> +z.

      n = [cos(el)cos(az), cos(el)sin(az), sin(el)]

A pure in-plane angle (``angle_deg``) is the special case ``elevation_deg = 0``,
i.e. ``angle_deg`` is an azimuth. Axis shorthands ("x"/"y"/"z") map to the
corresponding unit axis. An explicit ``vector: [x, y, z]`` is normalized as-is.

Direction specification (any one of the following)
--------------------------------------------------
The parser accepts a dict (typically one YAML row/column entry). It looks for
exactly one of these forms, in priority order:

  1. ``vector: [x, y, z]``           explicit components (z optional -> 0)
  2. ``elevation_deg`` and/or ``azimuth_deg``   full-sphere spherical angles
  3. ``angle_deg``                   in-plane azimuth (elevation 0)
  4. ``mode``/``axis``: "x"|"y"|"z"  axis shorthand

Sign flips: add 180 to ``azimuth_deg`` (or ``angle_deg``), or negate a
``gain``/scale downstream — both produce the opposite direction.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def axis_unit_vector(axis: str) -> np.ndarray:
    """Return the unit vector for axis shorthand 'x', 'y', or 'z'."""
    a = str(axis).strip().lower()
    if a not in _AXIS_INDEX:
        raise ValueError(f"axis must be one of 'x','y','z'; got {axis!r}")
    n = np.zeros(3)
    n[_AXIS_INDEX[a]] = 1.0
    return n


def spherical_unit_vector(elevation_deg: float = 0.0,
                          azimuth_deg: float = 0.0) -> np.ndarray:
    """Unit vector from elevation (above x-y plane) and azimuth (from +x toward +y)."""
    el = math.radians(float(elevation_deg))
    az = math.radians(float(azimuth_deg))
    return np.array([
        math.cos(el) * math.cos(az),
        math.cos(el) * math.sin(az),
        math.sin(el),
    ])


def normalize(vec: Sequence[float]) -> np.ndarray:
    """Return ``vec`` as a length-3 unit vector (pads missing z with 0)."""
    v = np.asarray(vec, dtype=float).ravel()
    if v.size == 2:
        v = np.array([v[0], v[1], 0.0])
    elif v.size == 3:
        v = v.astype(float)
    else:
        raise ValueError(f"vector must have 2 or 3 components; got {v.size}")
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("direction vector has (near) zero length; cannot normalize")
    return v / norm


def direction_unit_vector(spec: dict) -> np.ndarray:
    """Parse a direction specification dict into a 3-D unit vector.

    Accepts exactly one of (priority order): ``vector``,
    ``elevation_deg``/``azimuth_deg``, ``angle_deg``, or ``mode``/``axis``.
    See module docstring for the coordinate convention.
    """
    if not isinstance(spec, dict):
        raise TypeError(f"direction spec must be a dict; got {type(spec).__name__}")

    has_vector = spec.get("vector") is not None
    has_el = spec.get("elevation_deg") is not None
    has_az = spec.get("azimuth_deg") is not None
    has_angle = spec.get("angle_deg") is not None
    axis = spec.get("mode", spec.get("axis"))
    has_axis = axis is not None

    # Detect ambiguous over-specification (angle_deg and azimuth_deg both given,
    # or a vector alongside angles, etc.). Axis + angles is also ambiguous.
    n_families = sum([has_vector, (has_el or has_az), has_angle, has_axis])
    if n_families == 0:
        raise ValueError(
            "direction spec must provide one of: 'vector', 'elevation_deg'/"
            "'azimuth_deg', 'angle_deg', or 'mode'/'axis'. Got keys: "
            f"{sorted(spec.keys())}")
    if n_families > 1:
        raise ValueError(
            "direction spec is over-specified; provide exactly one of 'vector', "
            "'elevation_deg'/'azimuth_deg', 'angle_deg', or 'mode'/'axis'. Got keys: "
            f"{sorted(spec.keys())}")

    if has_vector:
        return normalize(spec["vector"])
    if has_el or has_az:
        return spherical_unit_vector(spec.get("elevation_deg", 0.0),
                                     spec.get("azimuth_deg", 0.0))
    if has_angle:
        return spherical_unit_vector(0.0, spec["angle_deg"])
    return axis_unit_vector(axis)


def select_dofs(unit_vec: np.ndarray, dofs: Sequence[str]) -> np.ndarray:
    """Project a full (x, y, z) unit vector onto the listed DOF axes (in order).

    ``dofs`` is e.g. ``["x", "y"]`` or ``["x", "y", "z"]``. Returns the
    components of ``unit_vec`` along those axes, preserving order. Does NOT
    renormalize — the caller decides how to treat an out-of-subspace remainder.
    """
    idx = []
    for d in dofs:
        a = str(d).strip().lower()
        if a not in _AXIS_INDEX:
            raise ValueError(f"dof must be one of 'x','y','z'; got {d!r}")
        idx.append(_AXIS_INDEX[a])
    return np.asarray(unit_vec, dtype=float)[idx]


def out_of_subspace_fraction(unit_vec: np.ndarray, dofs: Sequence[str]) -> float:
    """Return the norm of the component of ``unit_vec`` outside the DOF subspace.

    For a unit input this is ``sqrt(1 - ||projection||^2)`` — e.g. a z-pointing
    direction with ``dofs=["x","y"]`` returns ~1.0 (entirely unrealizable),
    while an in-plane direction returns ~0.0.
    """
    proj = select_dofs(unit_vec, dofs)
    rem = 1.0 - float(np.dot(proj, proj))
    return math.sqrt(max(0.0, rem))
