"""Point-to-point relative height/elevation measurement.

Implements the one form of "height estimation" that is mathematically
supported by what M1 + the pseudo-DSM (depthwizard.reconstruction.dsm)
actually provide: the *difference* in relative-elevation units between two
pixels the user selects. This is deliberately NOT the old prototype's
"depth-span" feature (Phase 0 audit Section 4 -- that feature read a
single scalar out of two badly-chosen image regions and called it a
height in meters; it is not reachable from this module and never will be
via this code path).

Two outputs are possible, and this module is explicit about which one it
is giving you:

  1. UNCALIBRATED (the default, always available): the raw difference in
     relative-elevation units between the two points. No unit conversion
     is performed. `calibrated` is False and `unit` is "relative".

  2. CALIBRATED (only if the caller supplies `meters_per_unit`, i.e. an
     external elevation reference has already been used -- e.g. Milestone
     P1's "external elevation anchoring" -- to derive a scale factor):
     the relative difference multiplied by that factor, in meters.
     `calibrated` is True, `unit` is "meters", and `calibration_source`
     records where the scale factor came from because that is supplied by
     the caller, not invented here.

This module never performs SRTM lookup, RANSAC, or any other automatic
calibration itself -- supplying `meters_per_unit` is entirely the caller's
responsibility, and this stays true even after that capability exists
elsewhere in the codebase (Milestone M2+).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def relative_height_between_points(
    elevation: np.ndarray,
    point_a: Tuple[int, int],
    point_b: Tuple[int, int],
    meters_per_unit: Optional[float] = None,
    calibration_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute the elevation difference between two pixel points.

    Args:
        elevation: (H, W) relative-elevation array, e.g. from
            depthwizard.reconstruction.dsm.depth_to_relative_elevation.
        point_a, point_b: (row, col) pixel coordinates. Must be inside
            `elevation`'s bounds.
        meters_per_unit: optional external scale factor (relative units ->
            meters). If omitted, the result stays uncalibrated.
        calibration_source: required alongside meters_per_unit -- a short
            human-readable string naming where the scale factor came from
            (e.g. "SRTM anchor at (12.34, 56.78), M2"), so a calibrated
            number in the UI is always traceable to its assumption rather
            than looking like ground truth.

    Returns:
        dict with point_a, point_b, elevation_a, elevation_b,
        relative_difference, calibrated, unit, and (if calibrated)
        meters_difference and calibration_source.

    Raises:
        ValueError: if a point is out of bounds, or meters_per_unit is
            given without calibration_source (a calibrated number must
            always carry its provenance).
    """
    h, w = elevation.shape
    for name, (r, c) in (("point_a", point_a), ("point_b", point_b)):
        if not (0 <= r < h and 0 <= c < w):
            raise ValueError(f"{name}={(r, c)} is out of bounds for elevation shape {(h, w)}")

    if meters_per_unit is not None and not calibration_source:
        raise ValueError(
            "meters_per_unit was supplied without calibration_source -- a calibrated "
            "measurement must always record where its scale factor came from."
        )

    elev_a = float(elevation[point_a[0], point_a[1]])
    elev_b = float(elevation[point_b[0], point_b[1]])
    rel_diff = elev_a - elev_b

    result: Dict[str, Any] = {
        "point_a": list(point_a),
        "point_b": list(point_b),
        "elevation_a": elev_a,
        "elevation_b": elev_b,
        "relative_difference": rel_diff,
        "calibrated": False,
        "unit": "relative",
    }

    if meters_per_unit is not None:
        result["calibrated"] = True
        result["unit"] = "meters"
        result["meters_difference"] = rel_diff * meters_per_unit
        result["calibration_source"] = calibration_source

    return result
