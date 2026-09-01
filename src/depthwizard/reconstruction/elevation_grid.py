"""Canonical structured elevation-grid representation.

Wraps a relative-elevation array (from reconstruction.dsm) together with
its validity mask and full provenance into one structure. This is the
canonical input to surface reconstruction (reconstruction.mesh) --
downstream code should build a grid via `build_elevation_grid` rather than
passing a bare numpy array around, so provenance and the finite/nodata
masks travel with the data instead of being silently dropped.

Honesty contract (unchanged from reconstruction.dsm, restated here because
this is now the canonical handoff point to mesh reconstruction):
    - `unit` is "relative" and `calibrated` is False unless a caller
      supplies an explicit `calibration_source` -- there is no code path
      in this module that invents one.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def build_elevation_grid(
    elevation: np.ndarray,
    *,
    depth_source: str,
    depth_status: str,
    calibrated: bool = False,
    calibration_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the canonical elevation-grid structure.

    Args:
        elevation: (rows, cols) float array -- typically the output of
            depthwizard.reconstruction.dsm.depth_to_relative_elevation.
        depth_source: provenance string from the DepthResult that produced
            this elevation (e.g. "depth-anything/Depth-Anything-V2-Small-hf").
        depth_status: "SUCCESS" or "DEMO_FALLBACK" (DepthStatus.value).
        calibrated: True only if `elevation` has already been converted to
            real-world units by a caller. False (the default) is correct
            for every current code path in this project.
        calibration_source: required, human-readable provenance if
            calibrated=True (mirrors depthwizard.height.measurement's
            same requirement) -- never a silent/implicit calibration.

    Returns:
        A dict with the elevation array, a boolean `finite_mask` (True
        where the value is usable), the complementary `nodata_mask`,
        `valid_fraction`, shape, and full provenance. The elevation array
        itself is NOT copied into a JSON-safe form here -- this structure
        is for internal pipeline use (see reconstruction.mesh); JSON
        export happens at the scene_export boundary.
    """
    if elevation.ndim != 2:
        raise ValueError(
            f"build_elevation_grid expects a 2D (rows, cols) array, got shape {elevation.shape}"
        )
    if calibrated and not calibration_source:
        raise ValueError("calibrated=True requires a non-empty calibration_source")

    finite_mask = np.isfinite(elevation)
    nodata_mask = ~finite_mask

    return {
        "rows": int(elevation.shape[0]),
        "cols": int(elevation.shape[1]),
        "elevation": elevation,
        "finite_mask": finite_mask,
        "nodata_mask": nodata_mask,
        "valid_fraction": float(finite_mask.mean()) if finite_mask.size else 0.0,
        "depth_source": depth_source,
        "depth_status": depth_status,
        "unit": "meters" if calibrated else "relative",
        "calibrated": bool(calibrated),
        "calibration_source": calibration_source,
    }
