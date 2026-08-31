"""Relative elevation ("pseudo-DSM") derived from monocular relative depth.

SCOPE AND HONESTY CONTRACT (read before using this module elsewhere):

    depthwizard.depth.DepthPredictor gives you D_rel(u,v) in [0, 1] -- a
    per-pixel *relative* depth-from-camera value with an unknown affine
    relationship to any real-world distance (Phase 0 audit, Section 3).

    This module reinterprets that same array as a per-pixel *relative
    elevation* surface, i.e. it treats "closer to the camera" as a proxy
    for "higher in the scene" (or the reverse, if `invert=False`). That
    reinterpretation is a genuine, named ASSUMPTION -- not a geometric
    fact -- because depth-from-camera and terrain elevation are only the
    same thing for an ideal nadir (straight-down) camera looking at a
    world where nothing occludes anything else. For oblique aerial shots,
    tall structures whose tops are closer to the camera than the ground
    around their base, this assumption can be directly wrong at object
    boundaries.

    The output of this module is therefore called a "relative elevation
    surface" or "pseudo-DSM" everywhere in this codebase and the UI --
    never "DSM" or "elevation" unqualified, and never in real-world units
    (meters) unless a caller supplies an explicit calibration (see
    depthwizard.height.measurement for how that calibration, when
    available, is applied and labeled).

No fabricated numbers: every function here is a deterministic, inspectable
transform of the input array. Nothing here invents a height, an accuracy
percentage, or a confidence score.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def depth_to_relative_elevation(depth: np.ndarray, invert: bool = True) -> np.ndarray:
    """Reinterpret a [0, 1] relative-depth array as a relative-elevation surface.

    Args:
        depth: (H, W) float array, values expected in [0, 1] (the contract
            produced by depthwizard.depth.DepthPredictor).
        invert: if True (default), elevation = 1 - depth, i.e. pixels the
            model judged "closer to the camera" become "higher". This
            matches the common case of a roughly-overhead camera looking
            down at terrain/rooftops. If the scene geometry or camera
            angle makes that assumption wrong, pass invert=False to use
            depth directly as elevation instead.

    Returns:
        (H, W) float32 array, still unitless and relative -- NOT meters.
    """
    if depth.ndim != 2:
        raise ValueError(f"depth_to_relative_elevation expects a 2D (H, W) array, got shape {depth.shape}")
    elevation = (1.0 - depth) if invert else depth.copy()
    return elevation.astype(np.float32)


def elevation_stats(elevation: np.ndarray) -> Dict[str, Any]:
    """Real, computed descriptive statistics of a relative-elevation surface.

    Every value here is directly computed from `elevation` -- nothing is
    estimated, guessed, or expressed in real-world units. `unit` is always
    "relative" unless a caller has separately performed and labeled a
    calibration step (see height.measurement.CalibratedHeight).
    """
    return {
        "min": float(np.min(elevation)),
        "max": float(np.max(elevation)),
        "mean": float(np.mean(elevation)),
        "std": float(np.std(elevation)),
        "unit": "relative",
        "calibrated": False,
    }
