"""Uncalibrated preview unprojection -- for the M1 viewer scaffold ONLY.

THIS IS NOT A CALIBRATED CAMERA MODEL AND MUST NEVER BE USED TO PRODUCE A
HEIGHT, DSM, OR ANY OTHER MEASUREMENT SHOWN TO A USER.

Scientific-safety rule E.3 explicitly forbids silently assuming
fx = fy = 0.9 * image_width as if it were a calibrated focal length. This
module does not do that: it takes an explicit, named,
configuration-driven field-of-view assumption
(configs/reconstruction.yaml: placeholder_preview.assumed_horizontal_fov_deg)
and every point cloud it produces is tagged
`calibrated: False` / `intrinsics_source: "placeholder_uncalibrated"` in its
scene metadata, so nothing downstream can mistake it for real geometry.
Real intrinsics (EXIF, GeoTIFF geotransform, or an estimator) are future
work for /geometry -- Phase 0 audit, Section 8.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def estimate_placeholder_intrinsics(
    width: int, height: int, assumed_hfov_deg: float = 60.0
) -> Tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) for an assumed horizontal field of view.

    This is a documented, configurable, EXPLICIT placeholder -- not a
    silent magic constant. It is only accurate if the assumption about the
    camera's field of view happens to be correct, which is not verified
    anywhere in this function.
    """
    hfov_rad = np.deg2rad(assumed_hfov_deg)
    fx = (width / 2.0) / np.tan(hfov_rad / 2.0)
    fy = fx  # square-pixel assumption; also unverified
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return float(fx), float(fy), float(cx), float(cy)


def unproject_placeholder(
    depth: np.ndarray,
    rgb: np.ndarray,
    assumed_hfov_deg: float = 60.0,
    max_points: int = 20000,
) -> dict:
    """Unproject a relative-depth array to an uncalibrated XYZRGB point set.

    Returns a plain dict (JSON-serializable) with:
        points: list of [x, y, z, r, g, b]
        calibrated: False (always)
        intrinsics_source: "placeholder_uncalibrated"
        assumed_hfov_deg: the FOV assumption actually used
    """
    h, w = depth.shape
    fx, fy, cx, cy = estimate_placeholder_intrinsics(w, h, assumed_hfov_deg)

    yy, xx = np.indices((h, w))
    z = depth
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy

    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, rgb.shape[-1])[:, :3] if rgb.ndim == 3 else np.tile(
        rgb.reshape(-1, 1), (1, 3)
    )

    n = pts.shape[0]
    stride = max(1, int(np.sqrt(n / max_points)))
    pts = pts[::stride]
    colors = colors[::stride]

    points = [
        [float(p[0]), float(p[1]), float(p[2]), int(c[0]), int(c[1]), int(c[2])]
        for p, c in zip(pts, colors)
    ]

    return {
        "points": points,
        "calibrated": False,
        "intrinsics_source": "placeholder_uncalibrated",
        "assumed_hfov_deg": assumed_hfov_deg,
        "point_count": len(points),
    }
