"""Build and export an exportable 3D scene from depth + relative elevation.

Reuses the existing, honestly-labeled uncalibrated unprojection
(depthwizard.reconstruction.placeholder_preview) as the geometry source --
it is not replaced here, only wired into an end-to-end
depth -> elevation -> point cloud -> file pipeline for the demo app, per
"understand before replacing" and "do not replace working components
without reason".

Every exported artifact carries the same provenance fields the rest of
this codebase already uses (calibrated, intrinsics_source, source /
status of the depth result) so a file on disk is exactly as trustworthy as
the number shown for it in the UI -- no information is lost on export.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from depthwizard.reconstruction.placeholder_preview import unproject_placeholder
from depthwizard.logging_setup import get_logger

log = get_logger("depthwizard.reconstruction.scene_export")


def build_scene(
    elevation: np.ndarray,
    rgb: np.ndarray,
    depth_source: str,
    depth_status: str,
    assumed_hfov_deg: float = 60.0,
    max_points: int = 40000,
) -> Dict[str, Any]:
    """Build a JSON-serializable scene dict (point cloud + provenance).

    `elevation` is used as the Z axis (via the existing uncalibrated
    unprojection), so the resulting point cloud is explicitly tagged
    calibrated=False regardless of how `elevation` was produced -- this
    function does not know or assume a real camera model.
    """
    scene = unproject_placeholder(
        elevation, rgb, assumed_hfov_deg=assumed_hfov_deg, max_points=max_points
    )
    scene["depth_source"] = depth_source
    scene["depth_status"] = depth_status
    scene["elevation_convention"] = (
        "Z is a relative-elevation proxy derived from monocular relative "
        "depth (see depthwizard.reconstruction.dsm) -- NOT metric height."
    )
    return scene


def export_scene_json(scene: Dict[str, Any], path: Path) -> Path:
    """Write a scene dict to disk as JSON. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(scene, fh)
    log.info("Exported scene JSON to %s (%d points)", path, scene.get("point_count", 0))
    return path


def export_elevation_geotiff(
    elevation: np.ndarray,
    path: Path,
    transform=None,
    crs_wkt: Optional[str] = None,
) -> Path:
    """Write `elevation` as a single-band float32 GeoTIFF, if rasterio is
    available and the source image was georeferenced.

    Raises RuntimeError (not a silent no-op) if rasterio is not installed
    or no georeferencing was supplied -- callers must check
    `rasterio_available()` / `meta.is_georeferenced` themselves and offer
    a non-georeferenced export path (e.g. PNG/JSON) instead, exactly like
    depthwizard.io.loader already does for input.
    """
    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError as exc:
        raise RuntimeError(
            "rasterio is not installed; cannot export a georeferenced elevation "
            "GeoTIFF. Install the `geo` extra (pip install -e '.[geo]')."
        ) from exc
    if transform is None or crs_wkt is None:
        raise RuntimeError(
            "export_elevation_geotiff requires a real transform and crs_wkt from "
            "a georeferenced input image -- refusing to write a GeoTIFF with "
            "fabricated or missing georeferencing."
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = elevation.shape
    aff = Affine(*transform) if not isinstance(transform, Affine) else transform
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype="float32", crs=crs_wkt, transform=aff,
    ) as dst:
        dst.write(elevation.astype(np.float32), 1)
    log.info("Exported relative-elevation GeoTIFF to %s", path)
    return path
