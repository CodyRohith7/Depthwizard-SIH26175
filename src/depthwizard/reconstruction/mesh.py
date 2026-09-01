"""Structured surface-mesh reconstruction from a canonical elevation grid.

This replaces the point-cloud-only preview as the primary 3D output: it
builds a real triangle mesh -- vertices connected into faces with normals
-- not a scattered point set with a large point size.

Method (deliberately simple and inspectable, not a learned/black-box
step):
    1. Decimate the elevation grid onto a regular sub-grid (a stride, not
       a resize/blur) so the vertex count stays practical for a browser
       scene. Decimation is *structured* -- row/col spacing is preserved --
       which is what makes triangle connectivity well-defined afterward.
    2. Unproject each sub-grid sample through the SAME named, uncalibrated
       placeholder camera model already used for the point-cloud preview
       (reconstruction.placeholder_preview) -- this module does not add a
       second, different geometry assumption.
    3. Connect each 2x2 block of *finite* neighboring samples into two
       triangles. A block touching any nodata sample is skipped entirely,
       so holes in the input are never bridged by a triangle spanning
       them -- the mesh simply has a hole there too.
    4. Reject any surviving triangle whose longest edge is a large
       outlier relative to the local grid spacing (a robust multiple of
       the median edge length) -- this catches the case of a single wild
       elevation value creating a degenerate spike, without requiring a
       second, more expensive smoothing pass.
    5. Compute per-vertex normals by averaging adjacent face normals, for
       real lighting instead of a flat-shaded/point appearance.

Every value here is a deterministic, inspectable function of the input
array (same discipline as reconstruction.dsm) -- nothing here estimates a
metric height, an accuracy score, or an uncertainty value. `calibrated` is
always False for the geometry this module produces, because it is built
from the same placeholder camera model as the existing point cloud.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from depthwizard.reconstruction.placeholder_preview import estimate_placeholder_intrinsics


def _decimation_stride(rows: int, cols: int, max_vertices: int) -> int:
    if max_vertices <= 0:
        raise ValueError("max_vertices must be positive")
    total = rows * cols
    if total <= max_vertices:
        return 1
    return int(np.ceil(np.sqrt(total / max_vertices)))


def _compute_vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Average adjacent face normals into a per-vertex normal.

    Vertices touched by no surviving triangle (fully isolated by
    decimation/hole-skipping) keep a zero normal -- Three.js treats an
    unnormalized/zero normal as "no lighting contribution" for that
    vertex rather than crashing, and such vertices are, by construction,
    not referenced by any triangle anyway, so they are never actually
    rendered as part of the surface.
    """
    normals = np.zeros_like(vertices)
    if triangles.shape[0] == 0:
        return normals
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, triangles[:, i], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths == 0, 1.0, lengths)
    return normals / lengths


def compute_elevation_colors(elevation_values: np.ndarray, colormap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    """Map per-vertex elevation values to RGB via the same colormap
    convention already used for the 2D elevation preview image, so the
    3D "Elevation" visual mode and the 2D elevation panel read as the
    same encoding.

    Uses nan-aware min/max: plain np.min/np.max propagate NaN across the
    *entire* output the moment a single nodata value is present, which
    would silently corrupt every vertex's color, not just the nodata
    ones. NaN entries themselves are mapped to 0 afterward -- they are
    never referenced by a triangle (see build_terrain_mesh's hole
    handling), so their color is never actually drawn.
    """
    finite = np.isfinite(elevation_values)
    if not finite.any():
        return np.zeros((elevation_values.shape[0], 3), dtype=np.uint8)
    lo = float(np.min(elevation_values[finite]))
    hi = float(np.max(elevation_values[finite]))
    span = hi - lo if hi > lo else 1.0
    norm = np.clip((elevation_values - lo) / span, 0.0, 1.0)
    norm = np.where(finite, norm, 0.0)
    u8 = (norm * 255).astype(np.uint8).reshape(-1, 1)
    colored = cv2.applyColorMap(u8, colormap)[:, 0, ::-1]  # BGR -> RGB
    return colored


def build_terrain_mesh(
    elevation_grid: Dict[str, Any],
    rgb: np.ndarray,
    assumed_hfov_deg: float = 60.0,
    max_vertices: int = 40000,
    outlier_edge_factor: float = 8.0,
) -> Dict[str, Any]:
    """Build a triangle-mesh surface from a canonical elevation grid.

    Returns a dict of numpy arrays (vertices, colors_rgb, colors_elevation,
    normals, triangles) plus scalar metadata (vertex_count, triangle_count,
    stride, timings are added by the caller). Use `mesh_to_json_safe` to
    convert to plain Python lists before json.dumps.
    """
    elevation = elevation_grid["elevation"]
    finite_mask = elevation_grid["finite_mask"]
    rows, cols = int(elevation_grid["rows"]), int(elevation_grid["cols"])

    stride = _decimation_stride(rows, cols, max_vertices)
    elev_s = elevation[::stride, ::stride]
    mask_s = finite_mask[::stride, ::stride]
    rgb_s = rgb[::stride, ::stride]
    sr, sc = elev_s.shape

    if sr < 2 or sc < 2:
        raise ValueError(
            f"Image too small to build a surface mesh after decimation "
            f"({sr}x{sc} samples from a {rows}x{cols} source, need at least 2x2). "
            f"Try a larger input image."
        )
    if not mask_s.any():
        raise ValueError("Elevation grid has no finite values -- nothing to reconstruct.")

    fx, fy, cx, cy = estimate_placeholder_intrinsics(cols, rows, assumed_hfov_deg)
    yy, xx = np.indices((sr, sc))
    orig_yy = (yy * stride).astype(np.float64)
    orig_xx = (xx * stride).astype(np.float64)
    z = elev_s.astype(np.float64)
    x = (orig_xx - cx) * z / fx
    y = (orig_yy - cy) * z / fy

    # Nodata samples get a finite placeholder position (0,0,0) so the
    # vertex array itself is always finite -- they are excluded from every
    # triangle by construction below, so they are never actually drawn.
    x = np.where(mask_s, x, 0.0)
    y = np.where(mask_s, y, 0.0)
    z = np.where(mask_s, z, 0.0)

    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    if rgb_s.ndim == 3:
        colors_rgb = rgb_s[:, :, :3].reshape(-1, 3).astype(np.uint8)
    else:
        colors_rgb = np.tile(rgb_s.reshape(-1, 1), (1, 3)).astype(np.uint8)
    colors_elevation = compute_elevation_colors(elev_s.reshape(-1))

    def idx(r: np.ndarray, c: np.ndarray) -> np.ndarray:
        return r * sc + c

    r0, c0 = np.indices((sr - 1, sc - 1))
    m00 = mask_s[:-1, :-1]
    m01 = mask_s[:-1, 1:]
    m10 = mask_s[1:, :-1]
    m11 = mask_s[1:, 1:]

    tri_a_ok = (m00 & m01 & m10).reshape(-1)
    tri_b_ok = (m01 & m10 & m11).reshape(-1)

    i00 = idx(r0, c0).reshape(-1)
    i01 = idx(r0, c0 + 1).reshape(-1)
    i10 = idx(r0 + 1, c0).reshape(-1)
    i11 = idx(r0 + 1, c0 + 1).reshape(-1)

    tri_a = np.stack([i00[tri_a_ok], i10[tri_a_ok], i01[tri_a_ok]], axis=1)
    tri_b = np.stack([i01[tri_b_ok], i10[tri_b_ok], i11[tri_b_ok]], axis=1)
    triangles = np.concatenate([tri_a, tri_b], axis=0) if (tri_a.size or tri_b.size) else np.zeros((0, 3), dtype=np.int64)

    if triangles.shape[0] == 0:
        raise ValueError(
            "No valid triangles could be formed -- the elevation grid has no "
            "connected 2x2 block of finite values after decimation."
        )

    # Reject degenerate/outlier triangles: a single wild elevation value
    # can otherwise produce one very long, very visible spike.
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    edge_lengths = np.stack([
        np.linalg.norm(v1 - v0, axis=1),
        np.linalg.norm(v2 - v1, axis=1),
        np.linalg.norm(v0 - v2, axis=1),
    ], axis=1)
    max_edge = edge_lengths.max(axis=1)
    median_edge = float(np.median(max_edge)) if max_edge.size else 0.0
    threshold = max(median_edge * outlier_edge_factor, 1e-9)
    keep = max_edge <= threshold
    dropped = int((~keep).sum())
    triangles = triangles[keep]

    if triangles.shape[0] == 0:
        raise ValueError(
            "All candidate triangles were rejected as degenerate outliers -- "
            "the elevation surface may be corrupt or contain no coherent structure."
        )

    normals = _compute_vertex_normals(vertices, triangles)

    valid_vertex_idx = np.unique(triangles.reshape(-1))
    valid_positions = vertices[valid_vertex_idx]
    bbox_min = valid_positions.min(axis=0)
    bbox_max = valid_positions.max(axis=0)

    return {
        "vertices": vertices,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "colors_rgb": colors_rgb,
        "colors_elevation": colors_elevation,
        "normals": normals,
        "triangles": triangles.astype(np.int64),
        "grid_rows": sr,
        "grid_cols": sc,
        "stride": stride,
        "source_rows": rows,
        "source_cols": cols,
        "vertex_count": int(vertices.shape[0]),
        "triangle_count": int(triangles.shape[0]),
        "degenerate_triangles_dropped": dropped,
        "calibrated": False,
        "intrinsics_source": "placeholder_uncalibrated",
        "assumed_hfov_deg": assumed_hfov_deg,
    }


def mesh_to_json_safe(mesh: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a build_terrain_mesh() result to plain Python types so it
    can go through json.dumps for the browser and the scene-JSON export.
    """
    out = dict(mesh)
    out["vertices"] = mesh["vertices"].astype(np.float64).tolist()
    out["colors_rgb"] = mesh["colors_rgb"].astype(int).tolist()
    out["colors_elevation"] = mesh["colors_elevation"].astype(int).tolist()
    out["normals"] = mesh["normals"].astype(np.float64).tolist()
    out["triangles"] = mesh["triangles"].astype(int).tolist()
    out["bbox_min"] = mesh["bbox_min"].astype(np.float64).tolist()
    out["bbox_max"] = mesh["bbox_max"].astype(np.float64).tolist()
    return out
