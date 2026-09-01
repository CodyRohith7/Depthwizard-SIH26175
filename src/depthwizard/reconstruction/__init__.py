"""reconstruction module.

Implemented (Final 3D/DSM engineering pass, on top of Ship-mode MVP):
    - elevation_grid.build_elevation_grid: the canonical structured
      elevation-grid representation (finite/nodata masks + provenance)
      that mesh reconstruction is built from.
    - mesh.build_terrain_mesh / mesh_to_json_safe / compute_elevation_colors:
      a real triangle-mesh surface reconstruction from the elevation grid
      (structured decimation, hole-aware triangulation, degenerate-triangle
      rejection, per-vertex normals, RGB + elevation vertex colors). This
      is the app's primary 3D output.
    - placeholder_preview.unproject_placeholder: the original uncalibrated
      XYZRGB point-cloud unprojection. Unchanged, still tested, still used
      by build_scene -- kept for backward compatibility, no longer the
      app's primary path (see scene_export.py's module docstring).
    - dsm.depth_to_relative_elevation / elevation_stats: reinterprets
      relative depth as a relative-elevation ("pseudo-DSM") surface. See
      dsm.py's module docstring for the exact assumption this makes and
      why it is not a real DSM.
    - scene_export.build_mesh_scene: combines an elevation grid + terrain
      mesh into one exportable dict -- the canonical scene representation.
    - scene_export.build_scene / export_scene_json /
      export_elevation_geotiff: point-cloud + file export, kept as-is.

Full scope (real camera intrinsics/pose, SRTM/GCP metric calibration,
texture-atlas UV mapping beyond per-vertex color) remains later-milestone
work -- see reconstruction.mesh's module docstring for exactly what
assumption this pass does and does not make.
"""
from depthwizard.reconstruction.placeholder_preview import (
    estimate_placeholder_intrinsics,
    unproject_placeholder,
)
from depthwizard.reconstruction.dsm import depth_to_relative_elevation, elevation_stats
from depthwizard.reconstruction.elevation_grid import build_elevation_grid
from depthwizard.reconstruction.mesh import (
    build_terrain_mesh,
    mesh_to_json_safe,
    compute_elevation_colors,
)
from depthwizard.reconstruction.scene_export import (
    build_scene,
    build_mesh_scene,
    export_scene_json,
    export_elevation_geotiff,
)

__all__ = [
    "estimate_placeholder_intrinsics",
    "unproject_placeholder",
    "depth_to_relative_elevation",
    "elevation_stats",
    "build_elevation_grid",
    "build_terrain_mesh",
    "mesh_to_json_safe",
    "compute_elevation_colors",
    "build_scene",
    "build_mesh_scene",
    "export_scene_json",
    "export_elevation_geotiff",
]
