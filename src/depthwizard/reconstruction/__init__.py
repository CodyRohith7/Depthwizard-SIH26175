"""reconstruction module.

Implemented (Ship-mode MVP pass, on top of M1):
    - placeholder_preview.unproject_placeholder: uncalibrated XYZRGB
      unprojection, explicitly tagged calibrated=False. Unchanged from M1.
    - dsm.depth_to_relative_elevation / elevation_stats: reinterprets
      relative depth as a relative-elevation ("pseudo-DSM") surface. See
      dsm.py's module docstring for the exact assumption this makes and
      why it is not a real DSM.
    - scene_export.build_scene / export_scene_json /
      export_elevation_geotiff: end-to-end point-cloud + file export for
      the demo app, reusing unproject_placeholder rather than replacing it.

Full scope (colored/filtered point clouds beyond a point cap, meshing,
observed/inferred/unknown tagging -- Phase 0 audit Section 12) is still NOT
implemented; that remains later-milestone work.
"""
from depthwizard.reconstruction.placeholder_preview import (
    estimate_placeholder_intrinsics,
    unproject_placeholder,
)
from depthwizard.reconstruction.dsm import depth_to_relative_elevation, elevation_stats
from depthwizard.reconstruction.scene_export import (
    build_scene,
    export_scene_json,
    export_elevation_geotiff,
)

__all__ = [
    "estimate_placeholder_intrinsics",
    "unproject_placeholder",
    "depth_to_relative_elevation",
    "elevation_stats",
    "build_scene",
    "export_scene_json",
    "export_elevation_geotiff",
]
