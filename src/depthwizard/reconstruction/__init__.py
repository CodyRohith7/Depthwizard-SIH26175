"""reconstruction module.

Full scope (colored/filtered point clouds, meshing, observed/inferred/
unknown tagging -- Phase 0 audit Section 12) is NOT implemented; that is
later-milestone work.

The one thing implemented now, `placeholder_preview.unproject_placeholder`,
exists ONLY to let the M1 frontend viewer scaffold have something real to
render (scripts/export_sample_scene.py). It is explicitly named,
documented, and configured (configs/reconstruction.yaml) as an uncalibrated
preview -- never a measurement. See its docstring before using it for
anything else.
"""
