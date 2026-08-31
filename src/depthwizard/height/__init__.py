"""height module.

Implemented (Ship-mode MVP pass): point-to-point relative elevation
measurement only -- see measurement.relative_height_between_points and its
docstring for exactly what is (and is not) being measured, and how a real
meters value can be attached if an external calibration is supplied.

NOT implemented: automatic metric calibration (SRTM/GCP/RANSAC), area or
volume measurement, or any per-pixel confidence/accuracy score. See the
Phase 0 audit's roadmap (Section 20) -- automatic calibration is
Milestone M2+ scope.
"""
from depthwizard.height.measurement import relative_height_between_points

__all__ = ["relative_height_between_points"]
