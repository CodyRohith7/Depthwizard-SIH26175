"""Exceptions for the io module.

Per the M1 scientific-safety and error-handling rules: failures here must
be visible and specific, never swallowed into a silent fallback.
"""


class DepthWizardIOError(Exception):
    """Base class for all depthwizard.io errors."""


class UnsupportedImageError(DepthWizardIOError):
    """Raised when a file cannot be read as an image at all (corrupted,
    truncated, or not an image format we support)."""


class CoordinateTransformError(DepthWizardIOError):
    """Raised when a pixel<->geo coordinate transform is requested but the
    raster has no valid affine transform (e.g. it is not georeferenced), or
    the transform is degenerate (non-invertible)."""
