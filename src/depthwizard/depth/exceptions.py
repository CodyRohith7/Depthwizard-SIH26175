"""Exceptions for the depth module.

M1 rule (from the approved architecture): the prototype's
    try:
        ...
    except Exception:
        return <silent fallback>
is unacceptable. A real backend failure must surface as one of these,
with the original exception chained via `raise ... from exc`, and it is
DepthPredictor's caller (not the backend) who decides whether a fallback
is acceptable.
"""


class DepthWizardModelError(Exception):
    """Base class for all depthwizard.depth errors."""


class ModelLoadError(DepthWizardModelError):
    """Raised when a real depth model backend could not be loaded
    (missing dependency, no network to fetch weights, incompatible
    hardware, corrupted checkpoint, etc.)."""


class ModelInferenceError(DepthWizardModelError):
    """Raised when a real depth model backend loaded successfully but
    failed during inference on a specific image (OOM, malformed input,
    unexpected output shape, etc.)."""
