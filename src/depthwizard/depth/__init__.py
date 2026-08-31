"""Model-agnostic monocular depth estimation.

Milestone M1 scope:
    - DepthResult: the standard output contract every backend returns.
    - DepthPredictor.predict(image, allow_fallback=...): the single public
      entry point. Real model failures are raised as ModelLoadError /
      ModelInferenceError and are NEVER silently swallowed into a fallback
      -- a caller must explicitly opt in with allow_fallback=True to get
      the deterministic DEMO_FALLBACK heuristic, and the result is always
      tagged status=DepthStatus.DEMO_FALLBACK so it can never be mistaken
      for a real prediction downstream.
    - DepthAnythingV2Backend: wraps the primary model
      (depth-anything/Depth-Anything-V2-Small-hf) via transformers. This
      backend's output is RELATIVE depth -- see the Phase 0 audit, Section
      3 -- and must not be treated as metric until a /scale calibration
      step (a later milestone) has run.

Deferred: additional backends (Depth Pro, etc.), fine-tuning, ensembling.
"""
from depthwizard.depth.base import DepthResult, DepthStatus
from depthwizard.depth.exceptions import (
    DepthWizardModelError,
    ModelLoadError,
    ModelInferenceError,
)
from depthwizard.depth.predictor import DepthPredictor

__all__ = [
    "DepthResult",
    "DepthStatus",
    "DepthWizardModelError",
    "ModelLoadError",
    "ModelInferenceError",
    "DepthPredictor",
]
