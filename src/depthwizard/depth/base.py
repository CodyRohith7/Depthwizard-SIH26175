"""DepthResult: the standard output contract for every depth backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Tuple

import numpy as np


class DepthStatus(str, Enum):
    """Never let a caller confuse a real prediction with the fallback."""

    SUCCESS = "SUCCESS"
    DEMO_FALLBACK = "DEMO_FALLBACK"


@dataclass(frozen=True)
class DepthResult:
    """Output of DepthPredictor.predict().

    Attributes:
        depth: relative depth array, shape (H, W), float32. This is
            D_rel(u,v) per the Phase 0 audit's Section 3 notation -- NOT
            metric depth. Never present it as a measurement.
        source: identifies what produced `depth`, e.g.
            "depth-anything-v2-small" for a real prediction, or the literal
            string "DEMO_FALLBACK" for the deterministic non-AI heuristic.
        status: DepthStatus.SUCCESS or DepthStatus.DEMO_FALLBACK. Check this
            (not just `source`) before trusting the result as a real
            prediction.
        inference_time_ms: wall-clock time for the predict call, in
            milliseconds.
        shape: (height, width) of `depth`.
        dtype: numpy dtype name of `depth` (e.g. "float32").
        metadata: backend-specific extra info (model id, device, and for a
            fallback result, `reason_for_fallback` explaining exactly why
            the real backend was not used).
    """

    depth: np.ndarray
    source: str
    status: DepthStatus
    inference_time_ms: float
    shape: Tuple[int, int]
    dtype: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.depth.shape != self.shape:
            raise ValueError(
                f"DepthResult.shape {self.shape} does not match depth.shape {self.depth.shape}"
            )
        if str(self.depth.dtype) != self.dtype:
            raise ValueError(
                f"DepthResult.dtype {self.dtype!r} does not match depth.dtype {self.depth.dtype!r}"
            )
        if self.status == DepthStatus.DEMO_FALLBACK and "DEMO_FALLBACK" not in self.source.upper():
            raise ValueError(
                "A DEMO_FALLBACK result's `source` must say so explicitly "
                "(scientific-safety rule: never make the fallback look like AI inference)."
            )
