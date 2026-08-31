"""Depth Anything V2 (Small) backend, via transformers.

Primary backend per the Phase 0 audit's model comparison (Section 9):
Apache-2.0 licensed, relative depth only. Requires `torch` and
`transformers` to be installed -- these are declared as required
dependencies in requirements.txt, but are optional at import time here so
that the rest of depthwizard (io, the fallback path, tests) stays usable
in an environment where they are not installed.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from depthwizard.depth.exceptions import ModelLoadError, ModelInferenceError
from depthwizard.logging_setup import get_logger

log = get_logger("depthwizard.depth.backends.depth_anything")

DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthAnythingV2Backend:
    """Lazily loads a transformers depth-estimation pipeline on first use.

    Any failure to import torch/transformers, download weights, or
    initialize the pipeline is raised as ModelLoadError with the original
    exception chained -- it is the caller's (DepthPredictor's)
    responsibility to decide whether that is fatal or whether a fallback
    is acceptable. This class never falls back on its own.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id = model_id
        self._pipe: Optional[Any] = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        log.info("Loading depth backend %r (this may download model weights)...", self.model_id)
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:
            raise ModelLoadError(
                f"Cannot load backend {self.model_id!r}: `transformers`/`torch` are not "
                "installed. Install the `depth-model` extra "
                "(pip install -e '.[depth-model]') to use real inference, or call "
                "predict(..., allow_fallback=True) for the explicit DEMO_FALLBACK path."
            ) from exc
        try:
            self._pipe = pipeline("depth-estimation", model=self.model_id)
        except Exception as exc:  # genuinely broad: any of a dozen library-specific
            # errors can come from here (network, disk, corrupt cache, OOM, ...).
            # Every one of them must surface, not vanish -- so we catch broadly
            # but ALWAYS re-raise as a clearly-typed, chained ModelLoadError
            # rather than swallowing it.
            raise ModelLoadError(
                f"Failed to load backend {self.model_id!r}: {exc}"
            ) from exc
        log.info("Depth backend %r loaded.", self.model_id)

    def predict(self, image) -> np.ndarray:
        """`image` is a PIL.Image. Returns relative depth as float32 (H, W),
        min-max normalized to [0, 1] (matches the prototype's normalization,
        which is a reasonable display/consistency convention -- it does NOT
        make the output metric)."""
        self.load()
        assert self._pipe is not None
        try:
            output = self._pipe(image)
            depth = np.asarray(output["depth"], dtype=np.float32)
        except Exception as exc:
            raise ModelInferenceError(
                f"Backend {self.model_id!r} failed during inference: {exc}"
            ) from exc

        d_min, d_max = float(depth.min()), float(depth.max())
        denom = d_max - d_min if (d_max - d_min) > 1e-8 else 1e-8
        return ((depth - d_min) / denom).astype(np.float32)
