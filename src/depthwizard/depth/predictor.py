"""DepthPredictor: the single public entry point for depth inference.

    from depthwizard.depth import DepthPredictor
    predictor = DepthPredictor()
    result = predictor.predict(image_array)                      # raises on real failure
    result = predictor.predict(image_array, allow_fallback=True) # falls back, tagged

This is deliberately the *only* place fallback logic is allowed to trigger,
and it never triggers unless the caller explicitly passes
allow_fallback=True. See depthwizard.depth.exceptions for what gets raised
otherwise.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from depthwizard.depth.base import DepthResult, DepthStatus
from depthwizard.depth.exceptions import DepthWizardModelError
from depthwizard.depth.fallback import run_fallback_depth
from depthwizard.depth.backends.depth_anything import DepthAnythingV2Backend
from depthwizard.logging_setup import get_logger

log = get_logger("depthwizard.depth.predictor")

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "depth.yaml"


def _load_config(config_path: Optional[Path] = None) -> dict:
    path = config_path or _DEFAULT_CONFIG_PATH
    if not path.exists():
        log.warning("Depth config %s not found; using built-in defaults.", path)
        return {
            "primary_backend": "depth-anything-v2-small",
            "huggingface_model_id": "depth-anything/Depth-Anything-V2-Small-hf",
            "fallback": {"gaussian_blur_sigma": 5.0, "min_value": 0.3, "max_value": 1.0},
        }
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class DepthPredictor:
    def __init__(self, config_path: Optional[Path] = None):
        self._config = _load_config(config_path)
        self._backend = DepthAnythingV2Backend(self._config["huggingface_model_id"])

    def predict(self, image: np.ndarray, allow_fallback: bool = False) -> DepthResult:
        """Run depth inference on an RGB (H, W, 3) or grayscale (H, W) array.

        Args:
            image: input image array.
            allow_fallback: if True and the real backend cannot be loaded
                or fails during inference, return a DepthStatus.DEMO_FALLBACK
                result instead of raising. If False (the default), any
                backend failure is raised as ModelLoadError /
                ModelInferenceError -- visible, not hidden.

        Raises:
            depthwizard.depth.exceptions.DepthWizardModelError: subclass
                raised when the real backend fails and allow_fallback=False.
        """
        t0 = time.perf_counter()
        try:
            from PIL import Image

            pil_image = Image.fromarray(image)
            depth = self._backend.predict(pil_image)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return DepthResult(
                depth=depth,
                source=self._config["huggingface_model_id"],
                status=DepthStatus.SUCCESS,
                inference_time_ms=elapsed_ms,
                shape=depth.shape,
                dtype=str(depth.dtype),
                metadata={"model_id": self._config["huggingface_model_id"]},
            )
        except DepthWizardModelError as exc:
            log.error("Real depth backend failed: %s", exc)
            if not allow_fallback:
                raise
            log.warning(
                "allow_fallback=True: returning DEMO_FALLBACK depth (NOT a real "
                "prediction) because the real backend failed: %s",
                exc,
            )
            fb_cfg = self._config.get("fallback", {})
            depth = run_fallback_depth(
                image,
                gaussian_blur_sigma=fb_cfg.get("gaussian_blur_sigma", 5.0),
                min_value=fb_cfg.get("min_value", 0.3),
                max_value=fb_cfg.get("max_value", 1.0),
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return DepthResult(
                depth=depth,
                source="DEMO_FALLBACK",
                status=DepthStatus.DEMO_FALLBACK,
                inference_time_ms=elapsed_ms,
                shape=depth.shape,
                dtype=str(depth.dtype),
                metadata={"reason_for_fallback": str(exc)},
            )

    def predict_fallback_only(self, image: np.ndarray) -> DepthResult:
        """Explicitly request the DEMO_FALLBACK path without attempting to
        load the real backend at all -- useful for fast dev-loop iteration
        and for tests that must not depend on torch/transformers being
        installed."""
        t0 = time.perf_counter()
        fb_cfg = self._config.get("fallback", {})
        depth = run_fallback_depth(
            image,
            gaussian_blur_sigma=fb_cfg.get("gaussian_blur_sigma", 5.0),
            min_value=fb_cfg.get("min_value", 0.3),
            max_value=fb_cfg.get("max_value", 1.0),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return DepthResult(
            depth=depth,
            source="DEMO_FALLBACK",
            status=DepthStatus.DEMO_FALLBACK,
            inference_time_ms=elapsed_ms,
            shape=depth.shape,
            dtype=str(depth.dtype),
            metadata={"reason_for_fallback": "explicitly requested via predict_fallback_only()"},
        )
