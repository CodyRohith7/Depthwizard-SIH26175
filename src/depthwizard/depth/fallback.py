"""Deterministic, non-AI fallback depth heuristic.

This is the same core idea as the original prototype's fallback (a
Gaussian-blurred, inverted grayscale proxy) -- see the Phase 0 audit,
Section 5: it is a genuinely useful "always works, offline" development
path, and worth keeping *as long as it is never mistaken for a real
prediction*. That is enforced structurally: this function's result is only
ever wrapped as DepthStatus.DEMO_FALLBACK by DepthPredictor, never as
DepthStatus.SUCCESS.

It is intentionally deterministic (no randomness) so tests can assert
exact-ish output.
"""
from __future__ import annotations

import numpy as np
import cv2


def run_fallback_depth(
    image: np.ndarray,
    gaussian_blur_sigma: float = 5.0,
    min_value: float = 0.3,
    max_value: float = 1.0,
) -> np.ndarray:
    """Produce a deterministic depth-shaped array from an RGB/gray image.

    NOT a depth prediction. Brighter regions are heuristically treated as
    "nearer" (lower value); this has no grounding in actual scene geometry
    and exists only so the rest of the pipeline (predictor, reconstruction
    preview, viewer) has something deterministic to run against when no
    real model is available.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        gray = image.astype(np.float32) / 255.0

    blurred = cv2.GaussianBlur(gray, (0, 0), gaussian_blur_sigma)
    span = max_value - min_value
    depth = (1.0 - blurred) * span + min_value
    return depth.astype(np.float32)
