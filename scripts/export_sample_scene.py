#!/usr/bin/env python3
"""Generate frontend/public/sample_scene.json for the M1 viewer scaffold.

Runs the REAL io -> depth -> reconstruction chain (not a hand-written
fixture): loads a small synthetic test image, runs DepthPredictor in
explicit DEMO_FALLBACK mode (no real model backend is available in this
environment -- see README.md's Implemented/Placeholder/Not-started table),
and unprojects the result with the explicitly-uncalibrated preview
unprojector. The output JSON is tagged with full provenance so the
frontend (and anyone reading the file) can see exactly how it was made.

Usage:
    PYTHONPATH=src python3 scripts/export_sample_scene.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from depthwizard.io.loader import load_image  # noqa: E402
from depthwizard.depth.predictor import DepthPredictor  # noqa: E402
from depthwizard.reconstruction.placeholder_preview import unproject_placeholder  # noqa: E402
from depthwizard.logging_setup import configure_logging, get_logger  # noqa: E402

configure_logging()
log = get_logger("depthwizard.scripts.export_sample_scene")


def _make_synthetic_test_image(path: Path) -> None:
    """A small procedurally generated RGB test image (not a real photo) --
    exists only so this script has *something* real to run io/depth on
    without depending on a fixture file being present."""
    h, w = 96, 128
    yy, xx = np.indices((h, w))
    r = (128 + 100 * np.sin(xx / 8.0)).clip(0, 255)
    g = (128 + 100 * np.cos(yy / 8.0)).clip(0, 255)
    b = np.full((h, w), 180.0)
    # A brighter "structure" block so the fallback heuristic has some
    # local contrast to work with.
    r[20:60, 40:90] = 230
    g[20:60, 40:90] = 220
    b[20:60, 40:90] = 200
    arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def main() -> None:
    fixtures_dir = REPO_ROOT / "data" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    test_image_path = fixtures_dir / "synthetic_sample.png"
    if not test_image_path.exists():
        log.info("Generating synthetic test image at %s", test_image_path)
        _make_synthetic_test_image(test_image_path)

    image, meta = load_image(test_image_path)
    log.info("Loaded %s: %dx%d, georeferenced=%s", test_image_path, meta.width, meta.height, meta.is_georeferenced)

    predictor = DepthPredictor()
    depth_result = predictor.predict(image, allow_fallback=True)
    log.info(
        "Depth source=%s status=%s (%.2f ms)",
        depth_result.source, depth_result.status.value, depth_result.inference_time_ms,
    )

    scene = unproject_placeholder(depth_result.depth, image)
    scene["provenance"] = {
        "source_image": str(test_image_path.relative_to(REPO_ROOT)),
        "depth_source": depth_result.source,
        "depth_status": depth_result.status.value,
        "reason_for_fallback": depth_result.metadata.get("reason_for_fallback"),
        "warning": (
            "This scene is a Milestone M1 viewer scaffold fixture. Depth is "
            "DEMO_FALLBACK (not a real model prediction) and intrinsics are "
            "an explicit, uncalibrated placeholder. Do not present this as a "
            "measurement or as real AI inference."
        ),
    }

    out_path = REPO_ROOT / "frontend" / "public" / "sample_scene.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(scene, fh)
    log.info("Wrote %s (%d points)", out_path, scene["point_count"])


if __name__ == "__main__":
    main()
