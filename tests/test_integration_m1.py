"""End-to-end M1 integration test: io -> depth -> preview reconstruction.

Uses a plain synthetic PNG (not a real GeoTIFF -- see tests/test_io.py's
module docstring for why the georeferenced branch is tested via mocking
instead) and the explicit DEMO_FALLBACK depth path, since no real depth
model backend is available in this environment (no torch/transformers, no
network access to install them). This test says so explicitly rather than
implying a real model ran.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from depthwizard.io.loader import load_image
from depthwizard.depth.predictor import DepthPredictor
from depthwizard.depth.base import DepthStatus
from depthwizard.reconstruction.placeholder_preview import unproject_placeholder


class TestM1Integration(unittest.TestCase):
    def test_geotiff_shaped_pipeline_with_explicit_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 1. /io: load a plain (non-georeferenced) test image. A real
            #    GeoTIFF fixture would additionally exercise is_georeferenced
            #    -> True; see tests/test_io.py's TestRasterioBranchLogic for
            #    that logic tested via a mocked rasterio, since the real
            #    dependency is not installed in this environment.
            arr = (np.random.default_rng(99).random((40, 60, 3)) * 255).astype(np.uint8)
            path = Path(tmp) / "scene.png"
            Image.fromarray(arr, mode="RGB").save(path)

            image, meta = load_image(path)
            self.assertFalse(meta.is_georeferenced)
            self.assertEqual(image.shape, (40, 60, 3))

            # 2. /depth: explicit fallback path -- asserted below, not just claimed.
            predictor = DepthPredictor()
            depth_result = predictor.predict_fallback_only(image)
            self.assertEqual(depth_result.status, DepthStatus.DEMO_FALLBACK)
            self.assertEqual(depth_result.source, "DEMO_FALLBACK")
            print(
                "[test_integration_m1] depth source used: "
                f"{depth_result.source} (status={depth_result.status.value}); "
                f"reason: {depth_result.metadata.get('reason_for_fallback')}"
            )

            # 3. /reconstruction (M1 preview only): unproject to an
            #    explicitly-uncalibrated point set.
            scene = unproject_placeholder(depth_result.depth, image)
            self.assertFalse(scene["calibrated"])
            self.assertGreater(scene["point_count"], 0)
            self.assertEqual(scene["intrinsics_source"], "placeholder_uncalibrated")


if __name__ == "__main__":
    unittest.main()
