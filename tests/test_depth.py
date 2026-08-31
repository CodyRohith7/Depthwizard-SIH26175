"""Tests for DepthResult and DepthPredictor.

IMPORTANT ENVIRONMENT NOTE: `torch` and `transformers` are NOT installed in
the environment these tests were authored and run in (no network access to
install them there -- confirmed via a direct `pip install` attempt that
failed with a 403/host-not-allowed error, not a code bug). This means:

    - Tests of DepthResult's own contract (shape/dtype/status validation)
      run for real, independent of any model.
    - Tests of predict_fallback_only() run for real (no model needed).
    - The test of predict() WITHOUT allow_fallback is a genuine, live test
      of the "real model failures are visible, not hidden" requirement --
      it actually exercises DepthAnythingV2Backend.load() hitting a real
      ImportError in this environment and asserts it surfaces as
      ModelLoadError rather than vanishing.
    - There is currently no environment available to this session in which
      a REAL Depth Anything V2 prediction (status=SUCCESS) can be executed
      and tested. That path is implemented and reviewed, but unverified by
      execution -- flagged explicitly rather than claimed as tested.
"""
import unittest

import numpy as np

from depthwizard.depth.base import DepthResult, DepthStatus
from depthwizard.depth.predictor import DepthPredictor
from depthwizard.depth.exceptions import ModelLoadError


class TestDepthResultContract(unittest.TestCase):
    def test_shape_mismatch_rejected(self):
        depth = np.zeros((4, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            DepthResult(
                depth=depth,
                source="DEMO_FALLBACK",
                status=DepthStatus.DEMO_FALLBACK,
                inference_time_ms=1.0,
                shape=(5, 5),  # wrong on purpose
                dtype="float32",
            )

    def test_dtype_mismatch_rejected(self):
        depth = np.zeros((4, 4), dtype=np.float64)
        with self.assertRaises(ValueError):
            DepthResult(
                depth=depth,
                source="DEMO_FALLBACK",
                status=DepthStatus.DEMO_FALLBACK,
                inference_time_ms=1.0,
                shape=(4, 4),
                dtype="float32",  # wrong on purpose
            )

    def test_fallback_must_be_labelled_in_source(self):
        """Scientific-safety rule: a DEMO_FALLBACK result's `source` must
        say so -- this is what stops the fallback from ever looking like
        real AI inference to a downstream consumer that only checks
        `source` as a display string."""
        depth = np.zeros((4, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            DepthResult(
                depth=depth,
                source="depth-anything-v2-small",  # NOT labelled as fallback
                status=DepthStatus.DEMO_FALLBACK,
                inference_time_ms=1.0,
                shape=(4, 4),
                dtype="float32",
            )

    def test_valid_fallback_result_constructs(self):
        depth = np.zeros((4, 4), dtype=np.float32)
        result = DepthResult(
            depth=depth,
            source="DEMO_FALLBACK",
            status=DepthStatus.DEMO_FALLBACK,
            inference_time_ms=1.0,
            shape=(4, 4),
            dtype="float32",
        )
        self.assertEqual(result.status, DepthStatus.DEMO_FALLBACK)


class TestDepthPredictor(unittest.TestCase):
    def setUp(self):
        self.image = (np.random.default_rng(7).random((20, 24, 3)) * 255).astype(np.uint8)
        self.predictor = DepthPredictor()

    def test_predict_fallback_only(self):
        result = self.predictor.predict_fallback_only(self.image)
        self.assertEqual(result.status, DepthStatus.DEMO_FALLBACK)
        self.assertEqual(result.source, "DEMO_FALLBACK")
        self.assertEqual(result.shape, (20, 24))
        self.assertIn("reason_for_fallback", result.metadata)

    def test_predict_without_fallback_raises_visibly_when_backend_unavailable(self):
        try:
            import transformers  # noqa: F401

            self.skipTest("transformers IS installed in this environment; this test "
                           "targets the transformers-unavailable failure path specifically.")
        except ImportError:
            pass

        with self.assertRaises(ModelLoadError) as ctx:
            self.predictor.predict(self.image, allow_fallback=False)
        self.assertIn("transformers", str(ctx.exception).lower())

    def test_predict_with_allow_fallback_returns_labelled_fallback_when_backend_unavailable(self):
        try:
            import transformers  # noqa: F401

            self.skipTest("transformers IS installed in this environment; this test "
                           "targets the transformers-unavailable fallback path specifically.")
        except ImportError:
            pass

        result = self.predictor.predict(self.image, allow_fallback=True)
        self.assertEqual(result.status, DepthStatus.DEMO_FALLBACK)
        self.assertEqual(result.source, "DEMO_FALLBACK")
        self.assertIn("transformers", result.metadata["reason_for_fallback"].lower())


if __name__ == "__main__":
    unittest.main()
