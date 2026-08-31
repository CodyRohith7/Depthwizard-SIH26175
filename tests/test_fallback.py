"""Tests for the deterministic DEMO_FALLBACK depth heuristic.

Runs for real in any environment with numpy/opencv installed -- no torch,
transformers, or network access required.
"""
import unittest

import numpy as np

from depthwizard.depth.fallback import run_fallback_depth


class TestFallbackDepth(unittest.TestCase):
    def setUp(self):
        self.image = (np.random.default_rng(42).random((32, 48, 3)) * 255).astype(np.uint8)

    def test_output_shape_and_dtype(self):
        depth = run_fallback_depth(self.image)
        self.assertEqual(depth.shape, (32, 48))
        self.assertEqual(depth.dtype, np.float32)

    def test_deterministic(self):
        d1 = run_fallback_depth(self.image)
        d2 = run_fallback_depth(self.image)
        np.testing.assert_array_equal(d1, d2)

    def test_within_configured_bounds(self):
        depth = run_fallback_depth(self.image, min_value=0.3, max_value=1.0)
        self.assertGreaterEqual(float(depth.min()), 0.3 - 1e-5)
        self.assertLessEqual(float(depth.max()), 1.0 + 1e-5)

    def test_grayscale_input_supported(self):
        gray = (np.random.default_rng(1).random((10, 10)) * 255).astype(np.uint8)
        depth = run_fallback_depth(gray)
        self.assertEqual(depth.shape, (10, 10))


if __name__ == "__main__":
    unittest.main()
