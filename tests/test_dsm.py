import unittest

import numpy as np

from depthwizard.reconstruction.dsm import depth_to_relative_elevation, elevation_stats


class TestDepthToRelativeElevation(unittest.TestCase):
    def test_invert_default(self):
        depth = np.array([[0.0, 1.0], [0.25, 0.75]], dtype=np.float32)
        elevation = depth_to_relative_elevation(depth)
        np.testing.assert_allclose(elevation, np.array([[1.0, 0.0], [0.75, 0.25]], dtype=np.float32))

    def test_no_invert(self):
        depth = np.array([[0.0, 1.0], [0.25, 0.75]], dtype=np.float32)
        elevation = depth_to_relative_elevation(depth, invert=False)
        np.testing.assert_allclose(elevation, depth)

    def test_dtype_is_float32(self):
        depth = np.zeros((4, 4), dtype=np.float64)
        elevation = depth_to_relative_elevation(depth)
        self.assertEqual(elevation.dtype, np.float32)

    def test_rejects_non_2d(self):
        with self.assertRaises(ValueError):
            depth_to_relative_elevation(np.zeros((2, 2, 3), dtype=np.float32))


class TestElevationStats(unittest.TestCase):
    def test_stats_are_real_computed_values(self):
        elevation = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        stats = elevation_stats(elevation)
        self.assertAlmostEqual(stats["min"], 0.0)
        self.assertAlmostEqual(stats["max"], 1.0)
        self.assertAlmostEqual(stats["mean"], 0.5)
        self.assertFalse(stats["calibrated"])
        self.assertEqual(stats["unit"], "relative")


if __name__ == "__main__":
    unittest.main()
