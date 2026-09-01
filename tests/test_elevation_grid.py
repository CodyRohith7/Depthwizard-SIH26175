import unittest

import numpy as np

from depthwizard.reconstruction.elevation_grid import build_elevation_grid


class TestBuildElevationGrid(unittest.TestCase):
    def test_rejects_non_2d(self):
        with self.assertRaises(ValueError):
            build_elevation_grid(
                np.zeros((2, 2, 3), dtype=np.float32),
                depth_source="depth-anything-v2-small",
                depth_status="SUCCESS",
            )

    def test_shape_and_provenance_are_preserved(self):
        elevation = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        grid = build_elevation_grid(
            elevation, depth_source="depth-anything-v2-small", depth_status="SUCCESS",
        )
        self.assertEqual(grid["rows"], 2)
        self.assertEqual(grid["cols"], 3)
        self.assertEqual(grid["depth_source"], "depth-anything-v2-small")
        self.assertEqual(grid["depth_status"], "SUCCESS")
        np.testing.assert_array_equal(grid["elevation"], elevation)

    def test_uncalibrated_by_default(self):
        elevation = np.ones((4, 4), dtype=np.float32)
        grid = build_elevation_grid(
            elevation, depth_source="depth-anything-v2-small", depth_status="SUCCESS",
        )
        self.assertFalse(grid["calibrated"])
        self.assertEqual(grid["unit"], "relative")
        self.assertIsNone(grid["calibration_source"])

    def test_calibrated_requires_a_source(self):
        elevation = np.ones((4, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            build_elevation_grid(
                elevation, depth_source="depth-anything-v2-small", depth_status="SUCCESS",
                calibrated=True,
            )

    def test_calibrated_with_source_sets_meters_unit(self):
        elevation = np.ones((4, 4), dtype=np.float32)
        grid = build_elevation_grid(
            elevation, depth_source="depth-anything-v2-small", depth_status="SUCCESS",
            calibrated=True, calibration_source="ground-control-points",
        )
        self.assertTrue(grid["calibrated"])
        self.assertEqual(grid["unit"], "meters")
        self.assertEqual(grid["calibration_source"], "ground-control-points")

    def test_finite_and_nodata_masks_are_complementary(self):
        elevation = np.array([[0.0, np.nan], [1.0, np.inf]], dtype=np.float32)
        grid = build_elevation_grid(
            elevation, depth_source="depth-anything-v2-small", depth_status="SUCCESS",
        )
        np.testing.assert_array_equal(
            grid["finite_mask"], np.array([[True, False], [True, False]])
        )
        np.testing.assert_array_equal(
            grid["nodata_mask"], np.array([[False, True], [False, True]])
        )

    def test_valid_fraction_reflects_hole_ratio(self):
        elevation = np.array([[0.0, np.nan], [1.0, 2.0]], dtype=np.float32)
        grid = build_elevation_grid(
            elevation, depth_source="depth-anything-v2-small", depth_status="SUCCESS",
        )
        self.assertAlmostEqual(grid["valid_fraction"], 0.75)

    def test_all_finite_gives_valid_fraction_one(self):
        elevation = np.zeros((5, 5), dtype=np.float32)
        grid = build_elevation_grid(
            elevation, depth_source="depth-anything-v2-small", depth_status="SUCCESS",
        )
        self.assertAlmostEqual(grid["valid_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
