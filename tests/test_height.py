import unittest

import numpy as np

from depthwizard.height.measurement import relative_height_between_points


class TestRelativeHeightBetweenPoints(unittest.TestCase):
    def setUp(self):
        self.elevation = np.array(
            [[0.0, 0.2, 0.4], [0.6, 0.8, 1.0], [0.1, 0.3, 0.5]], dtype=np.float32
        )

    def test_uncalibrated_difference(self):
        result = relative_height_between_points(self.elevation, (1, 1), (0, 0))
        self.assertAlmostEqual(result["elevation_a"], 0.8)
        self.assertAlmostEqual(result["elevation_b"], 0.0)
        self.assertAlmostEqual(result["relative_difference"], 0.8)
        self.assertFalse(result["calibrated"])
        self.assertEqual(result["unit"], "relative")
        self.assertNotIn("meters_difference", result)

    def test_calibrated_requires_source(self):
        with self.assertRaises(ValueError):
            relative_height_between_points(self.elevation, (0, 0), (1, 1), meters_per_unit=10.0)

    def test_calibrated_with_source(self):
        result = relative_height_between_points(
            self.elevation, (1, 1), (0, 0), meters_per_unit=10.0, calibration_source="test-anchor"
        )
        self.assertTrue(result["calibrated"])
        self.assertEqual(result["unit"], "meters")
        self.assertAlmostEqual(result["meters_difference"], 8.0, places=4)
        self.assertEqual(result["calibration_source"], "test-anchor")

    def test_out_of_bounds_raises(self):
        with self.assertRaises(ValueError):
            relative_height_between_points(self.elevation, (0, 0), (99, 99))


if __name__ == "__main__":
    unittest.main()
