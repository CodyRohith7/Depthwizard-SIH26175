"""Pure affine coordinate-transform tests -- no external dependencies.

NOTE ON TEST RUNNER: pytest is declared in requirements.txt/pyproject.toml
but is NOT installed in the environment these tests were authored and run
in (no network access to install it there). Every test in this file is
written as a unittest.TestCase so it is executable right now via
`python -m unittest discover`, and it will also be picked up automatically
by a real `pytest` run once that is installed -- pytest natively discovers
and runs unittest.TestCase subclasses.
"""
import unittest

from depthwizard.io.coordinates import pixel_to_geo, geo_to_pixel, round_trip_error
from depthwizard.io.exceptions import CoordinateTransformError


class TestPixelToGeo(unittest.TestCase):
    def test_identity_transform(self):
        transform = (1, 0, 0, 0, 1, 0)
        x, y = pixel_to_geo(transform, 10, 20)
        self.assertEqual((x, y), (10, 20))

    def test_known_affine(self):
        # 30m pixel size, origin (top-left) at UTM-like (500000, 4649000),
        # y decreasing with row (standard north-up raster convention).
        transform = (30.0, 0.0, 500000.0, 0.0, -30.0, 4649000.0)
        x, y = pixel_to_geo(transform, col=10, row=5)
        self.assertAlmostEqual(x, 500000.0 + 10 * 30.0)
        self.assertAlmostEqual(y, 4649000.0 - 5 * 30.0)


class TestGeoToPixel(unittest.TestCase):
    def test_inverts_known_affine(self):
        transform = (30.0, 0.0, 500000.0, 0.0, -30.0, 4649000.0)
        col, row = geo_to_pixel(transform, x=500300.0, y=4648850.0)
        self.assertAlmostEqual(col, 10.0)
        self.assertAlmostEqual(row, 5.0)

    def test_singular_transform_raises(self):
        # a*e - b*d == 0
        transform = (0.0, 0.0, 100.0, 0.0, 0.0, 200.0)
        with self.assertRaises(CoordinateTransformError):
            geo_to_pixel(transform, 1.0, 1.0)


class TestRoundTrip(unittest.TestCase):
    def test_round_trip_axis_aligned(self):
        transform = (30.0, 0.0, 500000.0, 0.0, -30.0, 4649000.0)
        for col, row in [(0, 0), (100, 200), (17.5, 3.25), (999, 1)]:
            err = round_trip_error(transform, col, row)
            self.assertLess(err, 1e-6, f"round trip error too large at ({col},{row}): {err}")

    def test_round_trip_with_rotation_and_skew(self):
        # A transform with rotation/shear terms (b, d nonzero) -- still
        # must round-trip exactly through the 2x2 linear solve.
        transform = (25.0, 3.0, 10000.0, -2.0, -24.0, 20000.0)
        for col, row in [(0, 0), (50, 50), (123.4, 56.7)]:
            err = round_trip_error(transform, col, row)
            self.assertLess(err, 1e-6, f"round trip error too large at ({col},{row}): {err}")


if __name__ == "__main__":
    unittest.main()
