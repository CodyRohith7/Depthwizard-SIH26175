"""Pure affine pixel<->geographic coordinate transforms.

Deliberately implemented with no dependency on rasterio: the transform is
just a 6-parameter affine map, and testing that math correctly does not
require the ability to open a real GeoTIFF. `loader.py` is what actually
extracts a real transform from a file (via rasterio, when installed) and
hands it to these functions.

Convention (matches rasterio/GDAL's Affine.a..f):
    x = a*col + b*row + c
    y = d*col + e*row + f

where (col, row) is (pixel_x, pixel_y) i.e. (u, v), and (x, y) is the
projected/geographic coordinate.
"""
from __future__ import annotations

from typing import Tuple

from depthwizard.io.exceptions import CoordinateTransformError

Affine6 = Tuple[float, float, float, float, float, float]


def pixel_to_geo(transform: Affine6, col: float, row: float) -> Tuple[float, float]:
    """Map pixel (col, row) -> geographic/projected (x, y).

    `col` and `row` may be fractional (e.g. the center of a pixel is
    typically col+0.5, row+0.5 -- callers decide the convention; this
    function does not add any offset itself).
    """
    a, b, c, d, e, f = transform
    x = a * col + b * row + c
    y = d * col + e * row + f
    return x, y


def geo_to_pixel(transform: Affine6, x: float, y: float) -> Tuple[float, float]:
    """Map geographic/projected (x, y) -> pixel (col, row).

    This is the exact algebraic inverse of pixel_to_geo: solves the 2x2
    linear system
        [a b] [col]   [x - c]
        [d e] [row] = [y - f]
    directly, rather than depending on rasterio's `~transform` operator, so
    it works even when rasterio is not installed.

    Raises:
        CoordinateTransformError: if the transform is singular (determinant
            ~= 0), which would happen for a degenerate/invalid transform.
    """
    a, b, c, d, e, f = transform
    det = a * e - b * d
    if abs(det) < 1e-12:
        raise CoordinateTransformError(
            f"Affine transform {transform!r} is singular (determinant={det!r}); "
            "cannot invert pixel<->geo mapping."
        )
    dx = x - c
    dy = y - f
    col = (e * dx - b * dy) / det
    row = (a * dy - d * dx) / det
    return col, row


def round_trip_error(transform: Affine6, col: float, row: float) -> float:
    """Convenience for tests: pixel -> geo -> pixel and return the max
    absolute error in pixel units. Should be ~0 for any valid transform."""
    x, y = pixel_to_geo(transform, col, row)
    col2, row2 = geo_to_pixel(transform, x, y)
    return max(abs(col2 - col), abs(row2 - row))
