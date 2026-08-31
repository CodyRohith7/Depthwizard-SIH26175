#!/usr/bin/env python3
"""Generate a tiny real GeoTIFF fixture for manual/CI testing on a machine
that has rasterio installed.

STATUS: UNVERIFIED. Written in a development session where rasterio is not
installed and could not be installed (no PyPI access) -- so this script has
never actually been run. Review it before trusting it; it is provided as a
documented starting point, not a tested tool.

Intended usage once rasterio is available:
    pip install -e ".[geo]"
    python3 scripts/make_test_geotiff.py
    # writes data/fixtures/test_geotiff.tif, a small synthetic raster with
    # a known CRS (EPSG:32643, UTM 43N) and a known affine transform, for
    # use as a real (not mocked) fixture in tests/test_io.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS
except ImportError as exc:
    raise SystemExit(
        "rasterio is not installed -- install the `geo` extra "
        "(pip install -e '.[geo]') to run this script."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    out_path = REPO_ROOT / "data" / "fixtures" / "test_geotiff.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 32, 24
    pixel_size = 30.0  # meters, matches SRTM's native resolution
    origin_x, origin_y = 500000.0, 4649000.0  # arbitrary UTM-like origin

    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)
    crs = CRS.from_epsg(32643)  # WGS84 / UTM zone 43N -- roughly India

    data = (np.random.default_rng(0).random((3, height, width)) * 255).astype(np.uint8)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
