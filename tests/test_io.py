"""Tests for depthwizard.io.loader.

Environment note (see tests/test_coordinates.py for the unittest-vs-pytest
rationale): `rasterio` is NOT installed in the environment these tests were
authored and executed in (no network access to install it). Two
consequences, both handled explicitly rather than silently:

    1. Tests that only need a plain PNG/JPG/TIFF run for real, right now,
       against the actual _load_plain() code path.
    2. Tests of the real-GeoTIFF (rasterio) code path use
       unittest.mock.patch to inject a fake rasterio module into
       depthwizard.io.loader and call the private _load_with_rasterio()
       function directly. This tests the *logic* of the georeferenced
       branch (CRS/transform/nodata/bounds extraction, the missing-CRS and
       identity-transform cases) even without a real rasterio installation
       -- but it is NOT a substitute for running these same tests again on
       a machine with rasterio installed against a real GeoTIFF fixture.
       scripts/make_test_geotiff.py is provided for that follow-up and is
       itself unverified here for the same reason.
"""
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from depthwizard.io.loader import load_image, rasterio_available
from depthwizard.io.exceptions import UnsupportedImageError


class TestPlainImageLoading(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _make_png(self, name="test.png", size=(16, 12)):
        arr = (np.random.default_rng(0).random((size[1], size[0], 3)) * 255).astype(np.uint8)
        path = Path(self.tmpdir.name) / name
        Image.fromarray(arr, mode="RGB").save(path)
        return path, arr

    def test_load_png_is_non_georeferenced(self):
        path, arr = self._make_png()
        loaded, meta = load_image(path)
        self.assertEqual(loaded.shape, arr.shape)
        self.assertFalse(meta.is_georeferenced)
        self.assertIsNone(meta.crs_wkt)
        self.assertIsNone(meta.transform)
        self.assertEqual(meta.band_count, 3)
        self.assertEqual(meta.width, 16)
        self.assertEqual(meta.height, 12)

    def test_load_jpg_is_non_georeferenced(self):
        arr = (np.random.default_rng(1).random((10, 20, 3)) * 255).astype(np.uint8)
        path = Path(self.tmpdir.name) / "test.jpg"
        Image.fromarray(arr, mode="RGB").save(path, quality=90)
        loaded, meta = load_image(path)
        self.assertEqual(loaded.shape, (10, 20, 3))
        self.assertFalse(meta.is_georeferenced)

    def test_plain_tiff_without_rasterio_is_explicitly_noted(self):
        # This exercises the REAL fallback-to-Pillow code path in this
        # environment, since rasterio genuinely is not installed here.
        if rasterio_available():
            self.skipTest("rasterio IS installed in this environment; this test "
                           "targets the rasterio-unavailable code path specifically.")
        arr = (np.random.default_rng(2).random((8, 8, 3)) * 255).astype(np.uint8)
        path = Path(self.tmpdir.name) / "test.tiff"
        Image.fromarray(arr, mode="RGB").save(path)
        loaded, meta = load_image(path)
        self.assertFalse(meta.is_georeferenced)
        self.assertTrue(
            any("rasterio not installed" in note for note in meta.notes),
            f"expected a note about rasterio being unavailable, got: {meta.notes}",
        )

    def test_missing_file_raises(self):
        with self.assertRaises(UnsupportedImageError):
            load_image(Path(self.tmpdir.name) / "does_not_exist.png")

    def test_corrupted_file_raises(self):
        path = Path(self.tmpdir.name) / "corrupt.png"
        with open(path, "wb") as fh:
            fh.write(b"this is not a real image file, just garbage bytes" * 4)
        with self.assertRaises(UnsupportedImageError):
            load_image(path)


class _FakeCRS:
    def __init__(self, wkt="FAKE_CRS_WKT"):
        self._wkt = wkt

    def to_wkt(self):
        return self._wkt


class _FakeDataset:
    """Stands in for a rasterio DatasetReader for logic testing."""

    def __init__(self, width=4, height=3, crs=None, transform=None, nodata=None,
                 bounds=None, band_count=3, array=None):
        self.width = width
        self.height = height
        self.crs = crs
        self.transform = transform or SimpleNamespace(a=1, b=0, c=0, d=0, e=1, f=0)
        self.nodata = nodata
        self.bounds = bounds or SimpleNamespace(left=0, bottom=0, right=width, top=height)
        self.count = band_count
        self._array = array if array is not None else np.zeros(
            (band_count, height, width), dtype=np.uint8
        )

    def read(self):
        return self._array

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestRasterioBranchLogic(unittest.TestCase):
    """Directly tests depthwizard.io.loader._load_with_rasterio via a fake
    rasterio module, independent of whether the real package is installed."""

    def _patched_loader(self, dataset: _FakeDataset):
        import depthwizard.io.loader as loader_mod

        fake_rasterio = SimpleNamespace(open=lambda path: dataset)
        return mock.patch.multiple(
            loader_mod,
            rasterio=fake_rasterio,
            RasterioIOError=IOError,
        )

    def test_georeferenced_geotiff(self):
        import depthwizard.io.loader as loader_mod

        transform = SimpleNamespace(a=30.0, b=0.0, c=500000.0, d=0.0, e=-30.0, f=4649000.0)
        ds = _FakeDataset(width=100, height=80, crs=_FakeCRS(), transform=transform)
        with self._patched_loader(ds):
            arr, meta = loader_mod._load_with_rasterio(Path("fake.tif"))

        self.assertTrue(meta.is_georeferenced)
        self.assertEqual(meta.crs_wkt, "FAKE_CRS_WKT")
        self.assertEqual(meta.transform, (30.0, 0.0, 500000.0, 0.0, -30.0, 4649000.0))
        self.assertEqual(meta.bounds, (0, 0, 100, 80))  # (left, bottom, right, top) from the default fake bounds
        self.assertEqual(arr.shape, (80, 100, 3))

    def test_missing_crs_is_non_georeferenced_not_an_error(self):
        import depthwizard.io.loader as loader_mod

        transform = SimpleNamespace(a=30.0, b=0.0, c=500000.0, d=0.0, e=-30.0, f=4649000.0)
        ds = _FakeDataset(width=10, height=10, crs=None, transform=transform)
        with self._patched_loader(ds):
            arr, meta = loader_mod._load_with_rasterio(Path("fake.tif"))

        self.assertFalse(meta.is_georeferenced)
        self.assertIsNone(meta.transform)
        self.assertTrue(any("no embedded CRS" in note for note in meta.notes))

    def test_identity_transform_is_non_georeferenced(self):
        import depthwizard.io.loader as loader_mod

        ds = _FakeDataset(width=10, height=10, crs=_FakeCRS())  # default identity transform
        with self._patched_loader(ds):
            arr, meta = loader_mod._load_with_rasterio(Path("fake.tif"))

        self.assertFalse(meta.is_georeferenced)
        self.assertTrue(any("identity transform" in note for note in meta.notes))


if __name__ == "__main__":
    unittest.main()
