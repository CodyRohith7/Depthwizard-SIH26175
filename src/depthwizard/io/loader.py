"""load_image(): the single entry point for reading PNG/JPG/TIFF/GeoTIFF.

Behavior summary (see tests/test_io.py for the executable spec):
    - .png/.jpg/.jpeg -> always loaded via Pillow, always non-georeferenced.
    - .tif/.tiff, rasterio installed -> loaded via rasterio; CRS/transform
      read if present. No CRS in the file -> non-georeferenced, clearly
      noted (not an error).
    - .tif/.tiff, rasterio NOT installed -> loaded via Pillow as a plain
      raster; non-georeferenced, and a note explicitly says georeferencing
      could not be checked (this is different from "this file has no CRS"
      and callers/tests must be able to tell the two apart).
    - corrupted / unreadable file -> UnsupportedImageError, chained from
      the underlying library exception. Never a silent fallback.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple

from depthwizard.io.metadata import RasterMetadata
from depthwizard.io.exceptions import UnsupportedImageError
from depthwizard.logging_setup import get_logger

log = get_logger("depthwizard.io.loader")

_TIFF_EXTS = {".tif", ".tiff"}
_PLAIN_EXTS = {".png", ".jpg", ".jpeg"}

try:
    import rasterio  # type: ignore
    from rasterio.errors import RasterioIOError  # type: ignore

    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover - exercised in environments without rasterio
    rasterio = None  # type: ignore
    RasterioIOError = Exception  # type: ignore
    _HAS_RASTERIO = False


def rasterio_available() -> bool:
    """Whether the optional `rasterio` dependency is importable in this
    environment. Exposed so callers/tests can assert on it explicitly
    instead of guessing from behavior."""
    return _HAS_RASTERIO


def load_image(path: str | Path) -> Tuple[np.ndarray, RasterMetadata]:
    """Load an image file and return (array, RasterMetadata).

    Raises:
        UnsupportedImageError: file does not exist, is not a readable
            image, or is corrupted.
    """
    path = Path(path)
    if not path.exists():
        raise UnsupportedImageError(f"File does not exist: {path}")

    ext = path.suffix.lower()

    if ext in _TIFF_EXTS:
        if _HAS_RASTERIO:
            return _load_with_rasterio(path)
        log.warning(
            "rasterio is not installed; loading %s as a plain raster via "
            "Pillow. Any embedded CRS/geotransform CANNOT be checked and "
            "will NOT be used, even if present in the file.",
            path,
        )
        return _load_plain(
            path,
            extra_notes=(
                "rasterio not installed -- GeoTIFF CRS/transform tags could "
                "not be checked (this is NOT the same as the file having no "
                "CRS; install the `geo` extra to check for real).",
            ),
        )

    if ext in _PLAIN_EXTS:
        return _load_plain(path)

    # Unknown extension: still try Pillow, since content matters more than
    # the extension, but be explicit that this path is unvalidated.
    log.warning("Unrecognized extension %r for %s; attempting best-effort load.", ext, path)
    return _load_plain(path)


def _load_plain(path: Path, extra_notes: tuple = ()) -> Tuple[np.ndarray, RasterMetadata]:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as im:
            im = im.convert("RGB") if im.mode not in ("RGB", "L") else im
            arr = np.asarray(im)
    except (UnidentifiedImageError, OSError) as exc:
        raise UnsupportedImageError(f"Could not read image {path}: {exc}") from exc

    if arr.ndim == 2:
        band_count = 1
    else:
        band_count = arr.shape[2]

    meta = RasterMetadata(
        source_path=str(path),
        width=arr.shape[1],
        height=arr.shape[0],
        band_count=band_count,
        dtype=str(arr.dtype),
        is_georeferenced=False,
        notes=extra_notes,
    )
    return arr, meta


def _load_with_rasterio(path: Path) -> Tuple[np.ndarray, RasterMetadata]:
    try:
        with rasterio.open(path) as ds:  # type: ignore[union-attr]
            arr = ds.read()  # (bands, H, W)
            crs = ds.crs
            transform = ds.transform
            nodata = ds.nodata
            bounds = ds.bounds
            width, height = ds.width, ds.height
            band_count = ds.count
            dtype = str(arr.dtype)
    except RasterioIOError as exc:
        raise UnsupportedImageError(f"Could not read GeoTIFF {path}: {exc}") from exc

    # (bands, H, W) -> (H, W, bands) for consistency with the plain-image path,
    # squeezing to (H, W) for single-band rasters.
    arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 1:
        arr = arr[..., 0]

    has_crs = crs is not None
    # rasterio's identity/None transform for a non-georeferenced TIFF is
    # Affine(1, 0, 0, 0, 1, 0) -- treat that (no CRS + identity transform) as
    # "not georeferenced" too, since it carries no real-world information.
    transform6 = (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
    is_identity = transform6 == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    is_georeferenced = has_crs and not is_identity

    notes = ()
    if not has_crs:
        notes = ("File has no embedded CRS -- treated as non-georeferenced.",)
    elif is_identity:
        notes = ("File has a CRS but an identity transform -- treated as non-georeferenced.",)

    meta = RasterMetadata(
        source_path=str(path),
        width=width,
        height=height,
        band_count=band_count,
        dtype=dtype,
        is_georeferenced=is_georeferenced,
        crs_wkt=crs.to_wkt() if has_crs else None,
        transform=transform6 if is_georeferenced else None,
        nodata=nodata,
        bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top) if is_georeferenced else None,
        notes=notes,
    )
    return arr, meta
