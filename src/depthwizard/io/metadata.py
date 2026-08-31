"""RasterMetadata: the clean metadata object every /io loader returns.

Deliberately plain (a frozen dataclass of primitives + tuples) so it is
trivially serializable (json.dumps(asdict(meta))) and has no hard
dependency on rasterio's own types -- callers that never installed
rasterio can still consume it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple


@dataclass(frozen=True)
class RasterMetadata:
    """Metadata for one loaded raster (plain image or GeoTIFF).

    Attributes:
        source_path: path the image was loaded from.
        width, height: pixel dimensions.
        band_count: number of bands (3 for typical RGB, 1 for grayscale, etc).
        dtype: numpy dtype name of the loaded array (e.g. "uint8").
        is_georeferenced: True only when a valid CRS *and* affine transform
            were both found. This is the single field callers should branch
            on to decide whether the georeferenced (SRTM/GCP) height
            pipeline or the non-georeferenced pipeline applies.
        crs_wkt: the coordinate reference system as WKT text, or None.
        transform: the 6-parameter affine (a, b, c, d, e, f) such that
            x = a*col + b*row + c
            y = d*col + e*row + f
            or None when not georeferenced. This is rasterio/GDAL's affine
            convention (Affine.a..f), stored as a plain tuple so it has no
            rasterio dependency.
        nodata: the raster's nodata value, or None if not set / not applicable.
        bounds: (minx, miny, maxx, maxy) in the raster's CRS, or None.
        notes: human-readable caveats about how this metadata was derived
            (e.g. "rasterio not installed -- GeoTIFF tags could not be
            checked" vs "file has no embedded CRS"). Always check `notes`
            before treating `is_georeferenced=False` as "definitely a plain
            photo" -- it may instead mean the check itself could not run.
    """

    source_path: str
    width: int
    height: int
    band_count: int
    dtype: str
    is_georeferenced: bool
    crs_wkt: Optional[str] = None
    transform: Optional[Tuple[float, float, float, float, float, float]] = None
    nodata: Optional[float] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return asdict(self)
