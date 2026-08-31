"""Image and geospatial I/O.

Milestone M1 scope (implemented):
    - load_image(path): PNG/JPG/TIFF/GeoTIFF -> (ndarray, RasterMetadata)
    - pixel_to_geo / geo_to_pixel: pure affine coordinate transforms

Deferred to later milestones:
    - EXIF focal-length/sensor extraction (see Phase 0 audit Section 8,
      /geometry) -- not implemented here yet.
    - SRTM/GCP sampling (see /scale, future milestone).

GeoTIFF support requires the optional `rasterio` dependency. When it is not
installed, TIFF files are still readable via Pillow but are always treated
as non-georeferenced (see loader.py's docstring for exactly how this is
surfaced -- it is never silently treated as "no CRS in the file").
"""
from depthwizard.io.metadata import RasterMetadata
from depthwizard.io.loader import load_image
from depthwizard.io.coordinates import pixel_to_geo, geo_to_pixel
from depthwizard.io.exceptions import DepthWizardIOError, UnsupportedImageError

__all__ = [
    "RasterMetadata",
    "load_image",
    "pixel_to_geo",
    "geo_to_pixel",
    "DepthWizardIOError",
    "UnsupportedImageError",
]
