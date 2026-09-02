# DepthWizard

Turns a single optical image into an interactive, explorable 3D elevation surface.

## Problem

[SIH26175](https://sih-explorer.amanuniyal47.workers.dev/problem/SIH26175) asks for single-view height estimation and a 3D fly-through: given one optical (aerial/satellite-style) image, estimate elevation/height information and let a user explore the result in 3D, without requiring stereo pairs, LiDAR, or a second viewpoint.

## Solution

DepthWizard runs a single uploaded image through a monocular depth model (Depth Anything V2), reinterprets the resulting relative depth map as a relative elevation surface, reconstructs that surface as a real triangle mesh (not a scattered point cloud), and renders it in an interactive, embedded 3D viewer. A point-to-point measurement tool reports the relative elevation difference between two pixels, with an optional manual scale factor if the user supplies an independent metric reference.

Every output is explicitly labeled relative/uncalibrated unless a real external calibration source is supplied — see [Scientific Limitations](#scientific-limitations).

## Key Capabilities

- Single-view depth inference (Depth Anything V2, with an explicit, always-labeled `DEMO_FALLBACK` path when the real model can't load — never silently substituted)
- Relative elevation reconstruction from the depth output
- Structured surface mesh: a canonical elevation grid, hole-aware triangulation, degenerate-triangle rejection, per-vertex normals — a real surface, not a point cloud
- Interactive 3D exploration: orbit, zoom, pan, RGB/elevation/wireframe coloring modes, a scripted Auto-Fly camera path, Fit/Reset
- Point-to-point relative elevation measurement, with optional manual metric scaling
- Scene JSON export (full mesh + provenance) and, for georeferenced input with `rasterio` installed, a relative-elevation GeoTIFF export
- Geospatial-ready input architecture: GeoTIFF I/O with CRS/transform handling when `rasterio` is available, degrading gracefully to plain raster I/O when it isn't

## Architecture

```
Input (PNG / JPG / TIFF / GeoTIFF)
        |
        v
Image / GeoTIFF I/O            depthwizard.io
        |
        v
Depth Inference                 depthwizard.depth
  (Depth Anything V2, or
   explicit DEMO_FALLBACK)
        |
        v
Elevation Grid                  depthwizard.reconstruction.dsm / .elevation_grid
  (relative depth -> relative
   elevation, finite/nodata
   masks, provenance)
        |
        v
Surface Mesh                    depthwizard.reconstruction.mesh
  (structured decimation,
   hole-aware triangulation,
   vertex normals + color)
        |
        v
3D Visualization                embedded Three.js viewer
  (orbit / zoom / pan /
   RGB / Elevation / Wireframe /
   Fit / Reset / Auto-Fly)
        |
        v
Measurement / Export             depthwizard.height, .reconstruction.scene_export
  (relative elevation diff,      (scene JSON, relative-elevation GeoTIFF)
   optional manual metric scale)
```

**Relative vs. metric path.** Nothing in this pipeline derives a metric scale on its own. The depth model outputs relative depth (correct ordering, not an absolute distance), so every downstream elevation value is relative to the current image only — not comparable across images and never shown as a metric height by default. The one exception is the point-to-point measurement tool: if the user supplies an independent, trusted meters-per-unit reference for the current scene, that single measurement (and only that measurement) is converted to meters and explicitly labeled `Metric`, with its source shown. GeoTIFF export follows the same rule — it writes *relative* elevation values, labeled as such, even when the input image itself is georeferenced (georeferencing gives you correct X/Y map coordinates; it does not by itself give you a metric Z/elevation scale for a monocular depth output).

## Technical Stack

- Python 3.10+
- [Streamlit](https://streamlit.io/) — application UI
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) via `transformers` + `torch` — monocular depth inference
- NumPy, OpenCV (`opencv-python-headless`), Pillow — image and array processing
- [rasterio](https://rasterio.readthedocs.io/) (optional) — GeoTIFF I/O with real CRS/transform handling
- [Three.js](https://threejs.org/) (loaded from a CDN via an import map, embedded through `streamlit.components.v1.html`) — the 3D viewer
- `pytest` / `unittest` — testing

## How It Works

1. **Upload.** The user uploads an image (PNG, JPG, TIFF, or GeoTIFF) in the sidebar. `depthwizard.io.load_image` reads it; if `rasterio` is installed and the file carries a CRS/transform, that georeferencing is captured (`is_georeferenced=True`) and used later for GeoTIFF export. Without `rasterio`, TIFF/GeoTIFF files are still read as plain rasters, with a visible note that embedded georeferencing was not checked.
2. **Depth inference.** `depthwizard.depth.DepthPredictor` runs Depth Anything V2 on the image. If the real backend can't load (missing `torch`/`transformers`, or a real failure) and the user has explicitly allowed it, an explicit `DEMO_FALLBACK` depth map is returned instead and labeled as such everywhere it appears — it is never presented as a real prediction. If fallback isn't allowed, the failure is shown as an error.
3. **Elevation grid.** `depthwizard.reconstruction.dsm.depth_to_relative_elevation` reinterprets the relative depth map as a relative elevation surface (with an invertible "closer-to-camera = higher elevation" assumption, since that's the common case for a roughly-overhead shot). `elevation_grid.build_elevation_grid` wraps this into a canonical structured representation with finite/nodata masks and provenance (depth source, status, calibration state).
4. **Surface mesh.** `reconstruction.mesh.build_terrain_mesh` decimates the elevation grid onto a regular sub-grid (a stride, not a resize/blur, so triangle connectivity stays well-defined), connects every 2x2 block of *finite* neighboring samples into two triangles (so holes in the input are never bridged), rejects degenerate/outlier triangles, and computes per-vertex normals and two color buffers (source RGB and elevation-mapped). The result is a real mesh, not a point cloud.
5. **3D visualization.** The mesh ships to the browser as JSON and renders as an indexed, lit `THREE.Mesh` with camera framing and clipping planes derived from the mesh's own bounding-sphere radius (not fixed constants), so the view adapts to any input scale. RGB / Elevation / Wireframe are cheap client-side toggles; Auto-Fly is a deterministic scripted orbit, not physically-simulated flight or collision-aware navigation.
6. **Measurement.** `depthwizard.height.measurement.relative_height_between_points` reports the relative elevation difference between two user-chosen pixels. It stays unitless/relative unless the user supplies an external meters-per-unit reference, in which case the result is converted and clearly labeled `Metric` with its source.
7. **Export.** The full mesh scene (vertices, triangles, normals, both color buffers, provenance) exports as JSON. If the input was georeferenced and `rasterio` is installed, a single-band relative-elevation GeoTIFF export is also available.

## Scientific Limitations

- **Monocular depth is inherently ambiguous.** A single 2D image does not fully constrain 3D geometry; the depth model is making a learned estimate, not measuring anything directly.
- **Non-georeferenced input produces relative output only.** Without an external reference, elevation values have no absolute meaning and are not comparable across different images or runs.
- **Metric elevation requires a valid external calibration/reference.** This pipeline does not derive a metric scale on its own anywhere. GeoTIFF export writes relative elevation values even for georeferenced input, since correct map X/Y coordinates do not imply a correct elevation (Z) scale from a monocular depth output.
- **Unobserved geometry cannot be guaranteed.** The reconstruction only represents what the single input image shows; occluded surfaces, overhangs, and anything outside the frame are not and cannot be reconstructed.
- **The fly-through is a visualization aid, not a surveying tool.** Auto-Fly follows a scripted camera path for exploration; it is not a substitute for photogrammetric survey and makes no claim of collision-aware or physically simulated navigation.
- **CPU inference runtime may be significant.** Depth Anything V2 without a GPU can take tens of seconds per image; this is not a real-time system.

## Installation

Tested on Windows with Python 3.10+.

```powershell
git clone <repository-url>
cd DepthWizard

python -m venv .venv
.venv\Scripts\activate

pip install -e ".[dev,geo,depth-model]"
```

- `dev` installs Streamlit and the test tooling (required to run the app or the tests at all).
- `geo` installs `rasterio` for real GeoTIFF support (optional — the app still runs and reads plain rasters without it).
- `depth-model` installs `torch`/`transformers`/`safetensors` for real Depth Anything V2 inference (optional — without it, only the explicit `DEMO_FALLBACK` depth path is available).

## Run

```powershell
streamlit run app/streamlit_app.py
```

## Project Structure

```
app/streamlit_app.py               # the SIH-facing application (this is the deliverable)
tools/dev_streamlit_harness.py     # lighter internal dev tool -- io/depth inspection only
src/depthwizard/
  io/                              # image + GeoTIFF loading, coordinate transforms
  depth/                           # DepthPredictor, DEMO_FALLBACK, Depth Anything V2 backend
  reconstruction/
    dsm.py                        # relative depth -> relative elevation
    elevation_grid.py             # canonical structured elevation-grid representation
    mesh.py                       # elevation grid -> triangle-mesh surface
    placeholder_preview.py        # shared uncalibrated camera model
    scene_export.py               # mesh scene JSON / relative-elevation GeoTIFF export
  height/                         # point-to-point relative elevation measurement
  geometry/ scale/ uncertainty/   # documented, not-yet-implemented placeholders for
  evaluation/                     # future milestones -- see Roadmap
tests/                             # unittest suite (68 tests as of this README)
data/fixtures/                     # sample input used by tests
configs/                           # YAML config (depth backend, logging, reconstruction defaults)
scripts/make_test_geotiff.py       # generates a synthetic GeoTIFF fixture for manual testing
prototype/                         # superseded pre-audit MVP, kept for historical reference only
video/                             # demo video script and shot list
```

## Testing

68 tests passing (`unittest`, includes coverage for I/O, coordinate transforms, depth inference and fallback labeling, elevation-grid construction, mesh generation/triangle connectivity/hole handling/degenerate-triangle rejection, height measurement, and scene export), verified by running:

```powershell
set PYTHONPATH=src
python -m unittest discover -s tests
```

This count reflects what was actually run and passing at the time this README was written; re-run the command above to confirm the current number.

## Demo

*Video link to be added here once the demonstration video is recorded — see `video/shot_list.md` and `video/prototype_explainer_script.txt` for the planning material.*

## Source Code

*Repository URL to be added here once published.*

## Roadmap

- Geospatial metric calibration (a real `scale`/`geometry` path — currently documented placeholder modules, not implemented)
- SRTM / ground-control-point integration for validating or anchoring elevation output
- Aerial-domain-specific model refinement
- Formal uncertainty estimation (the current `uncertainty` module is a documented placeholder)
- GPU-accelerated inference for faster turnaround

None of the above is implemented today; the corresponding `src/depthwizard/geometry`, `scale`, `uncertainty`, and `evaluation` packages exist as explicitly-labeled "not yet implemented" placeholders so importing them can't be mistaken for real functionality.

## License

No license has been chosen for this project yet. Licensing needs to be decided before this repository is made public or reused.
