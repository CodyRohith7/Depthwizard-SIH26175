# DepthWizard

**Single-View DSM Reconstruction & 3D Exploration**

DepthWizard is an engineering prototype developed for Smart India Hackathon 2026. It processes a single optical image or supported GeoTIFF to produce a relative depth field, a structured elevation surface, a continuous triangle mesh, and an interactive browser-based 3D visualization with point-to-point relative measurement capabilities.

---

### Hackathon Metadata

| Parameter | Details |
|---|---|
| **Competition** | Smart India Hackathon 2026 |
| **Problem Statement ID** | SIH26175 |
| **Title** | DepthWizard – Single-View Height Estimation and 3D Flythrough |
| **Organization** | Indian Space Research Organisation (ISRO) |
| **Category** | Software |
| **Repository** | [https://github.com/CodyRohith7/Depthwizard-SIH26175](https://github.com/CodyRohith7/Depthwizard-SIH26175) |

---

## Overview

Traditional digital surface model (DSM) generation typically relies on stereoscopic optical imagery, multi-view photogrammetry, airborne LiDAR, or radar interferometry (InSAR). These methods require synchronized multi-pass acquisitions or specialized sensor hardware.

DepthWizard explores single-view surface reconstruction from a single optical image:
1. **Input Ingestion**: Ingests standard optical rasters (PNG, JPEG) or georeferenced GeoTIFFs, extracting spatial extent, coordinate reference systems (CRS), and affine transforms when available.
2. **Relative Depth Inference**: Uses the Depth Anything V2 Small transformer model to infer a dense monocular relative depth map.
3. **Structured Elevation Grid**: Transforms the relative depth field into an inverted relative elevation surface with finite/nodata masks and complete provenance metadata.
4. **Surface Mesh Reconstruction**: Generates an indexed triangle mesh on a regular decimation grid, applying hole-aware triangulation, degenerate-triangle rejection, per-vertex normal computation, and dual-color buffers (RGB and elevation colormap).
5. **Interactive 3D Exploration**: Renders the reconstructed terrain mesh in an embedded WebGL/Three.js viewport supporting orbit, pan, zoom, camera framing, wireframe/elevation overlays, and scripted fly-through paths.
6. **Relative Measurement**: Calculates point-to-point relative elevation differences between user-selected pixel coordinates, with support for manual external metric scale factors when a known reference is supplied.

> [!NOTE]
> DepthWizard outputs a **relative (uncalibrated) elevation surface**. Monocular depth inference inherently contains scale and shift ambiguities. The prototype does not automatically produce globally accurate metric heights unless an external scale reference is explicitly provided by the user.

---

## Current Capabilities

The current codebase implements and validates the following functionality:

- **Monocular Depth Inference**: Dense relative depth estimation using `Depth Anything V2 Small` via PyTorch and Hugging Face Transformers. Includes an explicitly tagged fallback heuristic (`DEMO_FALLBACK`) when model weights or hardware acceleration are unavailable.
- **Structured Elevation Surface**: Canonical elevation grid representation (`ElevationGrid`) maintaining elevation values, finite valid masks, nodata flags, spatial metadata, and inference provenance.
- **Continuous Triangle Mesh Generation**: Structured surface meshing with regular-grid subsampling, hole-aware connectivity (preventing invalid triangles across nodata regions), per-face normal calculation, and per-vertex normal averaging.
- **Interactive 3D Viewport**: Embedded Three.js (r160) viewer utilizing ES module import maps, bounding-sphere camera auto-framing, OrbitControls, and responsive canvas sizing.
- **Rendering Modes**: Client-side toggles for source RGB texture, elevation-mapped pseudo-color, and wireframe overlay.
- **Exploration & Fly-Through**: Interactive manual camera manipulation (orbit, pan, zoom) alongside a scripted, deterministic circular fly-through mode.
- **Relative Elevation Measurement**: Interactive pixel-pair selection computing $\Delta Z_{\text{rel}}$, with manual scale factor support ($\text{meters} = \Delta Z_{\text{rel}} \times s$).
- **Geospatial Metadata Handling**: Optional GeoTIFF CRS and affine geotransform extraction via `rasterio`, preserving spatial reference metadata throughout the pipeline.
- **Export Formats**: Scene export to structured JSON (vertices, indices, normals, colors, metadata) and single-band relative-elevation GeoTIFFs (when georeferencing is present).
- **Automated Test Suite**: 68 unit and integration tests verifying I/O, coordinate transforms, depth inference contracts, elevation grid conversions, mesh topology, measurement math, and scene exports.

---

## Technical Pipeline

```
[ Optical Image / GeoTIFF ]
           │
           ▼
[ Input Validation & Metadata ] ────► depthwizard.io (load_image, metadata, CRS extraction)
           │
           ▼
[ Monocular Depth Inference ]   ────► depthwizard.depth (Depth Anything V2 / fallback)
           │
           ▼
[ Relative Depth Field ]        ────► Dense relative depth matrix (H × W, float32)
           │
           ▼
[ Structured Elevation Grid ]   ────► depthwizard.reconstruction.elevation_grid
           │
           ▼
[ Surface Mesh Reconstruction ] ────► depthwizard.reconstruction.mesh (triangulation, normals)
           │
           ▼
┌──────────────────────────────────────┴──────────────────────────────────────┐
│                                                                             │
▼                                                                             ▼
[ Interactive 3D Exploration ]                                  [ Analysis & Export ]
• Three.js WebGL viewport                                       • Point-to-point relative height
• RGB / Elevation / Wireframe modes                             • Scene JSON export
• Scripted auto-fly path                                        • Relative-elevation GeoTIFF
```

---

## Technology Stack

| Component | Technology | Version / Specification | Role in DepthWizard |
|---|---|---|---|
| **Core Runtime** | Python | `>= 3.10` | Primary application language and runtime environment |
| **Depth Model** | Depth Anything V2 | `Small (24.8M params)` | Dense monocular relative depth estimation |
| **ML Framework** | PyTorch / Transformers | `torch>=2.2`, `transformers>=4.45` | Model loading, tensor computation, and inference execution |
| **Array Processing** | NumPy | `>= 1.26` | Multi-dimensional grid manipulation, masks, and normal vectors |
| **Image Processing** | OpenCV / Pillow | `opencv-python-headless>=4.10`, `Pillow>=10` | Raster loading, resizing, color mapping, and array format conversions |
| **Geospatial I/O** | Rasterio | `>= 1.3` (Optional) | GeoTIFF reading/writing, CRS parsing, affine geotransform handling |
| **3D Rendering** | Three.js | `r160` (WebGL / ES Modules) | Client-side 3D terrain rendering, materials, lighting, and camera controls |
| **Application UI** | Streamlit | `>= 1.37` | Web-based interface, pipeline controls, diagnostics, and export actions |
| **Configuration** | PyYAML | `>= 6.0` | Pipeline settings, backend parameters, and logging configuration |
| **Testing** | unittest / pytest | `>= 8.0` | Automated regression testing and validation suite |

---

## Project Structure

```
DepthWizard/
├── .streamlit/
│   └── config.toml                  # Streamlit theme and server configuration
├── app/
│   └── streamlit_app.py             # Primary application interface and 3D viewer
├── configs/
│   ├── depth.yaml                   # Depth backend and fallback configuration
│   ├── logging.yaml                 # Logging format and level specifications
│   └── reconstruction.yaml          # Elevation grid and meshing parameters
├── data/
│   └── fixtures/                    # Test fixtures (synthetic test rasters)
├── presentation/
│   └── DepthWizard_Technical_Pipeline.svg # Vector architecture pipeline diagram
├── prototype/
│   ├── app.py                       # Historical prototype (retained for reference)
│   └── README.md                    # Historical prototype audit notes
├── scripts/
│   └── make_test_geotiff.py         # Synthetic georeferenced GeoTIFF generator
├── src/
│   └── depthwizard/                 # Core Python package
│       ├── depth/                   # Depth inference and predictor interfaces
│       │   ├── backends/            # Depth Anything V2 PyTorch/Transformers wrapper
│       │   ├── fallback.py          # Deterministic fallback heuristic
│       │   └── predictor.py         # Primary depth inference entrypoint
│       ├── height/                  # Height and elevation difference measurement
│       │   └── measurement.py       # Relative pixel-pair elevation delta computation
│       ├── io/                      # Raster and GeoTIFF I/O, coordinate mapping
│       │   ├── coordinates.py       # Pixel-to-geographic coordinate transformations
│       │   ├── loader.py            # Image/GeoTIFF loader with graceful degradation
│       │   └── metadata.py          # Spatial metadata dataclasses
│       ├── reconstruction/          # Elevation grids, surface meshing, and exports
│       │   ├── dsm.py               # Depth-to-relative-elevation conversion
│       │   ├── elevation_grid.py    # Structured canonical elevation grid container
│       │   ├── mesh.py              # Regular-grid terrain meshing and vertex normals
│       │   └── scene_export.py      # Mesh scene JSON and GeoTIFF export serialization
│       ├── geometry/                # (Planned) Geometric reasoning and plane fitting
│       ├── scale/                   # (Planned) Metric scale estimation modules
│       ├── uncertainty/             # (Planned) Uncertainty quantification modules
│       ├── evaluation/              # (Planned) Benchmark metrics and validation tools
│       └── logging_setup.py         # Centralized structured logging setup
├── tests/                           # Unit and integration test suite (68 tests)
│   ├── test_coordinates.py          # Coordinate transform tests
│   ├── test_depth.py                # DepthPredictor contract and exception tests
│   ├── test_dsm.py                  # Elevation conversion and statistics tests
│   ├── test_elevation_grid.py       # ElevationGrid validation tests
│   ├── test_fallback.py             # Fallback depth heuristic tests
│   ├── test_height.py               # Point-to-point measurement tests
│   ├── test_integration_m1.py       # End-to-end integration test
│   ├── test_io.py                   # Image and GeoTIFF loader tests
│   ├── test_mesh.py                 # Surface mesh and normal calculation tests
│   └── test_scene_export.py         # JSON and GeoTIFF export serialization tests
├── tools/
│   └── dev_streamlit_harness.py     # Developer inspection and sanity-check harness
├── video/
│   ├── prototype_explainer_script.txt # Demo explainer narration script
│   └── shot_list.md                 # Demo video shot list and scene breakdown
├── pyproject.toml                   # Package build metadata and dependencies
└── requirements.txt                 # Pinned dependencies file
```

---

## Running the Prototype

### Prerequisites

- Python `>= 3.10`
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/CodyRohith7/Depthwizard-SIH26175.git
   cd Depthwizard-SIH26175
   ```

2. **Create and activate a virtual environment:**
   - On Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - On Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   - *Minimal installation (Core + UI + Tests):*
     ```bash
     pip install -e ".[dev]"
     ```
   - *Full installation (Core + Depth Anything V2 + GeoTIFF support):*
     ```bash
     pip install -e ".[dev,geo,depth-model]"
     ```

### Launching the Application

Run the Streamlit application from the repository root:

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in a modern web browser supporting WebGL.

### Supported Input Formats

- **Standard Optical Rasters**: PNG (`.png`), JPEG (`.jpg`, `.jpeg`)
- **Geospatial Rasters**: TIFF / GeoTIFF (`.tif`, `.tiff`) — full georeferencing metadata extracted when `rasterio` is installed.

---

## Testing

The test suite covers core I/O, coordinate transformations, depth predictor contracts, structured elevation grid conversion, mesh triangulation, hole handling, height measurement arithmetic, and scene export serialization.

Run the test suite using Python's standard `unittest` discovery:

```bash
python -m unittest discover -s tests
```

**Current Test Status**: `68/68 tests passing` (3 environment-specific fallback tests skipped when full model dependencies are active).

---

## Current Limitations

1. **Monocular Scale Ambiguity**: A single 2D image does not mathematically constrain absolute scale. Without an external reference, reconstructed depths and elevations are relative unitless quantities.
2. **Relative Elevation Surface**: The generated 3D surface and height measurements represent relative vertical offsets, not absolute orthometric elevations above sea level.
3. **Remote-Sensing Domain Shift**: Depth Anything V2 is pre-trained primarily on terrestrial/ground-level datasets. When applied to nadir or high-altitude aerial imagery, vertical relief and structural edges may experience smoothing or distortions.
4. **Occluded Geometry**: Surfaces not directly visible to the single optical camera (e.g., steep building back-slopes, deep terrain shadows, areas under tree canopies) cannot be reconstructed from a single view.
5. **Compute Requirements**: Monocular transformer inference on CPU requires several seconds per image; optimal performance requires a CUDA-compatible GPU.
6. **No Automated Metric Calibration**: Automated scale derivation using reference digital elevation models (such as SRTM or Copernicus DEM) or ground control points is currently a planned roadmap extension.

---

## Roadmap

- [ ] **Automated Metric Calibration**: Integration of geospatial reference DEMs (SRTM 30m, Copernicus 30m) and metadata-derived GSD (Ground Sample Distance) to anchor relative depth to metric elevation ($Z_{\text{metric}} = a \cdot Z_{\text{rel}} + b$).
- [ ] **Remote-Sensing Fine-Tuning**: Domain adaptation on aerial and satellite datasets (e.g., SpaceNet, aerial drone photogrammetry benchmarks).
- [ ] **Quantitative Validation**: Benchmarking height and DSM accuracy against LiDAR ground truth surfaces.
- [ ] **GPU Batch Inference & WebGL Optimization**: Level-of-detail (LoD) mesh chunking for large-scale geographic rasters.
- [ ] **Geospatial Formats**: Point cloud export (LAS/LAZ) and 3D geospatial tiled formats (3D Tiles, glTF/GLB).

---

## Team & Hackathon Context

- **Competition**: Smart India Hackathon 2026
- **Problem Statement ID**: SIH26175
- **Organization**: ISRO
- **Project**: DepthWizard
- **Team**: DepthWizard Development Team
