# DepthWizard

SIH26175 -- Single-View Height Estimation and 3D Fly-Through (ISRO).
Converts a single image into a Digital Surface Model (DSM) and an
interactive 3D fly-through: non-georeferenced RGB -> relative DSM;
georeferenced GeoTIFF -> absolute DSM anchored via SRTM or ground control
points. Full problem framing, scientific-validity constraints, and the
approved architecture are in the Phase 0 audit (ask the team for the
current copy); this README tracks what is actually built.

**Status: Ship-mode MVP (on top of M0 + M1).** A demo app
(`app/streamlit_app.py`) now runs the full pipeline end to end: upload ->
real Depth Anything V2 inference (or explicit DEMO_FALLBACK) -> a
**relative elevation / "pseudo-DSM"** surface -> an uncalibrated 3D
point-cloud preview with orbit/zoom/pan/auto-orbit -> point-to-point
**relative** height measurement -> JSON/GeoTIFF export. Full automatic
metric calibration (SRTM/GCP-anchored absolute DSM, RANSAC, propagated
uncertainty) is still **not started** -- see the scientific-safety rules
below for exactly what is and is not a real measurement right now.

## Architecture

```
INPUT -> io -> depth -> geometry -> scale -> uncertainty -> height/DSM -> reconstruction -> frontend (Three.js)
```

Source lives under `src/depthwizard/` (a `src`-layout package). Note: the
approved architecture's folder names (`io/`, `depth/`, ...) are used as
Python subpackage names here, nested under `src/depthwizard/` rather than
at the repo root -- a top-level package literally named `io` would shadow
Python's own standard-library `io` module for anything importing this
project, which is a real and nasty bug, not a style preference. Folder
names and responsibilities otherwise match the approved architecture
exactly.

## Implemented / Placeholder / Not Started

| Component | Status | Notes |
|---|---|---|
| `depthwizard.io` (PNG/JPG/TIFF/GeoTIFF loading, CRS/transform extraction, pixel<->geo coordinates) | **IMPLEMENTED** | Real GeoTIFF path requires `rasterio` (optional dep, not installed in the dev session -- logic tested via mocking, see `tests/test_io.py`). Plain PNG/JPG path fully tested for real. |
| `depthwizard.depth` (model-agnostic `DepthPredictor`, `DepthResult`, DEMO_FALLBACK) | **IMPLEMENTED** | Depth Anything V2 Small backend requires `torch`+`transformers` (optional deps, not installed in the dev session). The "real backend fails visibly, fallback is explicit and labeled" contract is fully tested for real, because that failure genuinely occurs in this environment. A real (status=SUCCESS) prediction has never been executed anywhere in this project's history yet -- flagged, not claimed. |
| `depthwizard.reconstruction.placeholder_preview` | **IMPLEMENTED** | Explicitly uncalibrated preview unprojection, unchanged since M1. Now wired end-to-end via `scene_export.build_scene`. |
| `depthwizard.reconstruction.dsm` | **IMPLEMENTED** | Reinterprets relative depth as a "relative elevation" / pseudo-DSM surface via one named, documented assumption (closer-to-camera = higher). NOT a calibrated DSM -- see the module docstring. Unit-tested (`tests/test_dsm.py`). |
| `depthwizard.reconstruction.scene_export` | **IMPLEMENTED** | Builds an exportable point-cloud scene (JSON) and, when the input was georeferenced and `rasterio` is available, a relative-elevation GeoTIFF. Unit-tested (`tests/test_scene_export.py`). |
| `depthwizard.height.measurement` | **IMPLEMENTED (relative-only)** | Point-to-point relative elevation difference. Reports a meters value ONLY if the caller supplies an explicit `meters_per_unit` + `calibration_source` -- there is no automatic calibration (no SRTM/GCP/RANSAC). Unit-tested (`tests/test_height.py`). |
| `app/streamlit_app.py` (SIH-facing demo) | **IMPLEMENTED, PARTIALLY VERIFIED** | Full upload -> depth -> elevation -> 3D -> height -> export pipeline. Syntax/logic verified (`py_compile`, and every non-UI function it calls is unit-tested); the Streamlit process itself has not been run inside this Cowork session because `streamlit` is not installed there -- run it on the Windows machine, which already has `streamlit` installed. |
| `depthwizard.backend` (FastAPI skeleton) | **PLACEHOLDER, UNVERIFIED** | `fastapi` not installed in the dev session; never imported or run. Not on the demo's critical path. |
| `depthwizard.geometry`, `.scale`, `.uncertainty`, `.evaluation` | **NOT STARTED** | Empty packages with docstrings only. Deliberately no placeholder functions, so nothing can be mistaken for a real implementation. Automatic metric calibration (SRTM/GCP anchoring) lives here, later milestone. |
| Frontend (`frontend/index.html`, zero-build vanilla Three.js viewer) | **IMPLEMENTED, PARTIALLY VERIFIED** | Standalone static-scene viewer from M1, still present. The demo app's own embedded Three.js viewer (in `app/streamlit_app.py`) is the primary SIH-facing 3D view now; `frontend/` remains a secondary/dev artifact. |
| Dev harness (`tools/dev_streamlit_harness.py`) | **IMPLEMENTED, UNVERIFIED** | Superseded by `app/streamlit_app.py` as the demo entry point; kept as a lighter depth/metadata-only inspection tool. |
| Domain validation (`evaluation/domain_validation/`) | **NOT STARTED** | A dataset acquisition plan (ISPRS Potsdam/Vaihingen, WHU-MVS) was produced, but no images have been downloaded and no qualitative validation pass has been run yet -- this folder does not exist. Ship-mode MVP was prioritized first per explicit instruction. |
| Old prototype (`prototype/`) | **SUPERSEDED** | Kept for historical reference, annotated as superseded. Do not run or extend. |

## A note on this environment

Everything above that says "not installed in the dev session" reflects a
real, confirmed constraint of the environment this milestone was built in:
outbound network access to PyPI (`pip install`) and the npm registry
(`npm install`) is blocked there (`403 host_not_allowed`, confirmed with
direct `pip install numpy` and `npm install three` attempts, not just the
heavy optional packages). **This is a constraint of that specific
development session, not of the code itself or of your own machine** --
`pip install -e ".[geo,depth-model,backend,dev]"` should work normally
with regular internet access. Please run it and report back if anything
in the "UNVERIFIED" rows above turns out not to work; that is the
single highest-value thing to check before starting Milestone M2.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # or your preferred venv tool
pip install -e ".[geo,depth-model,dev]"
```
Add `backend` to the extras list once you start on Milestone M2's live API.

## Run

```bash
# SIH-facing demo app (the actual MVP -- upload, depth, DSM, 3D, height, export):
streamlit run app/streamlit_app.py

# Dev harness (lighter tool: image upload, depth inference, GeoTIFF metadata inspection):
streamlit run tools/dev_streamlit_harness.py

# Regenerate the sample scene for the standalone frontend/ viewer:
PYTHONPATH=src python3 scripts/export_sample_scene.py

# Standalone 3D viewer (no npm install needed -- see frontend/README.md):
cd frontend && python3 -m http.server 8080
# then open http://localhost:8080/
```

## Test

```bash
pip install -e ".[dev]"     # for pytest
pytest                      # or: PYTHONPATH=src python -m unittest discover -s tests
```
All test files are written as `unittest.TestCase` subclasses specifically
so they run without `pytest` installed (`python -m unittest discover`) --
`pytest` was not installed in the dev session either, so that is exactly
how these 26 tests were actually executed and verified. `pytest` will pick
up the same tests natively once installed.

Last real run in the dev session: **37 tests, 0 failures** (26 from M1 +
11 new: `test_dsm.py`, `test_height.py`, `test_scene_export.py`).

## Scientific-safety rules in force (see the Phase 0 audit for the full rationale)

1. No height or DSM value is produced anywhere in this codebase yet -- the
   old prototype's invalid "depth-span" measurement was removed, not fixed,
   because its math (not just its crash) was unsound.
2. `depthwizard.depth`'s output is always RELATIVE depth. Nothing treats it
   as metric.
3. No camera intrinsics are silently assumed. The one placeholder
   focal-length estimate that exists (`reconstruction.placeholder_preview`)
   is explicit, configurable, and tags every output `calibrated: False`.
4. Real model/backend failures raise typed, chained exceptions
   (`ModelLoadError`, `ModelInferenceError`, `UnsupportedImageError`, ...)
   and are logged. A fallback only ever runs when a caller explicitly
   passes `allow_fallback=True`, and its result is always tagged
   `DepthStatus.DEMO_FALLBACK` -- enforced in `DepthResult.__post_init__`,
   not just by convention.
5. No confidence percentage is reported anywhere (there is no uncertainty
   module yet at all). The demo app's "structural detail" indicator is a
   real, directly-computed statistic (mean elevation-gradient magnitude),
   explicitly labeled as a heuristic about output structure, never as
   accuracy.
6. `reconstruction.dsm` and `height.measurement` never invent a metric
   value. A height is shown in meters only when the user explicitly
   supplies a `meters_per_unit` scale AND names its source
   (`calibration_source`) -- otherwise every number stays in relative,
   unitless form and is labeled as such.

## Model licensing notes

- **Depth Anything V2 Small** (`depth-anything/Depth-Anything-V2-Small-hf`,
  the configured primary backend): Apache-2.0. Base/Large/Giant checkpoints
  are CC-BY-NC-4.0 (non-commercial) -- not used here.
- No other model backends are integrated yet. See the Phase 0 audit,
  Section 9, for the full comparison (Depth Pro, VGGT) and their licenses
  before adding one.

## Expected hardware

Everything implemented so far (`io`, the fallback depth path, coordinate
math, the placeholder unprojection) runs on CPU with no GPU required. Real
Depth Anything V2 Small inference is comfortable on a modest consumer GPU
(8 GB VRAM class) and usable, just slower, on CPU -- untested in this dev
session for the reasons above. `rasterio` needs a working GDAL install on
whichever machine runs the real GeoTIFF path.

## Repository layout

```
app/                # streamlit_app.py -- the SIH-facing demo MVP
src/depthwizard/   # real package (see "Architecture" above for why it's nested)
  io/               # IMPLEMENTED
  depth/            # IMPLEMENTED
  reconstruction/   # IMPLEMENTED (placeholder_preview, dsm, scene_export -- all uncalibrated by design)
  height/           # IMPLEMENTED (relative-only; calibrated only if caller supplies a scale)
  backend/          # PLACEHOLDER, unverified
  geometry/ scale/ uncertainty/ evaluation/   # NOT STARTED
frontend/           # IMPLEMENTED, partially verified (vanilla Three.js, standalone/secondary viewer)
tests/              # 37 tests, unittest-style (pytest-compatible)
tools/               # dev_streamlit_harness.py (lighter inspection tool, superseded as the demo)
scripts/            # export_sample_scene.py (verified); make_test_geotiff.py (unverified)
configs/            # depth.yaml, reconstruction.yaml, logging.yaml
data/fixtures/      # small generated test fixtures only, not real datasets
data/outputs/       # demo app's exported GeoTIFF artifacts (gitignored)
evaluation/         # domain_validation/ -- planned, not yet populated (see table above)
prototype/          # superseded, kept for reference
```
