# DepthWizard

SIH26175 -- Single-View Height Estimation and 3D Fly-Through (ISRO).
Converts a single image into a Digital Surface Model (DSM) and an
interactive 3D fly-through: non-georeferenced RGB -> relative DSM;
georeferenced GeoTIFF -> absolute DSM anchored via SRTM or ground control
points. Full problem framing, scientific-validity constraints, and the
approved architecture are in the Phase 0 audit (ask the team for the
current copy); this README tracks what is actually built.

**Status: Milestone M0 + M1 only.** M2 onward (metric scale calibration,
real height/DSM output, calibrated uncertainty, real 3D reconstruction) is
not started. Nothing in this repository currently produces a height or DSM
value -- on purpose, per the scientific-safety rules below.

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
| `depthwizard.reconstruction.placeholder_preview` | **PLACEHOLDER** | Explicitly uncalibrated preview unprojection for the M1 viewer scaffold only. Not the real `/reconstruction` module (point filtering, meshing, observed/inferred/unknown tagging) -- that's a later milestone. |
| `depthwizard.backend` (FastAPI skeleton) | **PLACEHOLDER, UNVERIFIED** | `fastapi` not installed in the dev session; never imported or run. Not on the M1 critical path (see "Frontend" below). |
| `depthwizard.geometry`, `.scale`, `.uncertainty`, `.height`, `.evaluation` | **NOT STARTED** | Empty packages with docstrings only. Deliberately no placeholder functions, so nothing can be mistaken for a real implementation. |
| Frontend (`frontend/index.html`, zero-build vanilla Three.js viewer) | **IMPLEMENTED, PARTIALLY VERIFIED** | Loads and renders `sample_scene.json` (real output of `scripts/export_sample_scene.py`). JS syntax verified with `node --check`; never opened in an actual browser in this dev session (no browser, no network to the CDN there) -- see `frontend/README.md`. |
| Dev harness (`tools/dev_streamlit_harness.py`) | **IMPLEMENTED, UNVERIFIED** | `streamlit` not installed in the dev session; syntax-checked only, never run. |
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
# Dev harness (image upload, depth inference, GeoTIFF metadata inspection):
streamlit run tools/dev_streamlit_harness.py

# Regenerate the sample scene for the 3D viewer:
PYTHONPATH=src python3 scripts/export_sample_scene.py

# 3D viewer (no npm install needed -- see frontend/README.md):
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

Last real run in the dev session: **26 tests, 0 failures** (see the
Milestone M1 delivery notes / commit message for the full log).

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
   module yet at all).

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
src/depthwizard/   # real package (see "Architecture" above for why it's nested)
  io/               # IMPLEMENTED
  depth/            # IMPLEMENTED
  reconstruction/   # PLACEHOLDER (preview-only)
  backend/          # PLACEHOLDER, unverified
  geometry/ scale/ uncertainty/ height/ evaluation/   # NOT STARTED
frontend/           # IMPLEMENTED, partially verified (vanilla Three.js)
tests/              # 26 tests, unittest-style (pytest-compatible)
tools/               # dev_streamlit_harness.py (unverified, streamlit missing)
scripts/            # export_sample_scene.py (verified); make_test_geotiff.py (unverified)
configs/            # depth.yaml, reconstruction.yaml, logging.yaml
data/fixtures/      # small generated test fixtures only, not real datasets
prototype/          # superseded, kept for reference
```
