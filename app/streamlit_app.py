"""DepthWizard -- SIH26175 competition build.

    streamlit run app/streamlit_app.py

This is the SIH-facing product surface (not the dev harness in
tools/dev_streamlit_harness.py, which stays a lighter internal tool).
It wires together, end to end, everything implemented so far, and tells
one linear story on screen:

    UPLOAD -> ANALYZE -> UNDERSTAND -> MEASURE -> EXPLORE -> EXPORT

        upload -> depthwizard.io.load_image
               -> depthwizard.depth.DepthPredictor           (real DA-V2 or
                  explicit DEMO_FALLBACK, never silently swapped)
               -> depthwizard.reconstruction.dsm              (relative
                  elevation / "pseudo-DSM", explicitly labeled as such)
               -> depthwizard.height.measurement               (point-to-point
                  relative height, optionally scaled if the user supplies an
                  external reference -- never auto-calibrated)
               -> depthwizard.reconstruction.scene_export       (point cloud +
                  JSON / GeoTIFF export)
               -> an embedded Three.js viewer (orbit / zoom / pan / auto-fly)

Scientific-honesty contract (carried over from M0/M1, unchanged):
    - Every depth result is labeled SUCCESS or DEMO_FALLBACK on screen;
      DEMO_FALLBACK is never allowed to look like a real prediction, and
      this banner is never hidden or demoted -- it stays prominent even
      after the rest of the interface was cleaned up for competition use.
    - The elevation surface is always called "relative elevation" /
      "pseudo-DSM", never "DSM" or "elevation" unqualified, because it is
      an explicit, named reinterpretation of relative depth -- see
      depthwizard.reconstruction.dsm's module docstring. The full raster
      is NEVER labeled "metric": this pipeline only ever produces a metric
      number at the single point-to-point measurement, and only when the
      user supplies an external meters-per-unit reference themselves.
    - No height number is ever shown in meters unless the user explicitly
      supplied that reference; the UI always names which case it's in
      ("Relative" vs "Metric") using the actual field, never inferred.
    - No accuracy, confidence, or uncertainty percentage is fabricated
      anywhere. The "structural detail" heuristic is a real, directly
      computed statistic of the output (mean elevation-gradient
      magnitude) -- it is kept out of the primary metrics so it cannot be
      mistaken for an accuracy score, and is labeled as a heuristic
      wherever it does appear (Technical details).
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import json

import numpy as np

from depthwizard.io.loader import load_image, rasterio_available
from depthwizard.io.exceptions import DepthWizardIOError
from depthwizard.depth.predictor import DepthPredictor
from depthwizard.depth.exceptions import DepthWizardModelError
from depthwizard.depth.base import DepthStatus
from depthwizard.reconstruction.dsm import depth_to_relative_elevation, elevation_stats
from depthwizard.reconstruction.elevation_grid import build_elevation_grid
from depthwizard.reconstruction.mesh import build_terrain_mesh, mesh_to_json_safe
from depthwizard.reconstruction.scene_export import (
    build_mesh_scene,
    export_scene_json,
    export_elevation_geotiff,
)
from depthwizard.height.measurement import relative_height_between_points
from depthwizard.logging_setup import configure_logging, get_logger

try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit(
        "streamlit is not installed. Install the `dev` extra: pip install -e '.[dev]'"
    ) from exc

import cv2

configure_logging()
log = get_logger("depthwizard.app.streamlit_app")

st.set_page_config(page_title="DepthWizard", layout="wide")

OUTPUT_DIR = REPO_ROOT / "data" / "outputs"

# ---------------------------------------------------------------------------
# Visual system: fonts + a small set of component tweaks. The dark palette
# itself lives in .streamlit/config.toml (Streamlit's native theming), which
# is more robust across versions than CSS-overriding Streamlit's internal
# class names. This block only adds what config.toml cannot: custom
# typefaces and a handful of layout refinements.
# ---------------------------------------------------------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
h1, h2, h3 { font-family: "Space Grotesk", sans-serif !important; letter-spacing: -0.01em; }
h1 { font-weight: 700 !important; }
h2, h3 { font-weight: 600 !important; }

[data-testid="stMetricValue"] { font-family: "IBM Plex Mono", monospace; }

.dw-tagline { color: #8b93a3; font-size: 0.95rem; margin-top: -0.6rem; margin-bottom: 1.4rem; }

.dw-badge { display: inline-block; padding: 3px 9px; border-radius: 4px; font-size: 0.72rem;
  font-family: "IBM Plex Mono", monospace; letter-spacing: 0.03em; text-transform: uppercase;
  border: 1px solid; margin-right: 6px; }
.dw-badge-relative { color: #f5a623; border-color: #4a3a1a; background: rgba(245,166,35,0.08); }
.dw-badge-metric { color: #3ddad0; border-color: #164e49; background: rgba(61,218,208,0.08); }
.dw-badge-neutral { color: #8b93a3; border-color: #232b38; background: rgba(139,147,163,0.06); }

.dw-section-rule { border: none; border-top: 1px solid #1c2430; margin: 1.6rem 0 1.2rem 0; }

.dw-flow { display: flex; align-items: center; gap: 10px; margin: 0.8rem 0 1.4rem 0; flex-wrap: wrap; }
.dw-flow-step { font-family: "Space Grotesk", sans-serif; font-weight: 600; font-size: 0.98rem;
  color: #e6e9ef; padding: 5px 12px; border: 1px solid #232b38; border-radius: 4px; background: #10151d; }
.dw-flow-arrow { color: #3ddad0; font-size: 1.1rem; }

[data-testid="stFileUploaderDropzone"] { border-radius: 6px; }
.stButton > button { border-radius: 6px; font-weight: 600; }
[data-testid="stExpander"] { border-radius: 6px; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


def render_error(problem: str, why: str, action: str) -> None:
    """Render a plain-language error card. Never shows a raw traceback --
    the actual exception detail is folded into `why`, which the caller
    controls, and the full traceback always still goes to the log."""
    st.error(f"**{problem}**\n\n{why}\n\n**What you can do:** {action}")


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="dw-badge dw-badge-{kind}">{text}</span>'


@st.cache_data(show_spinner=False)
def _peek_upload(file_bytes: bytes, suffix: str):
    """Cheap, cached read of just the header/pixels so Upload can show
    filename/resolution/georeferenced status before Analyze is even
    clicked, without re-running this on every widget interaction."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)
    try:
        image, meta = load_image(tmp_path)
        return {"ok": True, "shape": image.shape, "georeferenced": meta.is_georeferenced}
    except DepthWizardIOError as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("DepthWizard")
st.markdown(
    '<div class="dw-tagline">Single-view DSM reconstruction &amp; 3D exploration &nbsp;'
    '&mdash;&nbsp; from one optical image to an explorable elevation model. '
    '<span style="opacity:0.6;">SIH26175</span></div>',
    unsafe_allow_html=True,
)

if "scene" not in st.session_state:
    st.session_state.scene = None
    st.session_state.elevation = None
    st.session_state.elevation_grid = None
    st.session_state.mesh = None
    st.session_state.reconstruction_time_ms = None
    st.session_state.image = None
    st.session_state.depth_result = None
    st.session_state.meta = None
    st.session_state.height_result = None

# ---------------------------------------------------------------------------
# SIDEBAR -- upload, mode, processing controls, the single primary action
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Upload")
    uploaded = st.file_uploader("Aerial / optical image", type=["png", "jpg", "jpeg", "tif", "tiff"])

    upload_info = None
    if uploaded is not None:
        upload_info = _peek_upload(uploaded.getvalue(), Path(uploaded.name).suffix)
        if upload_info["ok"]:
            h, w = upload_info["shape"][:2]
            geo = "Georeferenced" if upload_info["georeferenced"] else "Not georeferenced"
            st.caption(
                f"**{uploaded.name}**  \n{w} x {h} px &middot; {Path(uploaded.name).suffix.lstrip('.').upper()} "
                f"&middot; {geo}"
            )
        else:
            st.caption(f"**{uploaded.name}**")
            st.warning(f"Could not read this file: {upload_info['error']}")

    st.subheader("Mode")
    allow_fallback = st.checkbox(
        "Allow DEMO_FALLBACK if the real model can't load",
        value=False,
        help="Off by default for the competition demo, which must start in "
             "real-model mode. Turn this on only for development, when the "
             "real Depth Anything V2 backend may be unavailable -- a real "
             "failure is otherwise shown as an error, never silently "
             "substituted with a non-AI heuristic.",
    )

    st.subheader("Processing controls")
    invert_elevation = st.checkbox(
        "Closer-to-camera = higher elevation", value=True,
        help="The standard assumption for a roughly-overhead shot. Turn off "
             "if your image's geometry makes this wrong -- see "
             "depthwizard.reconstruction.dsm docstring.",
    )
    assumed_hfov = st.slider(
        "Assumed horizontal FOV (deg, uncalibrated)", 20.0, 120.0, 60.0, step=5.0,
        help="Unprojection is always uncalibrated (calibrated=False) regardless "
             "of this value -- it only affects the shape of the preview point cloud.",
    )
    max_points = st.slider(
        "Max mesh vertices", 2000, 80000, 30000, step=2000,
        help="Caps the decimated surface-mesh resolution for real-time browser "
             "rendering. The elevation grid itself always stays at full source "
             "resolution -- this only controls how coarsely it is re-sampled "
             "onto the 3D mesh.",
    )

    can_run = uploaded is not None and (upload_info is None or upload_info.get("ok", False))
    run_clicked = st.button("Analyze Image", type="primary", disabled=not can_run, use_container_width=True)

# ---------------------------------------------------------------------------
# ANALYZE -- real staged progress, every failure surfaced explicitly
# ---------------------------------------------------------------------------
if run_clicked and uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)

    with st.status("Analyzing image", expanded=True) as status:
        try:
            status.write("Validating image...")
            try:
                image, meta = load_image(tmp_path)
            except DepthWizardIOError as exc:
                status.update(label="Analysis failed", state="error")
                render_error(
                    "Could not read this file.",
                    str(exc),
                    "Check that it is a valid PNG, JPG, TIFF, or GeoTIFF and try uploading again.",
                )
                st.stop()
            status.write("Image validated.")

            status.write("Running depth inference (Depth Anything V2)...")
            predictor = DepthPredictor()
            try:
                result = predictor.predict(image, allow_fallback=allow_fallback)
            except DepthWizardModelError as exc:
                status.update(label="Analysis failed", state="error")
                render_error(
                    "Depth inference failed.",
                    str(exc),
                    "Enable “Allow DEMO_FALLBACK” in the sidebar for a clearly-labeled "
                    "placeholder, or fix the model installation and try again.",
                )
                st.stop()
            status.write(f"Depth inference complete ({result.inference_time_ms:.0f} ms).")

            status.write("Computing relative elevation...")
            elevation = depth_to_relative_elevation(result.depth, invert=invert_elevation)
            elevation_grid = build_elevation_grid(
                elevation, depth_source=result.source, depth_status=result.status.value,
            )
            status.write("Elevation grid computed.")

            status.write("Reconstructing surface mesh...")
            recon_t0 = time.perf_counter()
            try:
                mesh = build_terrain_mesh(
                    elevation_grid, image, assumed_hfov_deg=assumed_hfov, max_vertices=max_points,
                )
            except ValueError as exc:
                status.update(label="Analysis failed", state="error")
                render_error(
                    "Could not reconstruct a surface from this image.",
                    str(exc),
                    "Try a larger image, or a different image with more visible structure.",
                )
                st.stop()
            reconstruction_time_ms = (time.perf_counter() - recon_t0) * 1000.0
            scene = build_mesh_scene(elevation_grid, mesh)
            status.write(
                f"Surface ready: {mesh['vertex_count']} vertices, "
                f"{mesh['triangle_count']} triangles ({reconstruction_time_ms:.0f} ms)."
            )

            status.update(label="Analysis complete", state="complete", expanded=False)
        except Exception as exc:  # last-resort guard -- never surface a raw traceback
            log.exception("Unexpected failure during analysis")
            status.update(label="Analysis failed", state="error")
            render_error(
                "Something went wrong during analysis.",
                str(exc),
                "Try a different image, or re-run with different processing controls. "
                "Full detail is in the application log.",
            )
            st.stop()

    st.session_state.image = image
    st.session_state.meta = meta
    st.session_state.depth_result = result
    st.session_state.elevation = elevation
    st.session_state.elevation_grid = elevation_grid
    st.session_state.mesh = mesh
    st.session_state.reconstruction_time_ms = reconstruction_time_ms
    st.session_state.scene = scene
    st.session_state.height_result = None

# ---------------------------------------------------------------------------
# Nothing analyzed yet -- a single clear call to action, then stop.
# ---------------------------------------------------------------------------
if st.session_state.scene is None:
    st.info("Upload an aerial or optical image in the sidebar, then click **Analyze Image** to begin.")
    st.stop()

image = st.session_state.image
meta = st.session_state.meta
result = st.session_state.depth_result
elevation = st.session_state.elevation
elevation_grid = st.session_state.elevation_grid
mesh = st.session_state.mesh
reconstruction_time_ms = st.session_state.reconstruction_time_ms
scene = st.session_state.scene
stats = elevation_stats(elevation)
surface_label = "Metric DSM" if elevation_grid["calibrated"] else "Relative Elevation Surface"

# ---------------------------------------------------------------------------
# UNDERSTAND -- original / depth / elevation, with an honest calibration badge
# ---------------------------------------------------------------------------
st.header("Understand")

if result.status == DepthStatus.DEMO_FALLBACK:
    st.warning(
        f"DEMO_FALLBACK IN USE — this is NOT a real model prediction. "
        f"Reason: {result.metadata.get('reason_for_fallback')}"
    )
else:
    st.success(f"Real prediction from {result.source} ({result.inference_time_ms:.0f} ms)")

st.markdown(badge("Relative / Uncalibrated", "relative"), unsafe_allow_html=True)
st.caption("Depth and elevation below are relative to this image only — not metric, not comparable across images.")

st.markdown(
    '<div class="dw-flow">'
    '<span class="dw-flow-step">Original Image</span><span class="dw-flow-arrow">&rarr;</span>'
    '<span class="dw-flow-step">Relative Depth</span><span class="dw-flow-arrow">&rarr;</span>'
    '<span class="dw-flow-step">Elevation Surface</span><span class="dw-flow-arrow">&rarr;</span>'
    '<span class="dw-flow-step">3D Reconstruction</span>'
    '</div>',
    unsafe_allow_html=True,
)

depth_u8 = (np.clip(result.depth, 0, 1) * 255).astype(np.uint8)
depth_colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)[:, :, ::-1]

elev_u8 = (np.clip(elevation, 0, 1) * 255).astype(np.uint8)
elev_colored = cv2.applyColorMap(elev_u8, cv2.COLORMAP_VIRIDIS)[:, :, ::-1]


def _legend_strip(colormap: int, lo_label: str, hi_label: str):
    ramp = np.tile(np.linspace(0, 255, 256).astype(np.uint8), (14, 1))
    strip = cv2.applyColorMap(ramp, colormap)[:, :, ::-1]
    return strip, lo_label, hi_label


c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.image(image, caption="Original", use_container_width=True)
with c2:
    with st.container(border=True):
        st.image(depth_colored, caption=f"Relative depth ({result.status.value})", use_container_width=True)
        strip, lo, hi = _legend_strip(cv2.COLORMAP_TURBO, "near", "far")
        st.image(strip, use_container_width=True)
        st.caption(f"{lo} → {hi}")
with c3:
    with st.container(border=True):
        st.image(elev_colored, caption=f"{surface_label} (pseudo-DSM)", use_container_width=True)
        strip, lo, hi = _legend_strip(cv2.COLORMAP_VIRIDIS, f"{stats['min']:.2f}", f"{stats['max']:.2f}")
        st.image(strip, use_container_width=True)
        st.caption(f"{lo} → {hi} (relative units)")

st.markdown('<hr class="dw-section-rule" />', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# METRICS -- primary / secondary hierarchy, no fabricated scores
# ---------------------------------------------------------------------------
st.subheader("Status")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Processing status", result.status.value)
m2.metric("Depth model", result.source.split("/")[-1] if "/" in result.source else result.source)
m3.metric("Input resolution", f"{result.shape[1]} x {result.shape[0]}")
m4.metric("Processing time", f"{result.inference_time_ms:.0f} ms")

sec1, sec2, sec3, sec4 = st.columns(4)
sec1.caption("Georeferenced")
sec1.write("Yes" if meta.is_georeferenced else "No")
sec2.caption("Scene status")
sec2.write("Ready" if scene else "Unavailable")
sec3.caption("Mesh vertices")
sec3.write(f"{mesh['vertex_count']}")
sec4.caption("Mesh triangles")
sec4.write(f"{mesh['triangle_count']}")
# Measurement calibration (Metric/Relative) is NOT shown here: this section
# runs before Measure computes it in each script pass, so reading
# st.session_state.height_result here would always show the *previous*
# run's value, one render behind. The Measure section below shows its own
# calibration badge live against the value it just computed instead.

with st.expander("Technical details"):
    grad_y, grad_x = np.gradient(elevation)
    structural_detail = float(np.mean(np.abs(grad_x)) + np.mean(np.abs(grad_y)))
    st.write(f"**Structural detail (heuristic):** {structural_detail:.4f}")
    st.caption(
        "A real, directly-computed statistic (mean local elevation-gradient magnitude) "
        "— NOT an accuracy or confidence score. Higher generally means the model found "
        "more local structure in this image; it says nothing about whether that structure "
        "is geometrically correct. See evaluation/domain_validation/REPORT.md for "
        "qualitative accuracy notes."
    )
    st.write(
        f"**Relative elevation stats:** min={stats['min']:.3f}  max={stats['max']:.3f}  "
        f"mean={stats['mean']:.3f}  std={stats['std']:.3f}  (unit: {stats['unit']})"
    )
    st.write(f"**Depth source:** {result.source}")
    st.write(f"**Depth status:** {result.status.value}")
    st.write("**3D calibration:** uncalibrated preview geometry (calibrated=False)")
    st.write(
        f"**Surface reconstruction:** source {mesh['source_rows']}x{mesh['source_cols']} px "
        f"→ decimation stride {mesh['stride']} → {mesh['grid_rows']}x{mesh['grid_cols']} mesh grid "
        f"({mesh['vertex_count']} vertices, {mesh['triangle_count']} triangles, "
        f"{mesh['degenerate_triangles_dropped']} degenerate triangles rejected) "
        f"in {reconstruction_time_ms:.0f} ms."
    )
    st.caption(
        "Reconstruction time is measured server-side (elevation grid → mesh). Renderer "
        "initialization time is measured client-side and shown in the 3D viewer's "
        "Diagnostics panel, not fabricated here."
    )

st.markdown('<hr class="dw-section-rule" />', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MEASURE -- point-to-point, mode made explicit and impossible to misread
# ---------------------------------------------------------------------------
st.header("Measure")
st.caption("Relative elevation / pseudo-DSM measurement between two points -- the main demo output.")
h, w = elevation.shape
pc1, pc2 = st.columns(2)
with pc1:
    st.caption("Point A")
    r1 = st.number_input("Row", 0, h - 1, 0, key="pt_a_row")
    c_1 = st.number_input("Col", 0, w - 1, 0, key="pt_a_col")
with pc2:
    st.caption("Point B")
    r2 = st.number_input("Row", 0, h - 1, h - 1, key="pt_b_row")
    c_2 = st.number_input("Col", 0, w - 1, w - 1, key="pt_b_col")

mpu = None
with st.expander("Advanced / Experimental -- manual metric calibration"):
    st.caption(
        "Not part of the core demo. This pipeline never derives metric scale on its "
        "own -- if you have an independent, trusted reference (e.g. a known building "
        "height in this scene), you may supply it here to convert the single "
        "measurement below into meters. Leave blank to keep the result honestly "
        "uncalibrated, which is the default and the recommended demo state."
    )
    meters_per_unit_raw = st.text_input(
        "Meters per relative unit", value="", placeholder="e.g. 2.5",
    )
    try:
        mpu = float(meters_per_unit_raw) if meters_per_unit_raw.strip() else None
    except ValueError:
        mpu = None
        st.error("External scale must be a number — ignoring it.")

try:
    height_result = relative_height_between_points(
        elevation, (int(r1), int(c_1)), (int(r2), int(c_2)),
        meters_per_unit=mpu,
        calibration_source="user-supplied manual reference" if mpu is not None else None,
    )
    st.session_state.height_result = height_result
except ValueError as exc:
    render_error("Invalid measurement points.", str(exc), "Choose two points within the image bounds.")
    height_result = None
    st.session_state.height_result = None

if height_result is not None:
    annotated = image.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2RGB)
    cv2.drawMarker(annotated, (int(c_1), int(r1)), (255, 0, 0), cv2.MARKER_CROSS, 20, 3)
    cv2.drawMarker(annotated, (int(c_2), int(r2)), (0, 128, 255), cv2.MARKER_CROSS, 20, 3)

    img_col, result_col = st.columns([2, 1])
    with img_col:
        st.image(annotated, caption="Point A (red) / Point B (orange)", use_container_width=True)
    with result_col:
        if height_result["calibrated"]:
            st.markdown(badge("Metric", "metric"), unsafe_allow_html=True)
            st.metric("Height", f"{height_result['meters_difference']:.2f} m")
            st.caption(f"Source: {height_result['calibration_source']}")
        else:
            st.markdown(badge("Relative", "relative"), unsafe_allow_html=True)
            st.metric("Δ relative elevation", f"{height_result['relative_difference']:.4f}")
            st.caption(
                "No metric scale supplied — this is a relative elevation unit, not meters. "
                "Supply an external scale above for a metric result."
            )

st.markdown('<hr class="dw-section-rule" />', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# EXPLORE -- the reconstructed surface mesh, the visual hero of the app
# ---------------------------------------------------------------------------
st.header("Explore in 3D")
st.caption(
    f"Reconstructed surface: {mesh['vertex_count']} vertices, {mesh['triangle_count']} "
    f"triangles. Drag to orbit, scroll to zoom, right-drag to pan."
)


def _round_list(arr, decimals=4):
    return np.round(np.asarray(arr, dtype=np.float64), decimals).tolist()


mesh_payload = {
    "vertices": _round_list(mesh["vertices"]),
    "normals": _round_list(mesh["normals"]),
    "colors_rgb": mesh["colors_rgb"].astype(int).tolist(),
    "colors_elevation": mesh["colors_elevation"].astype(int).tolist(),
    "triangles": mesh["triangles"].astype(int).tolist(),
    "bbox_min": _round_list(mesh["bbox_min"]),
    "bbox_max": _round_list(mesh["bbox_max"]),
}
mesh_json = json.dumps(mesh_payload)

# The viewer is built from a *raw* string template with a plain
# find-and-replace for the one dynamic value (the point data), never an
# f-string. An f-string here previously caused a real, hard-to-find bug:
# Python's own string parsing silently collapsed a JS "\n" escape into a
# literal newline before the browser ever saw it, breaking a quoted JS
# string and aborting the whole module with a SyntaxError. A raw string
# with no interpolation makes that entire bug class impossible.
_VIEWER_TEMPLATE = r"""
<div id="dw-viewer-wrap" style="position:relative;width:100%;">
  <div id="dw-viewer" style="width:100%;height:640px;background:#0a0e14;position:relative;border-radius:6px;overflow:hidden;"></div>
  <div id="dw-toolbar" style="position:absolute;top:10px;right:10px;display:flex;gap:10px;z-index:3;">
    <div class="dw-tgroup">
      <button id="dw-btn-rgb" class="dw-tbtn active" title="Show original RGB appearance">RGB</button>
      <button id="dw-btn-elevation" class="dw-tbtn" title="Show elevation-based color">Elevation</button>
      <button id="dw-btn-wireframe" class="dw-tbtn" title="Toggle mesh wireframe">Wireframe</button>
    </div>
    <div class="dw-tgroup">
      <button id="dw-btn-fit" class="dw-tbtn" title="Fit camera to scene">Fit</button>
      <button id="dw-btn-reset" class="dw-tbtn" title="Reset to the starting view">Reset</button>
      <button id="dw-btn-fly" class="dw-tbtn" title="Scripted cinematic fly-through">Auto-Fly</button>
    </div>
  </div>
  <button id="dw-btn-diag" class="dw-tbtn-quiet" title="Show technical diagnostics" style="position:absolute;bottom:10px;right:10px;z-index:3;">diag</button>
  <div id="dw-loading" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#8b93a3;font-family:monospace;font-size:13px;background:#0a0e14;z-index:4;text-align:center;padding:20px;">Loading 3D scene...</div>
  <pre id="dw-diag-panel" style="display:none;position:absolute;bottom:40px;right:10px;color:#3ddad0;background:rgba(10,14,20,0.92);font-family:monospace;font-size:11px;line-height:1.4;white-space:pre-wrap;z-index:2;margin:0;padding:8px 10px;pointer-events:none;max-width:60%;border-radius:4px;border:1px solid #232b38;"></pre>
</div>
<style>
.dw-tgroup { display:flex; gap:4px; background:rgba(10,14,20,0.5); border-radius:5px; padding:2px; }
.dw-tbtn { background:#131a24; color:#e6e9ef; border:1px solid #232b38; border-radius:4px;
  padding:5px 10px; font-family:sans-serif; font-size:12px; cursor:pointer; }
.dw-tbtn:hover { border-color:#3ddad0; color:#3ddad0; }
.dw-tbtn.active { background:#3ddad0; color:#0a0e14; border-color:#3ddad0; }
/* De-emphasized on purpose -- diagnostics is a developer aid, not part of
   the primary [RGB][Elevation][Wireframe][Fit][Reset][Auto-Fly] toolbar. */
.dw-tbtn-quiet { background:rgba(19,26,36,0.6); color:#5b6472; border:1px solid #232b38;
  border-radius:4px; padding:3px 8px; font-family:monospace; font-size:10px; cursor:pointer;
  opacity:0.7; }
.dw-tbtn-quiet:hover { color:#3ddad0; border-color:#3ddad0; opacity:1; }
.dw-tbtn-quiet.active { color:#3ddad0; opacity:1; }
</style>
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>
<script type="module">
// Streamlit's components.v1.html() does not execute plain/untyped <script>
// tags -- only type="module" and type="importmap" do (found by live
// debugging, not assumed). Everything lives in this one module script.
window.dwDiag = {
  three: "PENDING", orbit: "PENDING", sceneData: "PENDING",
  vertices: 0, triangles: 0, bbox: "PENDING", canvas: "PENDING",
  webgl: "PENDING", renderer: "PENDING", renderLoop: "PENDING",
  renderInitMs: "PENDING", mode: "rgb", error: ""
};
window.dwModuleRan = true;

function dwRenderPanel() {
  var d = window.dwDiag;
  var lines = [
    "[DepthWizard 3D Diagnostics]",
    "Three.js: " + d.three,
    "OrbitControls: " + d.orbit,
    "Scene data: " + d.sceneData,
    "Vertices: " + d.vertices,
    "Triangles: " + d.triangles,
    "Bounding box: " + d.bbox,
    "Canvas: " + d.canvas,
    "WebGL: " + d.webgl,
    "Renderer: " + d.renderer,
    "Render loop: " + d.renderLoop,
    "Render init: " + d.renderInitMs,
    "Color mode: " + d.mode,
  ];
  if (d.error) lines.push("ERROR: " + d.error);
  var el = document.getElementById("dw-diag-panel");
  if (el) el.textContent = lines.join("\n");
}
dwRenderPanel();
window.addEventListener("error", function (e) {
  window.dwDiag.error = (window.dwDiag.error ? window.dwDiag.error + " | " : "") + "window error: " + e.message;
  dwRenderPanel();
});
window.addEventListener("unhandledrejection", function (e) {
  var msg = (e.reason && e.reason.message) ? e.reason.message : String(e.reason);
  window.dwDiag.error = (window.dwDiag.error ? window.dwDiag.error + " | " : "") + "unhandled rejection: " + msg;
  dwRenderPanel();
});

var loadingEl = document.getElementById("dw-loading");
function dwShowViewerError(msg) {
  loadingEl.style.display = "flex";
  loadingEl.style.color = "#ef4444";
  loadingEl.textContent = msg;
}
function dwHideLoading() {
  loadingEl.style.display = "none";
}

(async () => {
  const dwStartTime = performance.now();
  const meshData = __DW_MESH_JSON__;
  const container = document.getElementById("dw-viewer");
  try {
    let THREE;
    try {
      THREE = await import("three");
      window.dwDiag.three = "LOADED";
    } catch (err) {
      window.dwDiag.three = "FAILED: " + err.message;
      dwRenderPanel();
      dwShowViewerError("Could not load the 3D viewer library. Check your internet connection and reload the page.");
      return;
    }
    dwRenderPanel();

    let OrbitControls;
    try {
      const orbitModule = await import("three/addons/controls/OrbitControls.js");
      OrbitControls = orbitModule.OrbitControls;
      window.dwDiag.orbit = "LOADED";
    } catch (err) {
      window.dwDiag.orbit = "FAILED: " + err.message;
    }
    dwRenderPanel();

    const vertCount = meshData.vertices.length;
    const triCount = meshData.triangles.length;
    window.dwDiag.sceneData = "INLINED, parsed OK";
    window.dwDiag.vertices = vertCount;
    window.dwDiag.triangles = triCount;
    dwRenderPanel();
    if (vertCount === 0 || triCount === 0) {
      window.dwDiag.error = "empty mesh (vertices=" + vertCount + ", triangles=" + triCount + ")";
      dwRenderPanel();
      dwShowViewerError("This scene has no reconstructed surface to display. Try re-running analysis, or a different image.");
      return;
    }

    const bboxMin = meshData.bbox_min, bboxMax = meshData.bbox_max;
    const bboxFinite = bboxMin.every(Number.isFinite) && bboxMax.every(Number.isFinite);
    if (!bboxFinite) {
      window.dwDiag.error = "non-finite bounding box: [" + bboxMin + "] .. [" + bboxMax + "]";
      dwRenderPanel();
      dwShowViewerError("The reconstructed surface has invalid geometry. Try a different image.");
      return;
    }
    window.dwDiag.bbox = "[" + bboxMin.map(function(v){return v.toFixed(3);}).join(",") +
      "] .. [" + bboxMax.map(function(v){return v.toFixed(3);}).join(",") + "]";
    dwRenderPanel();

    let width = container.clientWidth || 800;
    const height = 640;
    window.dwDiag.canvas = width + "x" + height;
    dwRenderPanel();

    const center = new THREE.Vector3(
      (bboxMin[0] + bboxMax[0]) / 2, (bboxMin[1] + bboxMax[1]) / 2, (bboxMin[2] + bboxMax[2]) / 2
    );
    const extent = Math.max(
      bboxMax[0] - bboxMin[0], bboxMax[1] - bboxMin[1], bboxMax[2] - bboxMin[2], 1e-6
    );

    // Robust, extent-derived clipping planes -- not arbitrary constants.
    // A fixed near/far pair only works for one scene scale; deriving both
    // from the actual bounding box is what fixes "zooming makes the scene
    // disappear" for any input, not just the one it happened to be tuned on.
    const near = Math.max(extent * 0.001, 1e-5);
    const far = Math.max(extent * 100, near * 100);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e14);
    const camera = new THREE.PerspectiveCamera(55, width / height, near, far);

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
      window.dwDiag.webgl = "AVAILABLE";
    } catch (err) {
      window.dwDiag.webgl = "FAILED: " + err.message;
      window.dwDiag.renderer = "FAILED";
      dwRenderPanel();
      dwShowViewerError("Your browser could not initialize WebGL. Try a different browser or enable hardware acceleration.");
      return;
    }
    const pixelRatio = window.devicePixelRatio || 1;
    renderer.setPixelRatio(pixelRatio);
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);
    window.dwDiag.renderer = "INITIALIZED";
    dwRenderPanel();

    // A real surface needs lighting; the old point cloud did not.
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9);
    sun.position.set(center.x + extent, center.y + extent, center.z + extent * 1.5);
    scene.add(sun);

    const positions = new Float32Array(vertCount * 3);
    const normalsArr = new Float32Array(vertCount * 3);
    const colorsRGB = new Float32Array(vertCount * 3);
    const colorsElev = new Float32Array(vertCount * 3);
    for (let i = 0; i < vertCount; i++) {
      const v = meshData.vertices[i];
      const n = meshData.normals[i];
      const cr = meshData.colors_rgb[i];
      const ce = meshData.colors_elevation[i];
      positions[i*3]=v[0]; positions[i*3+1]=v[1]; positions[i*3+2]=v[2];
      normalsArr[i*3]=n[0]; normalsArr[i*3+1]=n[1]; normalsArr[i*3+2]=n[2];
      colorsRGB[i*3]=cr[0]/255; colorsRGB[i*3+1]=cr[1]/255; colorsRGB[i*3+2]=cr[2]/255;
      colorsElev[i*3]=ce[0]/255; colorsElev[i*3+1]=ce[1]/255; colorsElev[i*3+2]=ce[2]/255;
    }
    const indices = new Uint32Array(triCount * 3);
    for (let i = 0; i < triCount; i++) {
      const t = meshData.triangles[i];
      indices[i*3]=t[0]; indices[i*3+1]=t[1]; indices[i*3+2]=t[2];
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("normal", new THREE.BufferAttribute(normalsArr, 3));
    const colorAttrRGB = new THREE.BufferAttribute(colorsRGB, 3);
    const colorAttrElev = new THREE.BufferAttribute(colorsElev, 3);
    geometry.setAttribute("color", colorAttrRGB);
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));

    // vertexColors draped per-vertex from the source RGB image (perfect
    // per-vertex correspondence by construction, since each vertex
    // samples both elevation and RGB at the same source pixel) --
    // deliberately not a UV-mapped texture, which would need a real
    // camera projection this pipeline does not have. side:DoubleSide
    // avoids the surface vanishing from backface culling if triangle
    // winding is viewed from "the wrong side".
    const material = new THREE.MeshStandardMaterial({
      vertexColors: true, side: THREE.DoubleSide, roughness: 1.0, metalness: 0.0,
    });
    const terrainMesh = new THREE.Mesh(geometry, material);
    scene.add(terrainMesh);

    camera.position.set(center.x + extent * 1.4, center.y + extent * 1.4, center.z + extent * 1.1);
    camera.lookAt(center);
    const defaultCamPos = camera.position.clone();

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(center);
    controls.enableDamping = true;
    // Zoom bounds derived from the same extent as the clip planes -- the
    // user physically cannot zoom past either clipping plane, which was
    // the earlier cause of the scene appearing to vanish when zooming.
    controls.minDistance = extent * 0.08;
    controls.maxDistance = extent * 30;
    controls.update();

    function dwFitCamera() {
      camera.position.copy(defaultCamPos);
      controls.target.copy(center);
      camera.lookAt(center);
      controls.update();
    }

    // -- Visual mode: RGB / Elevation color, independent Wireframe toggle --
    const rgbBtn = document.getElementById("dw-btn-rgb");
    const elevBtn = document.getElementById("dw-btn-elevation");
    const wireBtn = document.getElementById("dw-btn-wireframe");
    function dwSetColorMode(mode) {
      geometry.setAttribute("color", mode === "elevation" ? colorAttrElev : colorAttrRGB);
      window.dwDiag.mode = mode;
      if (rgbBtn) rgbBtn.classList.toggle("active", mode !== "elevation");
      if (elevBtn) elevBtn.classList.toggle("active", mode === "elevation");
      dwRenderPanel();
    }
    if (rgbBtn) rgbBtn.addEventListener("click", function () { dwSetColorMode("rgb"); });
    if (elevBtn) elevBtn.addEventListener("click", function () { dwSetColorMode("elevation"); });
    if (wireBtn) wireBtn.addEventListener("click", function () {
      material.wireframe = !material.wireframe;
      wireBtn.classList.toggle("active", material.wireframe);
    });

    // -- Fit / Reset. Drag-to-orbit is always on (via OrbitControls) except
    // while Auto-Fly is running -- no separate toggle button for it, per
    // the compact-toolbar requirement (RGB/Elevation/Wireframe/Fit/Reset/
    // Auto-Fly only). --
    const fitBtn = document.getElementById("dw-btn-fit");
    const resetBtn = document.getElementById("dw-btn-reset");
    const diagBtn = document.getElementById("dw-btn-diag");
    const diagPanel = document.getElementById("dw-diag-panel");
    if (fitBtn) fitBtn.addEventListener("click", dwFitCamera);
    if (resetBtn) resetBtn.addEventListener("click", dwFitCamera);
    if (diagBtn) diagBtn.addEventListener("click", function () {
      const visible = diagPanel.style.display !== "none";
      diagPanel.style.display = visible ? "none" : "block";
      diagBtn.classList.toggle("active", !visible);
    });

    // -- Auto-Fly: a deterministic scripted camera path, not a spin around
    // a fixed point. Orbits around the surface while gently varying radius
    // and height (elevation is the Z axis in this project's coordinate
    // convention -- see reconstruction.placeholder_preview) so the terrain
    // relief is actually visible during the fly-through, not just a flat
    // circling motion. This is a scripted path, not collision-aware
    // navigation -- it does not detect or avoid the mesh, it is simply
    // kept comfortably above the reconstructed surface by construction.
    const flyBtn = document.getElementById("dw-btn-fly");
    let flyEnabled = false;
    let flyAngle = 0;
    let flyLast = performance.now();
    const flyRadius = extent * 1.6;
    const flyBaseHeight = center.z + extent * 1.1;
    function dwUpdateFlyThrough(now) {
      const dt = Math.min((now - flyLast) / 1000, 0.1);
      flyLast = now;
      flyAngle += dt * 0.35;
      const wobble = Math.sin(flyAngle * 0.3) * extent * 0.25;
      const radius = flyRadius + wobble * 0.4;
      camera.position.x = center.x + Math.cos(flyAngle) * radius;
      camera.position.y = center.y + Math.sin(flyAngle) * radius;
      camera.position.z = flyBaseHeight + wobble;
      camera.lookAt(center);
    }
    if (flyBtn) flyBtn.addEventListener("click", function () {
      flyEnabled = !flyEnabled;
      flyLast = performance.now();
      controls.enabled = !flyEnabled;  // don't fight the scripted path
      flyBtn.classList.toggle("active", flyEnabled);
    });

    window.addEventListener("resize", function () {
      const w = container.clientWidth || width;
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
      renderer.setSize(w, height);
    });

    function animate() {
      requestAnimationFrame(animate);
      if (flyEnabled) {
        dwUpdateFlyThrough(performance.now());
      } else {
        controls.update();
      }
      renderer.render(scene, camera);
    }
    window.dwDiag.renderLoop = "RUNNING";
    window.dwDiag.renderInitMs = (performance.now() - dwStartTime).toFixed(1) + " ms";
    dwRenderPanel();
    dwHideLoading();
    animate();
  } catch (err) {
    window.dwDiag.error = (window.dwDiag.error ? window.dwDiag.error + " | " : "") + "top-level: " + err.message;
    dwRenderPanel();
    dwShowViewerError("The 3D viewer failed to initialize (" + err.message + "). Diagnostics are available via the Diagnostics button.");
  }
})();
</script>
"""

viewer_html = _VIEWER_TEMPLATE.replace("__DW_MESH_JSON__", mesh_json)
st.components.v1.html(viewer_html, height=680, scrolling=True)
st.caption(
    "Drag to orbit, scroll to zoom, right-drag to pan. RGB / Elevation switch the surface "
    "coloring; Wireframe shows the underlying mesh structure. “Auto-Fly” follows a scripted "
    "camera path around the surface (not collision-aware navigation); “Fit” / “Reset” return "
    "to the starting view. This is an uncalibrated reconstructed surface (calibrated=False), "
    "not surveyed 3D geometry."
)

st.markdown('<hr class="dw-section-rule" />', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------
st.header("Export")
ex1, ex2 = st.columns(2)
with ex1:
    st.download_button(
        "Download scene JSON", data=json.dumps(scene),
        file_name="depthwizard_scene.json", mime="application/json",
        use_container_width=True,
    )
    st.caption(
        "Reconstructed surface mesh (vertices, triangles, normals, RGB + elevation "
        "colors) plus provenance (depth source/status, calibration=False)."
    )
with ex2:
    if meta.is_georeferenced and rasterio_available():
        try:
            out_path = OUTPUT_DIR / "relative_elevation.tif"
            export_elevation_geotiff(
                elevation, out_path, transform=meta.transform, crs_wkt=meta.crs_wkt,
            )
            with open(out_path, "rb") as fh:
                st.download_button(
                    "Download relative elevation GeoTIFF", data=fh.read(),
                    file_name="relative_elevation.tif", mime="image/tiff",
                    use_container_width=True,
                )
            st.caption("Georeferenced raster of RELATIVE elevation values (not metric).")
        except RuntimeError as exc:
            render_error("GeoTIFF export failed.", str(exc), "Check that rasterio is installed and the input carries a valid CRS/transform.")
    else:
        st.button("Download relative elevation GeoTIFF", disabled=True, use_container_width=True)
        st.caption("Unavailable: input is not georeferenced and/or rasterio is not installed.")
