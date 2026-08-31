"""DepthWizard -- SIH26175 competition demo app.

    streamlit run app/streamlit_app.py

This is the actual SIH-facing MVP (not the dev harness in
tools/dev_streamlit_harness.py, which stays a lighter internal tool).
It wires together, end to end, everything implemented so far:

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
           -> an embedded Three.js viewer (orbit / zoom / pan / auto-orbit
              "fly-through")

Scientific-honesty contract (carried over from M0/M1, unchanged):
    - Every depth result is labeled SUCCESS or DEMO_FALLBACK on screen;
      DEMO_FALLBACK is never allowed to look like a real prediction.
    - The elevation surface is always called "relative elevation" /
      "pseudo-DSM", never "DSM" or "elevation" unqualified, because it is
      an explicit, named reinterpretation of relative depth -- see
      depthwizard.reconstruction.dsm's module docstring.
    - No height number is ever shown in meters unless the user explicitly
      supplied a meters-per-unit reference themselves; the UI always shows
      which case it's in.
    - No accuracy or confidence percentage is fabricated anywhere. The
      "structural detail" heuristic shown in the metrics panel is a real,
      directly-computed statistic of the output (mean elevation gradient
      magnitude) -- it measures how much local structure the model found,
      not how correct that structure is, and is labeled as a heuristic.
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
from depthwizard.reconstruction.scene_export import (
    build_scene,
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

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("DepthWizard")
st.caption("Single-View DSM Reconstruction & 3D Exploration -- SIH26175")

if "scene" not in st.session_state:
    st.session_state.scene = None
    st.session_state.elevation = None
    st.session_state.image = None
    st.session_state.depth_result = None
    st.session_state.meta = None

OUTPUT_DIR = REPO_ROOT / "data" / "outputs"

left, center, right = st.columns([1.1, 2.0, 1.1])

# ---------------------------------------------------------------------------
# LEFT -- input, mode, controls
# ---------------------------------------------------------------------------
with left:
    st.subheader("Input")
    uploaded = st.file_uploader("Aerial / optical image", type=["png", "jpg", "jpeg", "tif", "tiff"])

    st.subheader("Mode")
    allow_fallback = st.checkbox(
        "Allow DEMO_FALLBACK if the real model can't load",
        value=True,
        help="If unchecked, a real Depth Anything V2 failure is shown as an "
             "error instead of silently substituting a non-AI heuristic.",
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
    max_points = st.slider("Max 3D points", 2000, 80000, 30000, step=2000)

    run_clicked = st.button("Run DepthWizard", type="primary", disabled=uploaded is None)

# ---------------------------------------------------------------------------
# Processing (P0.9: every failure surfaced explicitly, nothing silently caught)
# ---------------------------------------------------------------------------
if run_clicked and uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)

    try:
        image, meta = load_image(tmp_path)
    except DepthWizardIOError as exc:
        st.error(f"Could not load image: {exc}")
        st.stop()

    predictor = DepthPredictor()
    try:
        with st.spinner("Running depth inference..."):
            result = predictor.predict(image, allow_fallback=allow_fallback)
    except DepthWizardModelError as exc:
        st.error(f"Depth backend failed and DEMO_FALLBACK was not allowed: {exc}")
        st.stop()

    elevation = depth_to_relative_elevation(result.depth, invert=invert_elevation)
    scene = build_scene(
        elevation, image, depth_source=result.source, depth_status=result.status.value,
        assumed_hfov_deg=assumed_hfov, max_points=max_points,
    )

    st.session_state.image = image
    st.session_state.meta = meta
    st.session_state.depth_result = result
    st.session_state.elevation = elevation
    st.session_state.scene = scene

# ---------------------------------------------------------------------------
# CENTER -- depth / DSM visualization
# ---------------------------------------------------------------------------
with center:
    st.subheader("Depth / Elevation")
    if st.session_state.scene is None:
        st.info("Upload an image and click **Run DepthWizard** to begin.")
    else:
        image = st.session_state.image
        result = st.session_state.depth_result
        elevation = st.session_state.elevation

        if result.status == DepthStatus.DEMO_FALLBACK:
            st.warning(
                f"DEMO_FALLBACK IN USE -- this is NOT a real model prediction. "
                f"Reason: {result.metadata.get('reason_for_fallback')}"
            )
        else:
            st.success(f"Real prediction from {result.source} ({result.inference_time_ms:.0f} ms)")

        depth_u8 = (np.clip(result.depth, 0, 1) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)[:, :, ::-1]

        elev_u8 = (np.clip(elevation, 0, 1) * 255).astype(np.uint8)
        elev_colored = cv2.applyColorMap(elev_u8, cv2.COLORMAP_VIRIDIS)[:, :, ::-1]

        c1, c2 = st.columns(2)
        c1.image(image, caption="Original", use_container_width=True)
        c2.image(depth_colored, caption=f"Relative depth ({result.status.value})", use_container_width=True)
        st.image(
            elev_colored,
            caption="Relative elevation / pseudo-DSM -- NOT metric, NOT a calibrated DSM",
            use_container_width=True,
        )

        st.subheader("Point-to-point height (region/object selection)")
        h, w = elevation.shape
        pc1, pc2, pc3 = st.columns(3)
        r1 = pc1.number_input("Point A row", 0, h - 1, 0)
        c_1 = pc1.number_input("Point A col", 0, w - 1, 0)
        r2 = pc2.number_input("Point B row", 0, h - 1, h - 1)
        c_2 = pc2.number_input("Point B col", 0, w - 1, w - 1)
        meters_per_unit_raw = pc3.text_input(
            "External scale (meters per relative unit) -- optional",
            value="",
            help="Only fill this in if you have an independent, trusted "
                 "reference (e.g. a known building height in this scene). "
                 "Leave blank to keep the result honestly uncalibrated.",
        )
        try:
            mpu = float(meters_per_unit_raw) if meters_per_unit_raw.strip() else None
        except ValueError:
            mpu = None
            st.error("External scale must be a number -- ignoring it.")

        try:
            height_result = relative_height_between_points(
                elevation, (int(r1), int(c_1)), (int(r2), int(c_2)),
                meters_per_unit=mpu,
                calibration_source="user-supplied manual reference" if mpu is not None else None,
            )
        except ValueError as exc:
            st.error(str(exc))
            height_result = None

        if height_result is not None:
            annotated = image.copy()
            if annotated.ndim == 2:
                annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2RGB)
            cv2.drawMarker(annotated, (int(c_1), int(r1)), (255, 0, 0), cv2.MARKER_CROSS, 20, 3)
            cv2.drawMarker(annotated, (int(c_2), int(r2)), (0, 128, 255), cv2.MARKER_CROSS, 20, 3)
            st.image(annotated, caption="Point A (red) / Point B (orange)", use_container_width=True)

            if height_result["calibrated"]:
                st.metric(
                    "Height difference (calibrated)",
                    f"{height_result['meters_difference']:.2f} m",
                    help=f"Source: {height_result['calibration_source']}",
                )
            else:
                st.metric(
                    "Height difference (relative, uncalibrated)",
                    f"{height_result['relative_difference']:.4f}",
                    help="No metric scale available for this image -- shown in "
                         "relative elevation units, not meters.",
                )

# ---------------------------------------------------------------------------
# RIGHT -- metrics, elevation, reliability, export
# ---------------------------------------------------------------------------
with right:
    st.subheader("Metrics")
    if st.session_state.scene is None:
        st.write("No run yet.")
    else:
        result = st.session_state.depth_result
        meta = st.session_state.meta
        elevation = st.session_state.elevation
        stats = elevation_stats(elevation)

        st.metric("Inference time", f"{result.inference_time_ms:.0f} ms")
        st.metric("Resolution", f"{result.shape[1]} x {result.shape[0]}")
        st.write("**Source:**", result.source)
        st.write("**Status:**", result.status.value)
        st.write("**Georeferenced input:**", "yes" if meta.is_georeferenced else "no")
        st.write("**3D calibrated:**", "no (uncalibrated preview geometry)")

        st.subheader("Relative elevation stats")
        st.write(
            f"min={stats['min']:.3f}  max={stats['max']:.3f}  "
            f"mean={stats['mean']:.3f}  std={stats['std']:.3f}  (unit: {stats['unit']})"
        )

        grad_y, grad_x = np.gradient(elevation)
        structural_detail = float(np.mean(np.abs(grad_x)) + np.mean(np.abs(grad_y)))
        st.subheader("Reliability / quality indicator")
        st.metric("Structural detail (heuristic)", f"{structural_detail:.4f}")
        st.caption(
            "A real, directly-computed statistic (mean local elevation-gradient "
            "magnitude) -- NOT an accuracy or confidence score. Higher generally "
            "means the model found more local structure in this image; it says "
            "nothing about whether that structure is geometrically correct. See "
            "evaluation/domain_validation/REPORT.md for qualitative accuracy notes."
        )

        st.subheader("Export")
        scene = st.session_state.scene
        st.download_button(
            "Download scene JSON", data=json.dumps(scene),
            file_name="depthwizard_scene.json", mime="application/json",
        )
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
                    )
            except RuntimeError as exc:
                st.error(f"GeoTIFF export failed: {exc}")
        else:
            st.caption(
                "GeoTIFF export unavailable: input is not georeferenced and/or "
                "rasterio is not installed."
            )

# ---------------------------------------------------------------------------
# BOTTOM -- interactive 3D fly-through
# ---------------------------------------------------------------------------
st.subheader("3D fly-through")
if st.session_state.scene is None:
    st.write("Run DepthWizard to generate a 3D scene.")
else:
    scene = st.session_state.scene
    points_json = json.dumps(scene["points"])
    auto_orbit = st.checkbox("Auto-orbit (fly-through)", value=False)

    viewer_html = f"""
<div id="dw-viewer" style="width:100%;height:520px;background:#111;position:relative;"></div>
<pre id="dw-diag-panel" style="position:absolute;top:8px;left:8px;color:#0f0;background:rgba(0,0,0,0.55);
     font-family:monospace;font-size:11px;line-height:1.4;white-space:pre-wrap;z-index:2;margin:0;
     padding:6px 8px;pointer-events:none;max-width:70%;">[DepthWizard 3D Diagnostics]
Three.js: PENDING</pre>
<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }}
}}
</script>
<script type="module">
// IMPORTANT (found by live debugging on the actual deployment, not assumed):
// a plain, untyped <script> tag injected via Streamlit's components.v1.html
// mechanism here does NOT execute -- only type="module" and type="importmap"
// scripts do. Everything, including the diagnostics-panel scaffolding that
// would normally live in a separate classic <script>, has to live in this
// one module script, or it silently never runs.
window.dwDiag = {{
  three: "PENDING", orbit: "PENDING", sceneData: "PENDING",
  points: 0, finitePoints: 0, bbox: "PENDING", canvas: "PENDING",
  webgl: "PENDING", renderer: "PENDING", renderLoop: "PENDING", error: ""
}};
window.dwModuleRan = true;
function dwRenderPanel() {{
  var d = window.dwDiag;
  var lines = [
    "[DepthWizard 3D Diagnostics]",
    "Three.js: " + d.three,
    "OrbitControls: " + d.orbit,
    "Scene data: " + d.sceneData,
    "Points: " + d.points,
    "Finite points: " + d.finitePoints,
    "Bounding box: " + d.bbox,
    "Canvas: " + d.canvas,
    "WebGL: " + d.webgl,
    "Renderer: " + d.renderer,
    "Render loop: " + d.renderLoop,
  ];
  if (d.error) lines.push("ERROR: " + d.error);
  var el = document.getElementById("dw-diag-panel");
  if (el) el.textContent = lines.join("\\n");
}}
dwRenderPanel();
window.addEventListener("error", function (e) {{
  window.dwDiag.error = (window.dwDiag.error ? window.dwDiag.error + " | " : "") + "window error: " + e.message;
  dwRenderPanel();
}});
window.addEventListener("unhandledrejection", function (e) {{
  var msg = (e.reason && e.reason.message) ? e.reason.message : String(e.reason);
  window.dwDiag.error = (window.dwDiag.error ? window.dwDiag.error + " | " : "") + "unhandled rejection: " + msg;
  dwRenderPanel();
}});

(async () => {{
  const rawPoints = {points_json};
  const container = document.getElementById("dw-viewer");
  try {{
    let THREE;
    try {{
      THREE = await import("three");
      window.dwDiag.three = "LOADED";
    }} catch (err) {{
      window.dwDiag.three = "FAILED: " + err.message;
      dwRenderPanel();
      return;
    }}
    dwRenderPanel();

    let OrbitControls;
    try {{
      const orbitModule = await import("three/addons/controls/OrbitControls.js");
      OrbitControls = orbitModule.OrbitControls;
      window.dwDiag.orbit = "LOADED";
    }} catch (err) {{
      window.dwDiag.orbit = "FAILED: " + err.message;
    }}
    dwRenderPanel();

    const points = rawPoints.filter(p => p.every(v => Number.isFinite(v)));
    window.dwDiag.sceneData = "INLINED, parsed OK";
    window.dwDiag.points = rawPoints.length;
    window.dwDiag.finitePoints = points.length;
    dwRenderPanel();
    if (points.length === 0) {{
      throw new Error("no finite points in scene JSON (point_count=" + rawPoints.length + ")");
    }}

    const width = container.clientWidth || 800;
    const height = 520;
    window.dwDiag.canvas = width + "x" + height;
    dwRenderPanel();

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.001, 10000);

    let renderer;
    try {{
      renderer = new THREE.WebGLRenderer({{ antialias: true }});
      window.dwDiag.webgl = "AVAILABLE";
    }} catch (err) {{
      window.dwDiag.webgl = "FAILED: " + err.message;
      window.dwDiag.renderer = "FAILED";
      dwRenderPanel();
      return;
    }}
    const pixelRatio = window.devicePixelRatio || 1;
    renderer.setPixelRatio(pixelRatio);
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);
    window.dwDiag.renderer = "INITIALIZED";
    dwRenderPanel();

    let minX=Infinity,minY=Infinity,minZ=Infinity,maxX=-Infinity,maxY=-Infinity,maxZ=-Infinity;
    const positions = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);
    for (let i = 0; i < points.length; i++) {{
      const p = points[i];
      positions[i*3]=p[0]; positions[i*3+1]=p[1]; positions[i*3+2]=p[2];
      colors[i*3]=p[3]/255; colors[i*3+1]=p[4]/255; colors[i*3+2]=p[5]/255;
      minX=Math.min(minX,p[0]); maxX=Math.max(maxX,p[0]);
      minY=Math.min(minY,p[1]); maxY=Math.max(maxY,p[1]);
      minZ=Math.min(minZ,p[2]); maxZ=Math.max(maxZ,p[2]);
    }}
    window.dwDiag.bbox = "[" + minX.toFixed(3) + "," + minY.toFixed(3) + "," + minZ.toFixed(3) + "] .. [" + maxX.toFixed(3) + "," + maxY.toFixed(3) + "," + maxZ.toFixed(3) + "]";
    dwRenderPanel();

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    // Confirmed by live testing: points *were* being drawn at the previous
    // fix, but PointsMaterial's default sizeAttenuation=true sizes points in
    // world units, which for this cloud's scale worked out to ~1-2 device
    // pixels -- indistinguishable from "nothing rendered" at a glance.
    // sizeAttenuation:false renders every point at a fixed, generous pixel
    // size regardless of scene scale or camera distance.
    const material = new THREE.PointsMaterial({{
      size: 4 * pixelRatio, sizeAttenuation: false, vertexColors: true,
    }});
    scene.add(new THREE.Points(geometry, material));

    // Hardcoded reference cube -- independent of the real point cloud's
    // data/math. If this ever stops being visible while the diagnostics
    // above all say LOADED/INITIALIZED/RUNNING, the bug is in the real
    // point cloud's data, not in WebGL/Three.js/the canvas itself.
    const center = new THREE.Vector3((minX+maxX)/2, (minY+maxY)/2, (minZ+maxZ)/2);
    const extent = Math.max(maxX-minX, maxY-minY, maxZ-minZ, 1e-6);
    const cubeSize = Math.max(extent * 0.12, 0.02);
    const refCube = new THREE.Mesh(
      new THREE.BoxGeometry(cubeSize, cubeSize, cubeSize),
      new THREE.MeshBasicMaterial({{ color: 0x00ff88, wireframe: true }}),
    );
    refCube.position.copy(center);
    scene.add(refCube);

    const dist = extent * 1.8 + 0.05;
    camera.position.set(center.x + dist, center.y + dist, center.z + dist);
    camera.lookAt(center);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(center);
    controls.enableDamping = true;
    controls.update();

    const autoOrbit = {str(auto_orbit).lower()};
    let angle = 0;
    function animate() {{
      requestAnimationFrame(animate);
      refCube.rotation.x += 0.01;
      refCube.rotation.y += 0.01;
      if (autoOrbit) {{
        angle += 0.004;
        camera.position.x = center.x + Math.cos(angle) * dist;
        camera.position.z = center.z + Math.sin(angle) * dist;
        camera.lookAt(center);
      }}
      controls.update();
      renderer.render(scene, camera);
    }}
    window.dwDiag.renderLoop = "RUNNING";
    dwRenderPanel();
    animate();
  }} catch (err) {{
    window.dwDiag.error = (window.dwDiag.error ? window.dwDiag.error + " | " : "") + "top-level: " + err.message;
    dwRenderPanel();
  }}
}})();
</script>
"""
    st.components.v1.html(viewer_html, height=560, scrolling=True)
    st.caption(
        "Orbit / zoom / pan via mouse. \"Fly-through\" = continuous auto-orbit "
        "around the reconstructed scene. The small rotating green wireframe cube "
        "is a fixed reference object, independent of the real data -- if it's not "
        "visible, the problem is WebGL/canvas, not the point cloud. This is an "
        "uncalibrated preview point cloud (calibrated=False), not surveyed 3D geometry."
    )
