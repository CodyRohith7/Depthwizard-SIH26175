"""Developer harness for DepthWizard -- NOT the SIH deliverable.

This replaces prototype/app.py. It is a fast internal tool for the team to
sanity-check the io -> depth pipeline while building, per the Phase 0
audit's Section 5 recommendation to keep this kind of tool around, clearly
labeled as a dev tool. The interactive Three.js fly-through
(frontend/index.html) is the actual competition-facing deliverable.

Run:
    pip install -e ".[dev]"     # installs streamlit (not installed in the
                                 # environment this file was written in --
                                 # see the top-level README's test log)
    streamlit run tools/dev_streamlit_harness.py

What changed vs. the old prototype/app.py, and why (see Phase 0 audit,
Sections 3, 4, 6):
    - The "Estimate depth-span" height button is GONE. It crashed on most
      realistic selections (confirmed, reproduced in the audit) and even
      when it didn't crash, its output was not a valid height measurement
      -- a single scalar cannot fix both unknowns in the D_metric = a*D_rel
      + b affine calibration. No replacement height feature exists yet;
      real height estimation needs /scale and /geometry, which are not
      implemented until a later milestone. This harness does not offer a
      height number of any kind.
    - Depth inference goes through depthwizard.depth.DepthPredictor instead
      of an inline try/except that silently swapped in a fake depth map.
      Real backend failures are shown to the user, not hidden; the
      DEMO_FALLBACK path is only used when explicitly requested and is
      always labeled as such on screen.
    - Image loading goes through depthwizard.io.load_image, so GeoTIFF
      metadata (CRS, transform, bounds) is surfaced when present, instead
      of being silently discarded by a plain PIL open.
    - The 3D export is gone from this tool. Use
      scripts/export_sample_scene.py + frontend/index.html for the
      Three.js viewer instead -- this harness is for depth/metadata
      inspection only.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from depthwizard.io.loader import load_image, rasterio_available
from depthwizard.io.exceptions import DepthWizardIOError
from depthwizard.depth.predictor import DepthPredictor
from depthwizard.depth.exceptions import DepthWizardModelError
from depthwizard.depth.base import DepthStatus
from depthwizard.logging_setup import configure_logging, get_logger

try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit(
        "streamlit is not installed. Install the `dev` extra: "
        "pip install -e '.[dev]'"
    ) from exc

import cv2

configure_logging()
log = get_logger("depthwizard.tools.dev_streamlit_harness")

st.set_page_config(page_title="DepthWizard (dev harness)", layout="wide")
st.title("DepthWizard -- developer harness")
st.caption(
    "Internal dev tool only -- NOT the SIH deliverable. "
    "See frontend/index.html for the interactive fly-through."
)
if not rasterio_available():
    st.warning(
        "rasterio is not installed in this environment. TIFF/GeoTIFF files "
        "will be read as plain rasters and any embedded CRS/geotransform "
        "will NOT be checked or used. Install the `geo` extra "
        "(`pip install -e '.[geo]'`) for real GeoTIFF support."
    )

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded:
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)

    try:
        image, meta = load_image(tmp_path)
    except DepthWizardIOError as exc:
        st.error(f"Could not load image: {exc}")
        st.stop()

    st.subheader("Metadata (/io)")
    meta_cols = st.columns(4)
    meta_cols[0].metric("Width", meta.width)
    meta_cols[1].metric("Height", meta.height)
    meta_cols[2].metric("Bands", meta.band_count)
    meta_cols[3].metric("Georeferenced", "YES" if meta.is_georeferenced else "no")
    if meta.is_georeferenced:
        st.write("**CRS (WKT):**", meta.crs_wkt)
        st.write("**Affine transform (a,b,c,d,e,f):**", meta.transform)
        st.write("**Bounds (minx,miny,maxx,maxy):**", meta.bounds)
    if meta.notes:
        for note in meta.notes:
            st.info(note)

    st.subheader("Depth (/depth)")
    allow_fallback = st.checkbox(
        "Allow DEMO_FALLBACK if the real model can't be loaded",
        value=True,
        help="If unchecked, a real backend failure will be shown as an error "
             "instead of silently substituting a fake depth map.",
    )
    if st.button("Run depth inference"):
        predictor = DepthPredictor()
        try:
            result = predictor.predict(image, allow_fallback=allow_fallback)
        except DepthWizardModelError as exc:
            st.error(f"Depth backend failed and no fallback was allowed: {exc}")
            st.stop()

        if result.status == DepthStatus.DEMO_FALLBACK:
            st.warning(
                f"DEMO_FALLBACK IN USE -- this is NOT a real model prediction. "
                f"Reason: {result.metadata.get('reason_for_fallback')}"
            )
        else:
            st.success(f"Real prediction from {result.source} ({result.inference_time_ms:.0f} ms)")

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Input", use_container_width=True)
        with col2:
            depth_u8 = (np.clip(result.depth, 0, 1) * 255).astype(np.uint8)
            colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)[:, :, ::-1]
            st.image(
                colored,
                caption=f"Depth ({result.source}, status={result.status.value}) -- "
                        f"RELATIVE depth only, not a measurement",
                use_container_width=True,
            )
