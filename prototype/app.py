"""
SUPERSEDED -- kept only for historical reference (Phase 0 audit).

Replaced by tools/dev_streamlit_harness.py, which:
  - fixes the confirmed crash in the "Estimate depth-span" button below
    (empty-slice IndexError for most realistic box selections -- see the
    Phase 0 audit, Section 4), by removing that feature entirely rather
    than papering over it, since its underlying math was also invalid
    (see the audit, Section 3);
  - routes image loading and depth inference through the new
    depthwizard.io / depthwizard.depth modules instead of inline,
    silently-swallowed logic.

Do not run this file as the dev harness going forward. Do not extend it.
"""

import time, math
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

def load_pipe():
    try:
        from transformers import pipeline
        return pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    except Exception:
        return None

def estimate(img, pipe):
    if pipe is None:
        g=np.asarray(img.convert("L"),dtype=np.float32)/255.0
        d=cv2.GaussianBlur(g,(0,0),5)
        d=(1-d)*0.7+0.3
        return d.astype(np.float32), "demo-fallback"
    o=pipe(img)
    d=np.asarray(o["depth"],dtype=np.float32)
    d=(d-d.min())/(d.max()-d.min()+1e-8)
    return d, "Depth Anything V2"

def colour(d):
    u=(np.clip(d,0,1)*255).astype(np.uint8)
    return cv2.applyColorMap(u,cv2.COLORMAP_TURBO)[:,:,::-1]

def to_xyz(depth):
    h,w=depth.shape
    yy,xx=np.indices((h,w))
    fx=0.9*w; fy=0.9*w; cx=(w-1)/2; cy=(h-1)/2
    X=(xx-cx)*depth/fx; Y=(yy-cy)*depth/fy; Z=depth
    pts=np.stack([X,Y,Z],axis=-1).reshape(-1,3)
    return pts[::max(1,int(math.sqrt(len(pts)/120000)))]

def save_ply(p, path):
    with open(path,"w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(p)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        for q in p: f.write(f"{q[0]:.5f} {q[1]:.5f} {q[2]:.5f}\n")

try:
    import streamlit as st
except Exception:
    raise SystemExit("Install dependencies: pip install -r requirements.txt")

st.set_page_config(page_title="DepthWizard",layout="wide")
st.title("DepthWizard")
st.caption("SIH26175 • Single-View Height Estimation + 3D Fly-Through • MVP")
pipe=load_pipe()
up=st.file_uploader("Upload a single image",type=["png","jpg","jpeg","tif","tiff"])
anchor=st.number_input("Optional reference dimension (m)",min_value=0.0,value=0.0,step=0.1)
if up:
    img=Image.open(up).convert("RGB")
    t=time.perf_counter(); d,mode=estimate(img,pipe); ms=(time.perf_counter()-t)*1000
    c1,c2,c3=st.columns(3)
    with c1: st.image(img,caption="Input",use_container_width=True)
    with c2: st.image(colour(d),caption="Depth",use_container_width=True)
    with c3:
        st.metric("Inference",f"{ms:.0f} ms")
        st.metric("Engine",mode)
        st.info("Competition build: replace the MVP confidence proxy with calibrated depth/scale/geometry uncertainty.")
    if anchor>0: metric=d*anchor
    else: metric=d
    st.subheader("3D point cloud")
    if st.button("Generate PLY"):
        pts=to_xyz(metric)
        out=Path("depthwizard_scene.ply"); save_ply(pts,out)
        st.success(f"Generated {len(pts):,} points")
        st.download_button("Download PLY",out.read_bytes(),"depthwizard_scene.ply")
    st.subheader("Height")
    st.caption("MVP only: use a selected region; the full competition system must fit camera/ground geometry before reporting vertical height.")
    cols=st.columns(4)
    xy=[cols[i].number_input(k,min_value=0,value=0,step=1) for i,k in enumerate(["x1","y1","x2","y2"])]
    if st.button("Estimate depth-span"):
        x1,y1,x2,y2=map(int,xy)
        if x2>x1 and y2>y1:
            a=float(np.percentile(metric[y1:max(y1+3,(y1+y2)//5),x1:x2],70))
            b=float(np.percentile(metric[max(y1,(y1+y2)*4//5):y2,x1:x2],30))
            st.success(f"Depth-span proxy: {abs(a-b):.2f} m")
        else: st.error("Invalid region.")
