from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from road_damage.config import MODEL_PATH
from road_damage.detector import ModelUnavailableError, load_yolo, predict_image
from road_damage.image_processing import find_damage_candidates
from road_damage.reporting import detections_to_csv_bytes, summarize_detections

st.set_page_config(page_title="Road Damage Detector", page_icon="🛣️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1250px;}
.hero {padding: 1.25rem 1.5rem; border-radius: 16px; background: linear-gradient(120deg,#0d1b2a,#1b4965); color: #f8fafc; margin-bottom: 1rem;}
.hero h1 {font-size: 2.0rem; margin: 0 0 .35rem 0; color: #ffffff;}
.hero p {margin: 0; color: #d8e4ee;}
.muted {color: #667085; font-size: .92rem;}
</style>
<div class="hero">
<h1>Road Damage Analysis</h1>
<p>OpenCV candidate visualisation and YOLO-based four-class road damage detection.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Analysis settings")
    mode = st.radio("Analysis mode", ["OpenCV candidate analysis", "YOLO detector"])
    conf = st.slider("YOLO confidence threshold", 0.05, 0.95, 0.25, 0.05, disabled=(mode != "YOLO detector"))
    st.divider()
    st.caption("RDD2022 classes")
    st.markdown("- **D00** Longitudinal crack\n- **D10** Transverse crack\n- **D20** Alligator crack\n- **D40** Pothole")
    st.caption(f"Default weights: `{MODEL_PATH}`")

uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "bmp", "webp"])
if uploaded is None:
    st.info("Upload a road image to begin. OpenCV analysis is available immediately; the YOLO option requires trained weights.")
    st.stop()

try:
    rgb = np.array(Image.open(uploaded).convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
except Exception as exc:
    st.error(f"Could not decode the image: {exc}")
    st.stop()

left, right = st.columns([1.3, 1])
with left:
    st.subheader("Input image")
    st.image(rgb, use_container_width=True)

if mode == "OpenCV candidate analysis":
    annotated, candidates, views = find_damage_candidates(bgr)
    out_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    with right:
        st.subheader("Candidate regions")
        st.image(out_rgb, use_container_width=True)
        st.metric("Regions highlighted", len(candidates))
        st.warning("These are edge/contour candidates, not trained D00/D10/D20/D40 classifications. Shadows, road markings and texture can cause false positives.")
    st.subheader("Image-processing stages")
    c1, c2, c3 = st.columns(3)
    c1.image(views["clahe"], caption="CLAHE contrast enhancement", clamp=True, use_container_width=True)
    c2.image(views["denoised"], caption="Gaussian denoising", clamp=True, use_container_width=True)
    c3.image(views["edges"], caption="Canny edges", clamp=True, use_container_width=True)
    if candidates:
        frame = pd.DataFrame([{
            "candidate": i + 1,
            "heuristic_score": item.heuristic_score,
            "area_pixels": item.area,
            "x1": item.x1, "y1": item.y1, "x2": item.x2, "y2": item.y2,
        } for i, item in enumerate(candidates)])
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button("Download candidate CSV", frame.to_csv(index=False).encode("utf-8"), "opencv_candidates.csv", "text/csv")
    else:
        st.info("No candidate regions passed the current contour filters. This does not prove that the road is damage-free.")
else:
    try:
        model = load_yolo()
        annotated, rows = predict_image(bgr, model=model, conf=conf)
        out_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        with right:
            st.subheader("YOLO detections")
            st.image(out_rgb, use_container_width=True)
            st.metric("Detections", len(rows))
        summary = summarize_detections(rows)
        st.subheader("Detection summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.download_button("Download detections CSV", detections_to_csv_bytes(rows), "road_damage_detections.csv", "text/csv")
        if not rows:
            st.info("No detections exceeded the selected confidence threshold. This is not proof that the road is damage-free.")
    except ModelUnavailableError as exc:
        with right:
            st.subheader("YOLO model not configured")
            st.warning(str(exc))
            st.markdown("1. Download and extract the RDD2022 India subset.\n2. Convert Pascal VOC XML annotations with `scripts/prepare_rdd2022.py`.\n3. Run `python train.py ...`.\n4. Set `ROAD_DAMAGE_MODEL` or place `best.pt` at `weights/best.pt`.")

st.divider()
st.markdown("<p class='muted'>Research prototype only. Results require human review and are not an official road-condition or safety assessment.</p>", unsafe_allow_html=True)
