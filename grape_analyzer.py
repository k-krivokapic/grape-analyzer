import cv2
import numpy as np
import pandas as pd
import streamlit as st
import os
import torch
from datetime import datetime
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ===== 1. MODEL LOADING (FIXED) =====
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
MODEL_NAME = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"

def _load_sam_model():
    """Pure model loading without UI elements"""
    if not os.path.exists(MODEL_NAME):
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_NAME)
        
    return sam_model_registry[MODEL_TYPE](checkpoint=MODEL_NAME).to('cpu')

@st.cache_resource
def load_sam():
    """Cached wrapper with progress notification"""
    with st.spinner("🍇 Loading AI model (this happens once)..."):
        return SamAutomaticMaskGenerator(_load_sam_model())

# ===== 2. ANALYSIS =====
def analyze_grape(uploaded_file):
    try:
        # Read image
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Generate masks
        masks = sam.generate(img_rgb)
        if not masks:
            st.warning("No grape detected")
            return None, None

        # Process results
        mask = max(masks, key=lambda x: x["area"])["segmentation"]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        avg_hue = np.mean(hsv[:, :, 0][mask > 0])
        
        return mask.astype(np.uint8), avg_hue
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None, None

# ===== 3. STREAMLIT UI =====
st.title("🍇 Grape Color Analyzer")

# Initialize model (outside cache function)
sam = load_sam()

uploaded_file = st.file_uploader("Upload grape image", type=["jpg","png","jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Original Image")
    
    with st.spinner("Analyzing..."):
        mask, hue = analyze_grape(uploaded_file)
    
    if mask is not None:
        with col2:
            st.image(mask*255, caption="Grape Mask", clamp=True)
        st.success(f"Average Hue: {hue:.2f}")