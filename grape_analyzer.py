import cv2
import numpy as np
import pandas as pd
import streamlit as st
import os
import sys
import torch
from datetime import datetime
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ===== 1. ENHANCED LOGGING =====
def log(message):
    """Force logs to show in Streamlit Cloud"""
    print(message, file=sys.stderr)
    st.toast(message)

# ===== 2. MODEL LOADING =====
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
MODEL_NAME = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"

@st.cache_resource
def load_sam():
    try:
        log("Starting model initialization...")
        
        # Download model if missing
        if not os.path.exists(MODEL_NAME):
            log(f"Downloading {MODEL_NAME}...")
            import urllib.request
            urllib.request.urlretrieve(MODEL_URL, MODEL_NAME)
            
            if os.path.getsize(MODEL_NAME) < 300_000_000:
                raise ValueError("Downloaded file too small")

        # Force CPU on Streamlit Cloud
        device = "cpu"
        log(f"Loading model to {device}...")
        
        sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_NAME).to(device)
        log("Model loaded successfully!")
        return SamAutomaticMaskGenerator(sam)
        
    except Exception as e:
        log(f"❌ Model loading failed: {str(e)}")
        st.error(f"Model initialization error: {str(e)}")
        st.stop()

# ===== 3. ANALYSIS =====
def analyze_grape(uploaded_file):
    try:
        log("Starting analysis...")
        
        # Read image
        uploaded_file.seek(0)
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        log("Image processed")
        
        # Generate masks
        log("Generating masks...")
        masks = sam.generate(img_rgb)
        if not masks:
            raise ValueError("No objects detected")

        # Process mask
        mask = max(masks, key=lambda x: x["area"])["segmentation"]
        mask = mask.astype(np.uint8)
        log("Mask created")
        
        # Calculate metrics
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        avg_hue = np.mean(hsv[:, :, 0][mask > 0])
        log(f"Calculated hue: {avg_hue:.2f}")
        
        return mask, avg_hue
        
    except Exception as e:
        log(f"Analysis error: {str(e)}")
        return None, None

# ===== 4. STREAMLIT UI =====
st.title("🍇 Grape Analyzer")

# Initialize with progress
with st.spinner("Loading AI model..."):
    sam = load_sam()

uploaded_file = st.file_uploader("Upload grape image", type=["jpg","png","jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Original", use_column_width=True)
    
    with st.spinner("Analyzing..."):
        mask, hue = analyze_grape(uploaded_file)
    
    if mask is not None:
        with col2:
            st.image(mask*255, caption="Grape Mask", clamp=True)
        st.success(f"Average Hue: {hue:.2f}")
    else:
        st.warning("Analysis failed - try another image")