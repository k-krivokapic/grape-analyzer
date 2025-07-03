import cv2
import numpy as np
import pandas as pd
import streamlit as st
import os
import torch
from datetime import datetime
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Initialize SAM model (with enhanced error handling)
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
MODEL_PATH = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"

@st.cache_resource
def load_sam():
    try:
        # Download model if missing
        if not os.path.exists(MODEL_PATH):
            with st.spinner("Downloading SAM model (375MB)..."):
                import urllib.request
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            
            # Verify download
            if os.path.getsize(MODEL_PATH) < 300_000_000:
                st.error("Model download failed - file too small")
                st.stop()
        
        # Load model with explicit device handling
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.cuda.empty_cache()  # Clear GPU memory
            
        sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH).to(device)
        return SamAutomaticMaskGenerator(sam)
    
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

sam = load_sam()

def analyze_grape(uploaded_file):
    try:
        # Reset file pointer and decode image
        uploaded_file.seek(0)
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error("Failed to decode image")
            return None, None

        # Convert to RGB for SAM
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Generate masks with timeout protection
        try:
            masks = sam.generate(img_rgb)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                st.warning("GPU memory full - retrying with CPU")
                torch.cuda.empty_cache()
                sam.model.to('cpu')
                masks = sam.generate(img_rgb)
            else:
                raise e
                
        if not masks:
            st.warning("No grape detected in image")
            return None, None

        # Process largest mask
        mask = max(masks, key=lambda x: x["area"])["segmentation"]
        mask = mask.astype(np.uint8)
        
        # Calculate color metrics
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        avg_hue = np.mean(hsv[:, :, 0][mask > 0])
        
        # Save results
        save_to_csv(uploaded_file.name, avg_hue)
        
        return mask, avg_hue
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None, None

def save_to_csv(image_name, avg_hue):
    try:
        new_row = pd.DataFrame({
            "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Image": [image_name],
            "Hue": [avg_hue]
        })
        
        if os.path.exists("results.csv"):
            history = pd.read_csv("results.csv")
            new_row = pd.concat([history, new_row])
            
        new_row.to_csv("results.csv", index=False)
    except Exception as e:
        st.error(f"Failed to save results: {str(e)}")

# Streamlit UI
st.title("🍇 Grape Color Analyzer")
uploaded_file = st.file_uploader("Upload grape image", type=["jpg","png","jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Original Image", use_column_width=True)
    
    with st.spinner("Analyzing..."):
        mask, hue = analyze_grape(uploaded_file)
    
    if mask is not None:
        with col2:
            st.image(mask*255, caption="Grape Mask", clamp=True)
        st.success(f"Average Hue: {hue:.2f}")
        
        # Show history
        try:
            history = pd.read_csv("results.csv")
            st.line_chart(history, x="Timestamp", y="Hue")
        except:
            st.info("No history yet")