import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os
import gdown
import torch
from datetime import datetime
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# initialize the SAM model
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
MODEL_PATH = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"

@st.cache_resource
def load_sam():
    # Download model if missing
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        with st.spinner("Downloading SAM model (375MB)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    # Verify download
    if os.path.getsize(MODEL_PATH) < 300_000_000:  # ~300MB minimum
        st.error("Model download failed or incomplete. Please try again.")
        st.stop()
        
    # Load with device auto-detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH).to(device)
    return SamAutomaticMaskGenerator(sam)

sam = load_sam()

# method to analyze grape color in an image
def analyze_grape(image_path, output_csv="grape_analysis.csv"):
    # load image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # generate masks
    masks = sam.generate(image_rgb)
    if not masks:
        print("no masks found!")
        return
    
    # get the largest mask (the grape)
    largest_mask = sorted(masks, key=lambda x: x['area'], reverse=True)[0]
    mask = largest_mask['segmentation'].astype("uint8")
    
    # get the hsv grape color values
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    grape_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    
    # calculate average color
    avg_hue = np.mean(grape_hsv[:, :, 0][mask > 0])
    avg_saturation = np.mean(grape_hsv[:, :, 1][mask > 0])
    avg_value = np.mean(grape_hsv[:, :, 2][mask > 0])
    
    # save to csv
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame({
        "Timestamp": [timestamp],
        "Avg_Hue": [avg_hue],
        "Avg_Saturation": [avg_saturation],
        "Avg_Value": [avg_value],
        "Image_Path": [image_path]
    })
    
    # append to existing CSV or create new
    try:
        existing_data = pd.read_csv(output_csv)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    except FileNotFoundError:
        updated_data = new_data
    
    updated_data.to_csv(output_csv, index=False)
    
    return mask, avg_hue

# streamlit UI
st.title("Grape Color Analyzer")
st.markdown("Upload images to analyze color changes over time!")

uploaded_file = st.file_uploader("Choose a grape image:", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Analyzing..."):
        mask, avg_hue = analyze_grape(uploaded_file)
        
    st.success(f"Average Hue: `{avg_hue:.2f}`")
    st.image(mask * 255, caption="Segmentation Mask", clamp=True)