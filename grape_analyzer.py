import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# initialize the SAM model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

# method to analyze grape color in an image
def analyze_grape(image_path, output_csv="grape_analysis.csv"):
    # load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # generate masks
    masks = mask_generator.generate(image_rgb)
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
    
    return {
        "mask": mask,
        "avg_hue": avg_hue,
        "avg_saturation": avg_saturation,
        "avg_value": avg_value
    }