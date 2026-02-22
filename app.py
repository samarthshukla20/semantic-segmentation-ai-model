import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image
import os

# --- CONFIGURATION & COLORS ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CLASSES = ['Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 'Ground Clutter', 
           'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky']

COLORS = np.array([
    [34, 139, 34],   # Trees
    [154, 205, 50],  # Lush Bushes
    [218, 165, 32],  # Dry Grass
    [139, 69, 19],   # Dry Bushes
    [128, 128, 128], # Ground Clutter
    [255, 105, 180], # Flowers
    [160, 82, 45],   # Logs
    [105, 105, 105], # Rocks
    [244, 164, 96],  # Landscape
    [135, 206, 235]  # Sky
], dtype=np.uint8)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = smp.Unet(encoder_name='resnet34', classes=10)
    
    if not os.path.exists("best_model.pth"):
        st.error("⚠️ best_model.pth not found! Please copy it from Laptop 1 into this folder.")
        return None
        
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# --- APP UI ---
st.set_page_config(page_title="Offroad AI", layout="wide")
st.title("🏜️ Offroad Environment Segmentation AI")
st.markdown("Upload a terrain image to let the AI map the environment for autonomous offroading.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # 1. Read Image
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    vis_image = cv2.resize(image_np, (512, 512))

    # 2. Preprocess for AI
    input_image = cv2.resize(image_np, (256, 256))
    input_tensor = input_image.astype('float32') / 255.0
    input_tensor = input_tensor.transpose(2, 0, 1)
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).float().to(DEVICE)

    # 3. Predict & Calculate Confidence
    with st.spinner('AI is mapping the terrain...'):
        with torch.no_grad():
            output = model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            # Calculate Live Confidence
            probabilities = torch.nn.functional.softmax(output, dim=1)
            max_probs, _ = torch.max(probabilities, dim=1)
            mean_confidence = max_probs.mean().item() * 100

    # 4. Colorize the Prediction
    pred_mask_resized = cv2.resize(pred_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    color_mask = COLORS[pred_mask_resized]

    # 5. Display Side-by-Side Images
    col1, col2 = st.columns(2)
    with col1:
        st.header("Original Image")
        st.image(vis_image, use_container_width=True)
    with col2:
        st.header("AI Perception")
        st.image(color_mask, use_container_width=True)

    # 6. Display Metrics Dashboard
    st.divider()
    st.markdown("### 📊 Model Performance & Analysis")
    metric1, metric2, metric3 = st.columns(3)
    
    with metric1:
        st.metric(
            label="Live AI Confidence", 
            value=f"{mean_confidence:.1f}%", 
            help="How sure the AI is about its prediction on this specific image."
        )
    with metric2:
        st.metric(
            label="Baseline Pixel Accuracy", 
            value="87.78%", 
            help="Pre-calculated accuracy on the validation dataset."
        )
    with metric3:
        st.metric(
            label="Baseline Mean IoU", 
            value="65.38%", 
            help="Pre-calculated Intersection over Union on the validation dataset."
        )

    # Expandable section for deep dive stats
    with st.expander("🔍 View Detailed Object Detection Scores (IoU)"):
        st.markdown("""
        Our **V3 Model** utilizes hybrid CrossEntropy + Dice Loss alongside geometric augmentation to successfully detect difficult micro-terrain features.
        
        * **Sky**: 98.73%
        * **Trees**: 87.63%
        * **Dry Grass**: 70.37%
        * **Lush Bushes**: 70.14%
        * **Landscape**: 69.78%
        * **Flowers**: 64.22%
        * **Logs**: 56.21% *(Massive improvement)*
        * **Dry Bushes**: 48.93%
        * **Rocks**: 47.84%
        * **Ground Clutter**: 39.98%
        """)

    # 7. Display Legend
    st.divider()
    st.markdown("### 🗺️ Terrain Legend")
    legend_cols = st.columns(5)
    for i, class_name in enumerate(CLASSES):
        col_idx = i % 5
        color_hex = '#%02x%02x%02x' % tuple(COLORS[i])
        with legend_cols[col_idx]:
            st.markdown(f"<div style='background-color:{color_hex}; padding:10px; border-radius:5px; text-align:center; color:white; font-weight:bold; margin-bottom:10px;'>{class_name}</div>", unsafe_allow_html=True)