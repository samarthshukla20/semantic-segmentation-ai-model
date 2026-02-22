import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import random

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = os.path.join("Offroad_Segmentation_Training_Dataset")
if not os.path.exists(os.path.join(BASE_DIR, "val")):
    BASE_DIR = os.path.join("Offroad_Segmentation_Training_Dataset", "Offroad_Segmentation_Training_Dataset")

VAL_IMG_DIR = os.path.join(BASE_DIR, "val", "Color_Images")
VAL_MASK_DIR = os.path.join(BASE_DIR, "val", "Segmentation")

CLASSES = ['Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 'Ground Clutter', 
           'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky']
# Mapping from the PDF
ID_MAPPING = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 
    600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

def map_mask(mask):
    # Create an empty mask filled with "Landscape" (8) as default to avoid 0s
    new_mask = np.full_like(mask, 8) 
    for raw_id, map_id in ID_MAPPING.items():
        new_mask[mask == raw_id] = map_id
    return new_mask

def check_performance():
    model = smp.Unet(encoder_name='resnet34', classes=len(CLASSES))
    try:
        model.load_state_dict(torch.load("best_model.pth"))
    except FileNotFoundError:
        print("❌ best_model.pth not found!")
        return

    model.to(DEVICE)
    model.eval()

    # Get a random image
    all_images = os.listdir(VAL_IMG_DIR)
    img_name = random.choice(all_images)
    print(f"📸 Checking Image: {img_name}")

    # 1. Load Image
    img_path = os.path.join(VAL_IMG_DIR, img_name)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vis_image = cv2.resize(image, (256, 256))

    # 2. Load Mask (THE FIX: Use -1 to read raw values)
    mask_name = img_name.replace(".jpg", ".png").replace(".jpeg", ".png")
    mask_path = os.path.join(VAL_MASK_DIR, mask_name)
    
    # Handle if mask name is same as image name
    if not os.path.exists(mask_path): 
        mask_path = os.path.join(VAL_MASK_DIR, img_name)
        
    true_mask = cv2.imread(mask_path, -1) # -1 reads exact values (100, 200, etc)
    
    if true_mask is None:
        print(f"❌ ERROR: Could not read mask at {mask_path}")
        return

    true_mask = cv2.resize(true_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    
    # DEBUG: Show us what is inside the mask
    print(f"🧐 Unique values found in mask file: {np.unique(true_mask)}")
    
    true_mask_mapped = map_mask(true_mask)

    # 3. Predict
    input_tensor = vis_image.astype('float32') / 255.0
    input_tensor = input_tensor.transpose(2, 0, 1)
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # 4. Calculate Accuracy
    match = (pred_mask == true_mask_mapped)
    accuracy = match.mean() * 100
    print(f"📊 Pixel Accuracy: {accuracy:.2f}%")

    # 5. Visualize
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(vis_image)
    ax[0].set_title("Original")
    
    # Use 'jet' so different numbers show different colors
    ax[1].imshow(true_mask_mapped, cmap='jet', vmin=0, vmax=9)
    ax[1].set_title("Ground Truth (Corrected)")
    
    ax[2].imshow(pred_mask, cmap='jet', vmin=0, vmax=9)
    ax[2].set_title(f"AI Prediction (Acc: {accuracy:.1f}%)")
    
    for a in ax: a.axis('off')
    plt.show()

if __name__ == "__main__":
    check_performance()