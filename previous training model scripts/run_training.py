import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset

# --- CONFIGURATION ---
# We look for the folder structure you showed me in the screenshot
# The script checks for the "double folder" nesting automatically.
BASE_DIR = os.path.join("Offroad_Segmentation_Training_Dataset", "Offroad_Segmentation_Training_Dataset")

# Fallback: If the double nesting isn't there, we look at the single folder
if not os.path.exists(BASE_DIR):
    BASE_DIR = "Offroad_Segmentation_Training_Dataset"

print(f"📂 Reading data from: {BASE_DIR}")

# Hyperparameters for RTX 4050
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 'Ground Clutter', 
           'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
IMAGE_SIZE = 256
EPOCHS = 15

class DesertDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        # Matches your structure: root/train/Color_Images
        self.images_dir = os.path.join(root_dir, split, "Color_Images")
        self.masks_dir = os.path.join(root_dir, split, "Segmentation")
        
        # Verify folders exist
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"❌ Could not find images at: {self.images_dir}")
        
        # Get list of images
        self.ids = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"✅ Found {len(self.ids)} images in {split} set.")

    def __getitem__(self, i):
        img_name = self.ids[i]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Try to find the matching mask (handles .jpg vs .png differences)
        mask_name = img_name.replace(".jpg", ".png").replace(".jpeg", ".png")
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        # Fallback if mask has the exact same name as image
        if not os.path.exists(mask_path):
             mask_path = os.path.join(self.masks_dir, img_name)

        # Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Load Mask
        mask = cv2.imread(mask_path, 0)
        
        # Safety net: If mask is missing, create a blank one (prevents crash)
        if mask is None:
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        else:
            mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        # Normalize
        image = image.astype('float32') / 255.0
        image = image.transpose(2, 0, 1)
        
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()

    def __len__(self):
        return len(self.ids)

def train():
    print("🚀 Initializing Training...")
    
    # Load Datasets
    try:
        train_dataset = DesertDataset(BASE_DIR, split="train")
        val_dataset = DesertDataset(BASE_DIR, split="val")
    except FileNotFoundError as e:
        print(e)
        print("⚠️ Check your folder structure again!")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model Setup
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=len(CLASSES), activation=None)
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()

    print("🔥 Starting Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.4f}")

        print(f"✅ Epoch {epoch+1} Complete! Avg Loss: {total_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), "best_model.pth")

if __name__ == "__main__":
    train()