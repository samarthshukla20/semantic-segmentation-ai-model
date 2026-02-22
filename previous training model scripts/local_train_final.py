import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import random

# --- CONFIGURATION (TUNED FOR RTX 4050 6GB) ---
BASE_DIR = os.path.join("Offroad_Segmentation_Training_Dataset")
if not os.path.exists(os.path.join(BASE_DIR, "train")):
    BASE_DIR = os.path.join("Offroad_Segmentation_Training_Dataset", "Offroad_Segmentation_Training_Dataset")

print(f"📂 Reading data from: {os.path.abspath(BASE_DIR)}")

ID_MAPPING = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 
    600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}
CLASSES = ['Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 'Ground Clutter', 
           'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- ⚠️ CRITICAL SETTINGS FOR 6GB VRAM ---
BATCH_SIZE = 2        # Lowered to preventing crashing
IMAGE_SIZE = 512      # High Definition (See small rocks!)
EPOCHS = 10           # 10 epochs at 512px > 50 epochs at 256px
# -----------------------------------------

class DesertDataset(Dataset):
    def __init__(self, root_dir, split="train", augment=False):
        self.images_dir = os.path.join(root_dir, split, "Color_Images")
        self.masks_dir = os.path.join(root_dir, split, "Segmentation")
        self.ids = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.augment = augment

    def __getitem__(self, i):
        img_name = self.ids[i]
        img_path = os.path.join(self.images_dir, img_name)
        
        mask_name = img_name.replace(".jpg", ".png").replace(".jpeg", ".png")
        mask_path = os.path.join(self.masks_dir, mask_name)
        if not os.path.exists(mask_path): mask_path = os.path.join(self.masks_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        
        mask = cv2.imread(mask_path, -1)
        if mask is None: mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int64)
        else: mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        # --- AUGMENTATION ---
        if self.augment:
            if random.random() > 0.5: # Horizontal
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            if random.random() > 0.5: # Vertical
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)

        # Map Mask
        new_mask = np.full_like(mask, 8) 
        for raw_id, map_id in ID_MAPPING.items():
            new_mask[mask == raw_id] = map_id
        mask = new_mask

        image = image.astype('float32') / 255.0
        image = image.transpose(2, 0, 1)
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()

    def __len__(self): return len(self.ids)

def train():
    print(f"🚀 ULTRA-MODE: Training @ {IMAGE_SIZE}x{IMAGE_SIZE} on {DEVICE}")
    
    # Enable Augmentation
    train_dataset = DesertDataset(BASE_DIR, split="train", augment=True)
    # num_workers=0 is safer for Windows to avoid memory leaks
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', classes=len(CLASSES))
    model.to(DEVICE)
    
    # Load your 62% model to make it even smarter
    if os.path.exists("best_model.pth"):
        print("🔄 Loading 62% model weights... Upgrading resolution!")
        try:
            state_dict = torch.load("best_model.pth")
            model.load_state_dict(state_dict)
            print("✅ Weights loaded successfully.")
        except:
            print("⚠️ Size mismatch (likely from old architecture). Starting fresh!")
    
    # Very low learning rate for Fine-Tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    
    # Combined Loss (Standard + Dice)
    criterion_dice = smp.losses.DiceLoss(mode='multiclass')
    criterion_ce = torch.nn.CrossEntropyLoss()

    print("🔥 Starting High-Res Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(images)
            
            loss = criterion_ce(output, masks) + criterion_dice(output, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Print every 50 batches (since batch size is small, we have many batches)
            if i % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), 'best_model.pth')

if __name__ == "__main__":
    train()