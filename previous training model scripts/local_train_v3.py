import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import random

# --- CONFIGURATION ---
# Robust path finding for your folder structure
BASE_DIR = os.path.join("Offroad_Segmentation_Training_Dataset")
if not os.path.exists(os.path.join(BASE_DIR, "train")):
    BASE_DIR = os.path.join("Offroad_Segmentation_Training_Dataset", "Offroad_Segmentation_Training_Dataset")

print(f"📂 Reading data from: {os.path.abspath(BASE_DIR)}")

# Class Mapping from Hackathon PDF
ID_MAPPING = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 
    600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}
CLASSES = ['Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 'Ground Clutter', 
           'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 6    # Fits in your RTX 4050
IMAGE_SIZE = 256
EPOCHS = 15       # We will train for 15 MORE epochs

class DesertDataset(Dataset):
    def __init__(self, root_dir, split="train", augment=False):
        self.images_dir = os.path.join(root_dir, split, "Color_Images")
        self.masks_dir = os.path.join(root_dir, split, "Segmentation")
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"❌ Path not found: {self.images_dir}")
            
        self.ids = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.augment = augment
        print(f"✅ Loaded {len(self.ids)} images for {split} (Augment={augment})")

    def __getitem__(self, i):
        img_name = self.ids[i]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Smart mask finding (handles .jpg vs .png)
        mask_name = img_name.replace(".jpg", ".png").replace(".jpeg", ".png")
        mask_path = os.path.join(self.masks_dir, mask_name)
        if not os.path.exists(mask_path): mask_path = os.path.join(self.masks_dir, img_name)

        # Load Image & Mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        
        mask = cv2.imread(mask_path, -1) # Read RAW values (100, 200...)
        if mask is None: mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int64)
        else: mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        # --- AUGMENTATION (The Secret Weapon) ---
        if self.augment:
            # Horizontal Flip (Mirror)
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            # Vertical Flip (Upside down - teaches shape independence)
            if random.random() > 0.5:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)
        # ----------------------------------------

        # Map Mask IDs to 0-9
        new_mask = np.full_like(mask, 8) # Default to Landscape (8)
        for raw_id, map_id in ID_MAPPING.items():
            new_mask[mask == raw_id] = map_id
        mask = new_mask

        # Normalize
        image = image.astype('float32') / 255.0
        image = image.transpose(2, 0, 1)
        
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()

    def __len__(self): return len(self.ids)

def train():
    print(f"🚀 V3 Training Started on: {DEVICE}")
    
    # 1. Setup Data with Augmentation
    train_dataset = DesertDataset(BASE_DIR, split="train", augment=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 2. Setup Model
    model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', classes=len(CLASSES))
    model.to(DEVICE)
    
    # 3. Load Previous Weights (Resume from 86%)
    if os.path.exists("best_model.pth"):
        print("🔄 Loading existing model weights to improve them...")
        model.load_state_dict(torch.load("best_model.pth"))
    else:
        print("⚠️ No previous model found. Starting from scratch!")
    
    # 4. Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # Cosine Scheduler: Starts fast, ends slow for precision
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.00001)
    
    # 5. Hybrid Loss Function (The fix for small objects)
    # DiceLoss focuses on overlap (good for small objects like logs/rocks)
    criterion_dice = smp.losses.DiceLoss(mode='multiclass')
    criterion_ce = torch.nn.CrossEntropyLoss()

    print("🔥 Starting Pro Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(images)
            
            # Combined Loss: 50% Standard + 50% Dice
            loss = criterion_ce(output, masks) + criterion_dice(output, masks)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if i % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.4f}")
        
        # Step the scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = total_loss / len(train_loader)
        
        print(f"✅ Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f} (LR: {current_lr:.6f})")
        torch.save(model.state_dict(), 'best_model.pth')

if __name__ == "__main__":
    train()