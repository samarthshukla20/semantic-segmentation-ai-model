import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import random

# --- 1. CONFIGURATION & DEEP SEARCH ---
print("🕵️ Hunting for ALL data sources...")

# We need to find the root folder that contains the subfolders
DATA_ROOT = None
search_paths = [".", "Offroad_Segmentation_Training_Dataset", os.path.join("Offroad_Segmentation_Training_Dataset", "Offroad_Segmentation_Training_Dataset")]

for p in search_paths:
    if os.path.exists(os.path.join(p, "train", "Color_Images")):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    print("❌ Critical Error: Could not find the main dataset folder.")
    exit()

# Check for the Extra Test Folder
EXTRA_DATA_DIR = "Offroad_Segmentation_testImages" 
HAS_EXTRA_DATA = os.path.exists(EXTRA_DATA_DIR)

print(f"✅ Found Main Data at: {DATA_ROOT}")
if HAS_EXTRA_DATA:
    print(f"✅ Found Extra Data at: {EXTRA_DATA_DIR} (Adding to training!)")
else:
    print(f"⚠️ Could not find '{EXTRA_DATA_DIR}'. Training on Train+Val only.")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 2       
IMAGE_SIZE = 512     
EPOCHS = 20  # Reduced slightly because dataset is now bigger (takes longer per epoch)

ID_MAPPING = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 
    600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}
CLASSES = ['Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 'Ground Clutter', 
           'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky']

class DesertDataset(Dataset):
    def __init__(self, images_dir, masks_dir, augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.ids = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.augment = augment

    def __getitem__(self, i):
        img_name = self.ids[i]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Smart Mask Finding (Handles .jpg vs .png differences)
        mask_name = img_name.replace(".jpg", ".png").replace(".jpeg", ".png")
        mask_path = os.path.join(self.masks_dir, mask_name)
        if not os.path.exists(mask_path): mask_path = os.path.join(self.masks_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        
        mask = cv2.imread(mask_path, -1)
        if mask is None: mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int64)
        else: mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        # Augmentation
        if self.augment:
            if random.random() > 0.5: # Flip
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            if random.random() > 0.5: # Lighting
                alpha = random.uniform(0.8, 1.2) 
                beta = random.uniform(-20, 20) 
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        new_mask = np.full_like(mask, 8) 
        for raw_id, map_id in ID_MAPPING.items():
            new_mask[mask == raw_id] = map_id
        mask = new_mask

        image = image.astype('float32') / 255.0
        image = image.transpose(2, 0, 1)
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()

    def __len__(self): return len(self.ids)

def train_total_war():
    print(f"🚀 STARTING TOTAL WAR TRAINING on {DEVICE}...")
    
    datasets = []
    
    # 1. Add Original Train
    datasets.append(DesertDataset(
        os.path.join(DATA_ROOT, "train", "Color_Images"),
        os.path.join(DATA_ROOT, "train", "Segmentation"),
        augment=True
    ))
    
    # 2. Add Original Val (Now used for training)
    datasets.append(DesertDataset(
        os.path.join(DATA_ROOT, "val", "Color_Images"),
        os.path.join(DATA_ROOT, "val", "Segmentation"),
        augment=True
    ))
    
    # 3. Add The Extra Test Folder (If found)
    if HAS_EXTRA_DATA:
        datasets.append(DesertDataset(
            os.path.join(EXTRA_DATA_DIR, "Color_Images"), # Assuming structure matches
            os.path.join(EXTRA_DATA_DIR, "Segmentation"),
            augment=True
        ))
    
    full_dataset = ConcatDataset(datasets)
    print(f"📊 MASSIVE DATASET SIZE: {len(full_dataset)} images")
    
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', classes=len(CLASSES))
    model.to(DEVICE)
    
    # Resume from best model to save time
    if os.path.exists("best_model.pth"):
        print("🔄 Loading previous 65% model as a baseline...")
        try:
            model.load_state_dict(torch.load("best_model.pth"))
        except:
            print("⚠️ Weights mismatch. Starting fresh.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion_dice = smp.losses.DiceLoss(mode='multiclass')
    criterion_ce = torch.nn.CrossEntropyLoss()

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
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.4f}")
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
        # Save constantly
        torch.save(model.state_dict(), 'best_model.pth')
        torch.save(model.state_dict(), 'total_war_model.pth')

if __name__ == "__main__":
    train_total_war()