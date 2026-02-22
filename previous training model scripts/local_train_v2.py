import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset

# --- CONFIGURATION ---
BASE_DIR = os.path.join("Offroad_Segmentation_Training_Dataset")
if not os.path.exists(os.path.join(BASE_DIR, "train")):
    BASE_DIR = os.path.join("Offroad_Segmentation_Training_Dataset", "Offroad_Segmentation_Training_Dataset")

print(f"📂 Reading data from: {os.path.abspath(BASE_DIR)}")

# --- HACKATHON CLASS MAPPING ---
# Based on the PDF and common dataset errors
# We map the specific ID found in the image to a clean index (0, 1, 2...)
ID_MAPPING = {
    100: 0,   # Trees
    200: 1,   # Lush Bushes
    300: 2,   # Dry Grass
    500: 3,   # Dry Bushes
    550: 4,   # Ground Clutter
    600: 5,   # Flowers
    700: 6,   # Logs
    800: 7,   # Rocks
    7100: 8,  # Landscape
    10000: 9  # Sky
}
# Fallback for weird values (like 39) due to grayscale conversion
# We will treat unknown values as "Landscape" (Class 8) to prevent crashes
DEFAULT_CLASS = 8 

CLASSES = ['Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 'Ground Clutter', 
           'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky']

# FORCE GPU CHECK
if torch.cuda.is_available():
    DEVICE = 'cuda'
    print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = 'cpu'
    print("⚠️ GPU NOT DETECTED. Training will be slow.")

BATCH_SIZE = 6
IMAGE_SIZE = 256
EPOCHS = 15

class DesertDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.images_dir = os.path.join(root_dir, split, "Color_Images")
        self.masks_dir = os.path.join(root_dir, split, "Segmentation")
        self.ids = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __getitem__(self, i):
        img_name = self.ids[i]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Mask matching logic
        mask_name = img_name.replace(".jpg", ".png").replace(".jpeg", ".png")
        mask_path = os.path.join(self.masks_dir, mask_name)
        if not os.path.exists(mask_path): mask_path = os.path.join(self.masks_dir, img_name)

        # Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Load Mask (Read UNCHANGED to get raw IDs like 100, 200)
        mask = cv2.imread(mask_path, -1) 
        if mask is None:
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int64)
        else:
            mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        # --- THE FIX: MAP VALUES ---
        # Create a new mask with valid 0-9 IDs
        new_mask = np.full_like(mask, DEFAULT_CLASS) # Fill with default first
        for raw_id, map_id in ID_MAPPING.items():
            new_mask[mask == raw_id] = map_id
        
        mask = new_mask
        # ---------------------------

        image = image.astype('float32') / 255.0
        image = image.transpose(2, 0, 1)
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()

    def __len__(self): return len(self.ids)

def train():
    train_dataset = DesertDataset(BASE_DIR, split="train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', classes=len(CLASSES))
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()

    print("🔥 Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if i % 10 == 0: print(f"Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.4f}")
        
        print(f"✅ Epoch {epoch+1} Done. Avg Loss: {total_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), 'best_model.pth')

if __name__ == "__main__":
    train()