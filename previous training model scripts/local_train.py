import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset

# --- CONFIGURATION ---
# Matches your screenshot: DESERT_HACKATHON > Offroad_Segmentation_Training_Dataset > train
BASE_DIR = os.path.join("Offroad_Segmentation_Training_Dataset")

# Double check if the folder is nested (common issue)
if not os.path.exists(os.path.join(BASE_DIR, "train")):
    # Try one level deeper
    BASE_DIR = os.path.join("Offroad_Segmentation_Training_Dataset", "Offroad_Segmentation_Training_Dataset")

print(f"📂 Reading data from: {os.path.abspath(BASE_DIR)}")

# RTX 4050 Settings
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 'Ground Clutter', 
           'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 6 # Safe for 6GB VRAM
IMAGE_SIZE = 256
EPOCHS = 15

class DesertDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.images_dir = os.path.join(root_dir, split, "Color_Images")
        self.masks_dir = os.path.join(root_dir, split, "Segmentation")
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"❌ Missing: {self.images_dir}")
            
        self.ids = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"✅ Found {len(self.ids)} images for {split}")

    def __getitem__(self, i):
        img_name = self.ids[i]
        img_path = os.path.join(self.images_dir, img_name)
        mask_name = img_name.replace(".jpg", ".png").replace(".jpeg", ".png")
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        if not os.path.exists(mask_path): mask_path = os.path.join(self.masks_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        
        mask = cv2.imread(mask_path, 0)
        if mask is None: mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        else: mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        image = image.astype('float32') / 255.0
        image = image.transpose(2, 0, 1)
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()

    def __len__(self): return len(self.ids)

def train():
    print(f"🚀 Training on {DEVICE}...")
    train_dataset = DesertDataset(BASE_DIR, split="train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=len(CLASSES))
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()

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
        
        print(f"✅ Epoch {epoch+1} Complete. Avg Loss: {total_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), 'best_model.pth')

if __name__ == "__main__":
    train()