import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = os.path.join("Offroad_Segmentation_Training_Dataset")
if not os.path.exists(os.path.join(BASE_DIR, "val")):
    BASE_DIR = os.path.join("Offroad_Segmentation_Training_Dataset", "Offroad_Segmentation_Training_Dataset")

CLASSES = ['Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 'Ground Clutter', 
           'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky']
ID_MAPPING = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 
    600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

class ValDataset(Dataset):
    def __init__(self, root_dir):
        self.images_dir = os.path.join(root_dir, "val", "Color_Images")
        self.masks_dir = os.path.join(root_dir, "val", "Segmentation")
        self.ids = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __getitem__(self, i):
        img_name = self.ids[i]
        img_path = os.path.join(self.images_dir, img_name)
        
        mask_name = img_name.replace(".jpg", ".png").replace(".jpeg", ".png")
        mask_path = os.path.join(self.masks_dir, mask_name)
        if not os.path.exists(mask_path): mask_path = os.path.join(self.masks_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        
        # READ MASK IN RAW MODE (Crucial for correct IoU)
        mask = cv2.imread(mask_path, -1)
        if mask is None: mask = np.zeros((256, 256), dtype=np.int64)
        else: mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Map to 0-9
        new_mask = np.full_like(mask, 8) 
        for raw_id, map_id in ID_MAPPING.items():
            new_mask[mask == raw_id] = map_id
        
        image = image.astype('float32') / 255.0
        image = image.transpose(2, 0, 1)
        return torch.from_numpy(image).float(), torch.from_numpy(new_mask).long()

    def __len__(self): return len(self.ids)

def calculate_iou():
    print(f"🚀 Calculating Official IoU Score on {DEVICE}...")
    
    val_dataset = ValDataset(BASE_DIR)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=0)
    
    model = smp.Unet(encoder_name='resnet34', classes=len(CLASSES))
    model.to(DEVICE)
    
    try:
        model.load_state_dict(torch.load("best_model.pth"))
    except FileNotFoundError:
        print("❌ Error: best_model.pth not found!")
        return

    model.eval()
    
    # Store Intersection and Union for each class
    intersection_meter = np.zeros(len(CLASSES))
    union_meter = np.zeros(len(CLASSES))

    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images = images.to(DEVICE)
            masks = masks.cpu().numpy() # Keep masks on CPU for numpy calculation
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Calculate IoU for this batch
            for class_id in range(len(CLASSES)):
                pred_mask = (preds == class_id)
                true_mask = (masks == class_id)
                
                intersection = np.logical_and(pred_mask, true_mask).sum()
                union = np.logical_or(pred_mask, true_mask).sum()
                
                intersection_meter[class_id] += intersection
                union_meter[class_id] += union
            
            if i % 10 == 0: print(f"Processing Batch {i}...")

    # Final Calculation
    print("\n" + "="*40)
    print("🏆 FINAL IoU SCORES PER CLASS")
    print("="*40)
    
    iou_per_class = intersection_meter / (union_meter + 1e-6) # Avoid divide by zero
    
    for i, class_name in enumerate(CLASSES):
        print(f"{class_name.ljust(15)}: {iou_per_class[i]*100:.2f}%")
        
    print("-" * 40)
    print(f"🌟 MEAN IoU (Official Score): {np.mean(iou_per_class)*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    calculate_iou()