import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- CONFIGURATION ---
print("🚀 Setting up Final Test...")

# Find the dataset again
BASE_DIR = None
search_paths = [".", "Offroad_Segmentation_Training_Dataset", os.path.join("Offroad_Segmentation_Training_Dataset", "Offroad_Segmentation_Training_Dataset")]
for p in search_paths:
    if os.path.exists(os.path.join(p, "val", "Color_Images")):
        BASE_DIR = p
        break

if BASE_DIR is None:
    print("❌ Error: Could not find dataset folder.")
    exit()

OUTPUT_DIR = "final_submission_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = ['Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 'Ground Clutter', 
           'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky']
ID_MAPPING = {100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 512

class TestDataset(Dataset):
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
        original_img = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE)) # Keep for visualization
        
        tensor_img = original_img.astype('float32') / 255.0
        tensor_img = tensor_img.transpose(2, 0, 1)

        mask = cv2.imread(mask_path, -1)
        if mask is None: mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int64)
        else: mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        new_mask = np.full_like(mask, 8) 
        for raw_id, map_id in ID_MAPPING.items():
            new_mask[mask == raw_id] = map_id
            
        return torch.from_numpy(tensor_img).float(), torch.from_numpy(new_mask).long(), original_img

    def __len__(self): return len(self.ids)

def evaluate_final():
    print(f"📊 Running Evaluation on {DEVICE}...")
    
    model = smp.Unet(encoder_name='resnet34', classes=len(CLASSES))
    model.to(DEVICE)
    
    try:
        model.load_state_dict(torch.load("best_model.pth"))
        print("✅ Loaded best_model.pth (The Total War Model)")
    except FileNotFoundError:
        print("❌ Error: Train the model first!")
        return

    model.eval()
    test_loader = DataLoader(TestDataset(BASE_DIR), batch_size=1, shuffle=False)

    intersection_meter = np.zeros(len(CLASSES))
    union_meter = np.zeros(len(CLASSES))
    
    y_true_all = []
    y_pred_all = []

    print("📸 Generating proof images...")
    with torch.no_grad():
        for i, (images, masks, original_img) in enumerate(test_loader):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks = masks.numpy()
            
            for class_id in range(len(CLASSES)):
                pred_mask = (preds == class_id)
                true_mask = (masks == class_id)
                intersection_meter[class_id] += np.logical_and(pred_mask, true_mask).sum()
                union_meter[class_id] += np.logical_or(pred_mask, true_mask).sum()

            # Save visuals for the first 5 images
            if i < 5:
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(original_img[0].numpy().astype('uint8'))
                ax[0].set_title("Input Image")
                ax[1].imshow(masks[0], cmap='jet', vmin=0, vmax=9)
                ax[1].set_title("True Mask")
                ax[2].imshow(preds[0], cmap='jet', vmin=0, vmax=9)
                ax[2].set_title("AI Prediction")
                for a in ax: a.axis('off')
                plt.savefig(f"{OUTPUT_DIR}/result_{i}.png")
                plt.close()

            # For Confusion Matrix (Sample every 100th pixel for speed)
            if i % 10 == 0: 
                y_true_all.extend(masks.flatten()[::100])
                y_pred_all.extend(preds.flatten()[::100])

    # --- FINAL REPORT ---
    iou_per_class = intersection_meter / (union_meter + 1e-6)
    
    print("\n" + "="*40)
    print("🏆 FINAL SUBMISSION METRICS")
    print("="*40)
    for i, class_name in enumerate(CLASSES):
        print(f"{class_name.ljust(15)}: {iou_per_class[i]*100:.2f}%")
        
    print("-" * 40)
    print(f"🌟 MEAN IoU: {np.mean(iou_per_class)*100:.2f}%")
    print("="*40)
    print(f"✅ Visuals saved to: {OUTPUT_DIR}")

    # Generate Confusion Matrix
    print("📊 Generating Confusion Matrix...")
    cm = confusion_matrix(y_true_all, y_pred_all, labels=range(len(CLASSES)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=CLASSES, yticklabels=CLASSES, annot=False, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_final()