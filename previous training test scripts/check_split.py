import os

# Define your paths (adjust if your folder name is different)
BASE_DIR = os.path.join("Offroad_Segmentation_Training_Dataset", "Offroad_Segmentation_Training_Dataset")
if not os.path.exists(BASE_DIR):
    BASE_DIR = "Offroad_Segmentation_Training_Dataset"

train_dir = os.path.join(BASE_DIR, "train", "Color_Images")
val_dir = os.path.join(BASE_DIR, "val", "Color_Images")

def count_files(directory):
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Get counts
num_train = count_files(train_dir)
num_val = count_files(val_dir)
total = num_train + num_val

if total == 0:
    print("❌ No images found! Check your paths.")
else:
    print(f"--- DATASET SPLIT REPORT ---")
    print(f"Total Images: {total}")
    print(f"Train Images: {num_train} ({num_train/total*100:.1f}%)")
    print(f"Val Images:   {num_val}   ({num_val/total*100:.1f}%)")
    print("----------------------------")
    
    # Verdict
    if 75 < (num_train/total*100) < 85:
        print("✅ Split looks perfect (approx 80/20).")
    else:
        print("⚠️ Split might be unusual. Usually we want ~80% in Train.")