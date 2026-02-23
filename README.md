# 🏜️ Offroad Environment Segmentation AI

> **Production-grade semantic segmentation model for autonomous offroad navigation in desert terrain.**
>
> Built for the **Startathon Desert Hackathon** — classifies every pixel of a terrain image into one of 10 environmental categories to enable safe autonomous offroading.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![React](https://img.shields.io/badge/React-18.x-61dafb)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📸 Sample Predictions

> Each row shows: **Input Image** | **Ground Truth** | **AI Prediction** | **Overlay Image**

![sample2](https://github.com/user-attachments/assets/ee6741d8-b982-45bf-990a-2f073db7fb2b)

---

## 🧠 Model Architecture

| Component | Details |
|-----------|---------|
| **Architecture** | U-Net |
| **Encoder** | EfficientNet-B4 (ImageNet pretrained) |
| **Framework** | [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) |
| **Input Resolution** | 512 × 512 |
| **Output Classes** | 10 |
| **Loss Function** | CrossEntropy + Dice (Hybrid Wrapper) |
| **Optimizer** | AdamW (Initial LR: 5e-4, Fine-tune LR: 1e-4) |
| **Augmentation** | Albumentations (Affine, H-Flip, RandomBrightnessContrast) |

---

## 🏷️ Terrain Classes

| Class ID | Raw Pixel Value | Class Name | Legend Color |
|:--------:|:---------------:|------------|:------------:|
| 0 | 100 | Trees | 🟩 `#228B22` |
| 1 | 200 | Lush Bushes | 🟢 `#9ACD32` |
| 2 | 300 | Dry Grass | 🟨 `#DAA520` |
| 3 | 500 | Dry Bushes | 🟫 `#8B4513` |
| 4 | 550 | Ground Clutter | ⬜ `#808080` |
| 5 | 600 | Flowers | 🩷 `#FF69B4` |
| 6 | 700 | Logs | 🟤 `#A0522D` |
| 7 | 800 | Rocks | ⬛ `#696969` |
| 8 | 7100 | Landscape | 🟧 `#F4A460` |
| 9 | 10000 | Sky | 🔵 `#87CEEB` |

![terrain_class](https://github.com/user-attachments/assets/a16bf38d-9c4c-40ab-9fa2-a75f9ca3e774) 

---

## 📊 Performance

### Overall Metrics (Validation Set)

| Metric | Score |
|--------|------:|
| **Pixel Accuracy** | ~89.15% |
| **Mean IoU** | ~70% |

**🎯 Metric Selection**: Why Mean IoU and F1-Score instead of mAP50?
While mAP50 is a standard metric for *Object Detection* (counting individual bounding boxes), we architected a *Semantic Segmentation* (U-Net) model to achieve pixel-perfect terrain mapping. Semantic models classify materials, not distinct instances. For autonomous offroad navigation, knowing the exact irregular shape of a winding dirt path or a jagged rock is much more valuable than a generalized bounding box. 

### Per-Class IoU (Intersection over Union)

The Production Model achieved massive improvements in highly irregular micro-terrains (Rocks & Ground Clutter) compared to our baseline.

* **Sky**: 98.77%
* **Trees**: 88.59%
* **Lush Bushes**: 73.16%
* **Landscape**: 72.69%
* **Dry Grass**: 71.88%
* **Flowers**: 71.31%
* **Logs**: 65.82% 
* **Rocks**: 57.49% *(+10% improvement over Baseline)*
* **Dry Bushes**: 53.33%
* **Ground Clutter**: 45.78% *(+4% improvement over Baseline)*

---

## 🏋️ Training Evolution

The model was iteratively improved, ultimately transitioning to Cloud GPU infrastructure to support heavier architectures and aggressive fine-tuning:

| Phase | Backbone | Resolution | Loss | Augmentation | Key Concept |
|:-------:|------|:----------:|------|:------------:|-----------------|
| **Baseline** | ResNet-34 | 256x256 | CE | ❌ | Local GPU prototyping |
| **V1** | ResNet-34 | 512x512 | CE | ❌ | Fixed raw ID mask mapping |
| **V2** | ResNet-34 | 512x512 | CE + Dice | Basic Flips | Introduced Hybrid Loss |
| **Production** | **EfficientNet-B4** | 512x512 | CE + Dice | **Albumentations** | 16GB Cloud GPU training |
| **Fine-Tuning** | EfficientNet-B4 | 512x512 | CE + Dice | Albumentations | **Learning Rate Decay (1e-4)** |

### The "Production AI" Upgrades
- **EfficientNet-B4 Upgrade:** Swapped the legacy ResNet encoder for EfficientNet to significantly improve the AI's ability to understand chaotic, unstructured ground textures.
- **Heavy Augmentation:** Implemented `Albumentations` (Affine transforms, rotations, scale, and brightness adjustments) to force the AI to learn the geometric shapes of rocks and logs rather than just memorizing dataset lighting.
- **Learning Rate Decay (Fine-Tuning):** After converging at a learning rate of `5e-4` for 15 epochs, the model was fine-tuned for an additional 25 epochs at `1e-4` to mathematically shave down the borders of minority classes without overfitting.

---

## 📂 Project Structure

```
desert_hackathon/
├── app.py                      # Streamlit web app for live inference
├── best_efficientnet_model.pth # Production model weights (~130 MB)
├── requirements.txt            # Python dependencies
│
├── web-ui/                     # React Frontend Application
│   ├── api_server.py           # Flask/FastAPI backend serving the PyTorch model
│   ├── src/                    # React components and assets
│   └── package.json            
│
├── Offroad_Segmentation_Training_Dataset/  # Dataset (gitignored)
```

---

### 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js (For frontend)
- NVIDIA GPU with CUDA support (tested on RTX 4050 — 6 GB VRAM)

### 1)- Installation

```bash
git clone https://github.com/samarthshukla20/semantic-segmentation-ai-model.git
cd semantic-segmentation-ai-model
pip install -r requirements.txt
```

### 2)- Frontend Installation

```bash
cd web-ui
npm install
cd ..
```

## 🧪 Evaluation

```bash
# Quick visual check on a random validation image
python check_model.py

# Accurate visual check with correct mask reading
python accurate_check.py

# Per-class IoU on the full validation set
python check_iou.py

# Full evaluation: IoU + confusion matrix + 5 visual results
python final_test.py

# Verify train/val split ratio
python check_split.py

# (Optional, requires GPU) Generate bar charts & sample prediction grids
python generate_readme_assets.py
```

---

## 🌐 Web App

(Option 1) Launch the modern react web app:
You will need two terminal windows to run the full stack.

#Terminal 1 (Backend API):
```bash
cd web-ui
python api_server.py
```

#Terminal 2 (Frontend UI):
```bash
cd web-ui
npm run dev
```

(Option 2) Launch the interactive Streamlit demo:

```bash
streamlit run app.py
```

**Features:**
- 📤 Upload any terrain image for real-time segmentation
- 🖼️ Side-by-side original vs. AI perception view
- 📈 Live confidence score (softmax-based)
- 📊 Pre-computed baseline metrics dashboard
- 🔍 Expandable detailed per-class IoU breakdown
- 🗺️ Color-coded terrain legend

---

## 🔑 Key Technical Decisions

### 1. Hybrid CrossEntropy + Dice Loss
Standard Cross-Entropy loss evaluates pixels individually, causing the AI to perform exceptionally well on massive objects (Sky: 98%) but fail completely on small hazards. By wrapping nn.CrossEntropyLoss with smp.losses.DiceLoss, the AI is mathematically penalized for missing the entire shape of thin logs or small rocks, forcing it to respect precise object boundaries.

### 2. ImageNet Pixel Normalization
During the transition to EfficientNet, the input tensors in the deployment server (api_server.py and app.py) were explicitly normalized using the ImageNet mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]. Skipping this step in production leads to catastrophic feature misalignment and garbage inference outputs.

### 3. Progressive Training
Instead of training at 512×512 from scratch (GPU memory-prohibitive at batch sizes needed), we first converge at 256×256 and then fine-tune at 512×512 with a very low learning rate. Faster convergence, lower memory usage.

### 4. Avoiding mAP50 for Semantic Validation
As U-Net handles Semantic Segmentation rather than Instance Segmentation, the model classifies amorphous material blobs rather than discrete, countable objects. Calculating mAP50 requires artificial bounding-box generation via OpenCV, which artificially punishes the score if an object (like a log) is partially buried and split into two visual pieces. Validation relies strictly on pixel-wise Mean IoU and Dice (F1) scores.

---

## 👥 Team

| Name | Role |
|------|------|
| `Dhruv Bajpai` | `Team Lead` |
| `Samarth Shukla` | `Backend` |
| `Kshitij Trivedi` | `Frontend` |

---

## 📄 License

This project was developed for the **Startathon Desert Hackathon**. Please check with the organizers for dataset licensing and usage terms.

---

## 🙏 Acknowledgments

- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) by Pavel Iakubovskii
- [Streamlit](https://streamlit.io/) for the interactive demo framework
- Hackathon organizers for the Offroad Segmentation dataset
