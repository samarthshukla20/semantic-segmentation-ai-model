"""
DesertNav.AI — Flask API Server
Loads the trained U-Net segmentation model and serves predictions
to the React frontend via a REST endpoint.
"""

import os
import sys
import io
import base64
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# ═══════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'best_model.pth')

CLASSES = [
    'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 'Ground Clutter',
    'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

COLORS = np.array([
    [34, 139, 34],    # Trees
    [154, 205, 50],   # Lush Bushes
    [218, 165, 32],   # Dry Grass
    [139, 69, 19],    # Dry Bushes
    [128, 128, 128],  # Ground Clutter
    [255, 105, 180],  # Flowers
    [160, 82, 45],    # Logs
    [105, 105, 105],  # Rocks
    [244, 164, 96],   # Landscape
    [135, 206, 235],  # Sky
], dtype=np.uint8)

# ═══════════════════════════════════════════════════
#  LOAD MODEL
# ═══════════════════════════════════════════════════
print(f"🏜️  DesertNav API starting on {DEVICE}...")
print(f"📦  Loading model from: {os.path.abspath(MODEL_PATH)}")

model = smp.Unet(encoder_name='resnet34', encoder_weights=None, classes=len(CLASSES))

if not os.path.exists(MODEL_PATH):
    print("⚠️  best_model.pth not found! Copy it to the project root.")
    sys.exit(1)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("✅  Model loaded successfully!")

# ═══════════════════════════════════════════════════
#  FLASK APP
# ═══════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)  # Allow React dev server (localhost:5173) to call us


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'device': DEVICE,
        'model': 'U-Net (ResNet-34)',
        'classes': len(CLASSES),
    })


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # ── Read image ──
        img_bytes = file.read()
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_np = np.array(img_pil)

        # ── Preprocess ──
        input_img = cv2.resize(img_np, (256, 256))
        input_tensor = input_img.astype('float32') / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)  # HWC → CHW
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).float().to(DEVICE)

        # ── Inference ──
        with torch.no_grad():
            output = model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)

            # Confidence
            probs = torch.nn.functional.softmax(output, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            confidence = max_probs.mean().item() * 100

        # ── Colorize mask ──
        # Resize to a nice display size
        pred_resized = cv2.resize(pred_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        color_mask = COLORS[pred_resized]  # (512, 512, 3)

        # ── Encode to base64 PNG ──
        _, buffer = cv2.imencode('.png', cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        mask_b64 = base64.b64encode(buffer).decode('utf-8')

        # ── Per-class pixel percentages ──
        total_pixels = pred_resized.size
        class_stats = []
        for i, name in enumerate(CLASSES):
            count = int(np.sum(pred_resized == i))
            pct = round(count / total_pixels * 100, 2)
            if pct > 0:
                class_stats.append({
                    'name': name,
                    'percentage': pct,
                    'color': f'#{COLORS[i][0]:02x}{COLORS[i][1]:02x}{COLORS[i][2]:02x}',
                })
        class_stats.sort(key=lambda x: x['percentage'], reverse=True)

        return jsonify({
            'mask': mask_b64,
            'confidence': round(confidence, 2),
            'classes_detected': class_stats,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀  API server running at http://localhost:5050")
    app.run(host='0.0.0.0', port=5050, debug=False)
