# main.py
# Inference pipeline to detect scale bar & LIPSS region, calculate FFT, and show periodicity

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from utils.fft_periodicity import estimate_periodicity_and_plot

# ===== USER INPUT =====
REAL_SCALE_VALUE = float(input("Enter the real-world scale value (e.g., 50 for 50 um): "))
REAL_SCALE_UNIT = input("Enter the unit (e.g., um, nm): ").strip()

# ===== LOAD MODEL =====
model = YOLO("runs/detect/train/weights/best.pt")  # Trained YOLOv8 model

# ===== IMAGE PROCESSING FUNCTION =====
def process_image(image_path):
    results = model(image_path)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    img = cv2.imread(image_path)

    scale_bar_box = None
    lipss_box = None

    for cls_id, box in zip(classes, boxes):
        x1, y1, x2, y2 = map(int, box)
        if cls_id == 0:  # scale_bar
            scale_bar_box = (x1, y1, x2, y2)
        elif cls_id == 1:  # lipss_surface
            lipss_box = (x1, y1, x2, y2)

    if scale_bar_box is None or lipss_box is None:
        print(f"❌ Could not detect both scale bar and LIPSS region in {image_path}")
        return

    # Calculate pixel size from scale bar
    scale_width_px = scale_bar_box[2] - scale_bar_box[0]
    pixel_size = REAL_SCALE_VALUE / scale_width_px  # e.g., 50 um / 100 px = 0.5 um/px
    print(f"📏 {REAL_SCALE_VALUE} {REAL_SCALE_UNIT} = {scale_width_px} px → Pixel size = {pixel_size:.3f} {REAL_SCALE_UNIT}/px")

    # Crop LIPSS region
    lx1, ly1, lx2, ly2 = lipss_box
    lipss_crop = img[ly1:ly2, lx1:lx2]

    # Estimate periodicity and plot FFT
    periodicity = estimate_periodicity_and_plot(lipss_crop, 1/pixel_size, REAL_SCALE_UNIT)
    print(f"🔎 Estimated Periodicity: {periodicity:.3f} {REAL_SCALE_UNIT}")

# ===== BATCH INFERENCE ON DIRECTORY =====
folder_path = "dataset/images"
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".png", ".tif")):
        print(f"\n🖼 Processing: {filename}")
        process_image(os.path.join(folder_path, filename))
