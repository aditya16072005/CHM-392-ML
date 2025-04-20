import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt
import os


def estimate_periodicity_and_plot(cropped_img, pixel_size_um, scale_unit):
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    # FFT computation
    fft_image = fft2(gray)
    fft_image_shifted = fftshift(fft_image)
    magnitude = np.log(np.abs(fft_image_shifted) + 1)

    # Frequency axes in µm⁻¹
    height, width = gray.shape
    freq_x = np.fft.fftshift(np.fft.fftfreq(width, d=pixel_size_um))
    freq_y = np.fft.fftshift(np.fft.fftfreq(height, d=pixel_size_um))

    # Extract central horizontal line for dominant frequency
    center = magnitude.shape[0] // 2
    spectrum = magnitude[center, magnitude.shape[1] // 2 + 1:]
    dominant_idx = np.argmax(spectrum)
    dominant_freq = freq_x[freq_x.shape[0] // 2 + 1 + dominant_idx]
    periodicity = 1 / dominant_freq if dominant_freq != 0 else float("inf")

    # Plot FFT
    plt.figure(figsize=(8, 6))
    plt.imshow(magnitude, cmap='viridis',
               extent=[freq_x.min(), freq_x.max(), freq_y.min(), freq_y.max()])
    plt.title(f"2D FFT (Periodicity ≈ {periodicity:.2f} {scale_unit})")
    plt.xlabel(f"Spatial Frequency ({scale_unit}⁻¹)")
    plt.ylabel(f"Spatial Frequency ({scale_unit}⁻¹)")
    plt.colorbar(label="Magnitude (log scale)")
    plt.tight_layout()
    plt.show()

    return periodicity


def process_image(image_path, scale_distance, scale_unit, model_path):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    results = model(image)[0]

    scale_bar_crop = None
    lipss_crop = None
    pixels_per_unit = None

    for box in results.boxes:
        cls = int(box.cls[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = image[y1:y2, x1:x2]

        if cls == 1:  # scale_bar
            scale_bar_crop = crop
            pixels_per_unit = (x2 - x1) / scale_distance
        elif cls == 0:  # lipss_surface
            lipss_crop = crop

    if lipss_crop is not None and pixels_per_unit is not None:
        periodicity = estimate_periodicity_and_plot(lipss_crop, 1 / pixels_per_unit, scale_unit)
        print(f"✅ Estimated Periodicity: {periodicity:.2f} {scale_unit}")
    else:
        print("❌ Detection failed: Scale bar or LIPSS region not found.")


# === Run Script ===
scale_distance = float(input("Enter the known scale bar length (e.g., 2.0): "))
scale_unit = input("Enter the unit (e.g., um, nm): ").strip()
image_path = input("Enter the image path: ").strip()
model_path = "runs/detect/train14/weights/best.pt"

if os.path.exists(image_path):
    process_image(image_path, scale_distance, scale_unit, model_path)
else:
    print("⚠️ Invalid image path.")
