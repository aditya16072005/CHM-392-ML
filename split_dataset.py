import os
import random
import shutil

# Set seed for reproducibility
random.seed(42)

# Paths
DATASET_DIR = "dataset"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")
OUTPUT_IMAGE_DIR = os.path.join(IMAGE_DIR)
OUTPUT_LABEL_DIR = os.path.join(LABEL_DIR)

# Create train/val directories if not exist
for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_IMAGE_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_LABEL_DIR, split), exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png', '.jpeg', '.tif'))]
random.shuffle(image_files)

# Split 80-20
split_idx = int(len(image_files) * 0.8)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# Move function
def move_files(file_list, split):
    for img in file_list:
        label = os.path.splitext(img)[0] + ".txt"
        # Move image
        img_src = os.path.join(IMAGE_DIR, img)
        img_dst = os.path.join(OUTPUT_IMAGE_DIR, split, img)
        if os.path.exists(img_src):
            shutil.move(img_src, img_dst)
        # Move label
        label_src = os.path.join(LABEL_DIR, label)
        label_dst = os.path.join(OUTPUT_LABEL_DIR, split, label)
        if os.path.exists(label_src):
            shutil.move(label_src, label_dst)

# Move the files
move_files(train_files, "train")
move_files(val_files, "val")

print(f"✅ Moved {len(train_files)} images to train and {len(val_files)} to val.")
