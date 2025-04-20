import os

# Set this to the root directory containing your label files
LABELS_DIR = 'dataset/labels'  # Adjust if your label path differs

def remap_labels(root_dir):
    count = 0
    for split in ['train', 'val', 'test']:  # Include all relevant splits
        folder = os.path.join(root_dir, split)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if file.endswith(".txt"):
                path = os.path.join(folder, file)
                with open(path, 'r') as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    label = int(parts[0])
                    if label == 2:
                        parts[0] = '0'  # scale bar becomes class 0
                    elif label == 1:
                        parts[0] = '1'  # LIPSS surface remains class 1
                    else:
                        continue  # skip if any unexpected label
                    new_lines.append(" ".join(parts) + "\n")
                with open(path, 'w') as f:
                    f.writelines(new_lines)
                count += 1
    print(f"✅ Updated labels in {count} files.")

if __name__ == "__main__":
    remap_labels(LABELS_DIR)
