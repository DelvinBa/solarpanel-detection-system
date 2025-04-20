# !mkdir -p dataset/train/images dataset/train/labels
# !mkdir -p dataset/val/images dataset/val/labels
# !mkdir -p dataset/test/images dataset/test/labels

import os
import shutil
import random

# Set base path to a proper data directory structure following ML project best practices
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
data_path = os.path.join(base_path, 'processed', 'SateliteData')
print(f"Setting up data in: {data_path}")

# Create structured directories
os.makedirs(data_path, exist_ok=True)

# Source directories
raw_data_path = os.path.join(base_path, 'raw')
os.makedirs(raw_data_path, exist_ok=True)

# Image and label source paths
labels_hd_path = os.path.join(raw_data_path, 'labels_hd')
labels_native_path = os.path.join(raw_data_path, 'labels_native')
images_hd_path = os.path.join(raw_data_path, 'image_chips_hd')
images_native_path = os.path.join(raw_data_path, 'image_chips_native')

os.makedirs(labels_hd_path, exist_ok=True)
os.makedirs(labels_native_path, exist_ok=True)
os.makedirs(images_hd_path, exist_ok=True)
os.makedirs(images_native_path, exist_ok=True)

# Training split directories
train_images_path = os.path.join(data_path, 'train', 'images')
train_labels_path = os.path.join(data_path, 'train', 'labels')
val_images_path = os.path.join(data_path, 'val', 'images')
val_labels_path = os.path.join(data_path, 'val', 'labels')
test_images_path = os.path.join(data_path, 'test', 'images')
test_labels_path = os.path.join(data_path, 'test', 'labels')

os.makedirs(train_images_path, exist_ok=True)
os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)
os.makedirs(test_images_path, exist_ok=True)
os.makedirs(test_labels_path, exist_ok=True)

print(f"Directory structure created at {data_path}")
print(f"Please place your source images in: {images_hd_path}")
print(f"Please place your source labels in: {labels_hd_path}")

# Check if source directories have data
if not os.path.exists(images_hd_path) or len(os.listdir(images_hd_path)) == 0:
    print(f"Warning: No images found in {images_hd_path}")
    print("Please add your satellite images to this directory and run this script again.")
    exit(1)

# List all image files
image_files = [f for f in os.listdir(images_hd_path) if f.endswith('.tif')]
print(f"Found {len(image_files)} images")

# Shuffle to randomize the dataset
random.shuffle(image_files)

train_split = int(0.8 * len(image_files))
val_split = int(0.9 * len(image_files))

train_files = image_files[:train_split]
val_files = image_files[train_split:val_split]
test_files = image_files[val_split:]

def copy_files(file_list, src_image_path, src_label_path, dest_image_path, dest_label_path):
    for file_name in file_list:
        shutil.copy(os.path.join(src_image_path, file_name), os.path.join(dest_image_path, file_name))
        
        try:
            label_name = file_name.replace('.tif', '.txt')
            shutil.copy(os.path.join(src_label_path, label_name), os.path.join(dest_label_path, label_name))
            print(f"Copied {file_name} and its label")
        except Exception as e:
            print(f"Error copying {label_name}: {e}")

# Copy the files to the new structure
print("Copying training files...")
copy_files(train_files, images_hd_path, labels_hd_path, train_images_path, train_labels_path)
print("Copying validation files...")
copy_files(val_files, images_hd_path, labels_hd_path, val_images_path, val_labels_path)
print("Copying test files...")
copy_files(test_files, images_hd_path, labels_hd_path, test_images_path, test_labels_path)
print("Data preparation complete!")
