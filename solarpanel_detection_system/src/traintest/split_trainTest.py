# !mkdir -p dataset/train/images dataset/train/labels
# !mkdir -p dataset/val/images dataset/val/labels
# !mkdir -p dataset/test/images dataset/test/labels

import os
import shutil
import random

base_path = os.path.join(os.path.dirname(__file__), 'SateliteData')
print(base_path)
labels_hd_path = os.path.join(base_path, 'labels', 'labels_hd')
labels_native_path = os.path.join(base_path, 'labels', 'labels_native')
images_hd_path = os.path.join(base_path, 'Maxar_HD_and_Native_Solar_Panel_Image_Chips', 'image_chips', 'image_chips_hd')
images_native_path = os.path.join(base_path, 'Maxar_HD_and_Native_Solar_Panel_Image_Chips', 'image_chips', 'image_chips_native')

train_images_path = os.path.join(base_path, 'train', 'images')
train_labels_path = os.path.join(base_path, 'train', 'labels')
val_images_path = os.path.join(base_path, 'val', 'images')
val_labels_path = os.path.join(base_path, 'val', 'labels')
test_images_path = os.path.join(base_path, 'test', 'images')
test_labels_path = os.path.join(base_path, 'test', 'labels')

# # list of all image files
image_files = [f for f in os.listdir(images_hd_path) if f.endswith('.tif')]
#print(image_files)

# # Shuffle to randomize the dataset
random.shuffle(image_files)

train_split = int(0.8 * len(image_files))
val_split = int(0.9 * len(image_files))

train_files = image_files[:train_split]
val_files = image_files[train_split:val_split]
test_files = image_files[val_split:]

def move_files(file_list, src_image_path, src_label_path, dest_image_path, dest_label_path):
    for file_name in file_list:
        shutil.move(os.path.join(src_image_path, file_name), os.path.join(dest_image_path, file_name))
        
        try:
            label_name = file_name.replace('.tif', '.txt')
            shutil.move(os.path.join(src_label_path, label_name), os.path.join(dest_label_path, label_name))
        except Exception as e:
            print(f"Error moving {label_name}: {e}")

# # Move the files to the new structure
move_files(train_files, images_hd_path, labels_hd_path, train_images_path, train_labels_path)
move_files(val_files, images_hd_path, labels_hd_path, val_images_path, val_labels_path)
move_files(test_files, images_hd_path, labels_hd_path, test_images_path, test_labels_path)
