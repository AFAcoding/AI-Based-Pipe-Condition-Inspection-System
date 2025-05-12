# -*- coding: utf-8 -*-
"""
"""

pip install ultralytics

"""**Imports**"""

import zipfile
import os
import shutil
from sklearn.model_selection import train_test_split
from glob import glob
import cv2

from ultralytics import YOLO

"""**Load Dataset**"""

with zipfile.ZipFile("Cus_Dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("Cus_Dataset")

"""Path of Data"""

# Paths
images_path = "Cus_Dataset/Cus_Dataset/Img"
labels_path = "Cus_Dataset/Cus_Dataset/labels"

print(os.listdir(images_path))
print(os.listdir(labels_path))

image_files = [
    os.path.join(images_path, f)
    for f in os.listdir(images_path)
    if f.lower().endswith('.jpg') and not f.lower().endswith('.zip')
]

image_files = sorted(image_files)

label_files = [
    os.path.join(labels_path, f)
    for f in os.listdir(labels_path)
    if f.lower().endswith('.txt')
]

label_files = sorted(label_files)

"""## **Data preprocessing**"""

# Mapping of images to their labels (including bounding boxes)
image_label_mapping = []

# Loop through each image file
for image_file in image_files:
    label_file = None
    # Extract the image filename without the extension
    image_filename = os.path.splitext(os.path.basename(image_file))[0]  # Obtener nombre de la imagen sin extensión

    # Search for the corresponding label file by comparing base names
    for label in label_files:
        if image_filename in os.path.basename(label):
            label_file = label
            break

    bounding_boxes = []
    label = "no_blockage"  # Default label is "no_blockage"

    # If a label file exists, process its contents
    if label_file:
        with open(label_file, 'r') as f:
            # Read all lines from the label file
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    # Extract the class id and bounding box information
                    class_id, x_center, y_center, width, height = parts
                    width = float(width)
                    height = float(height)

                    # Only consider valid bounding boxes (>0 in size)
                    if width > 0 and height > 0:
                        class_id = '0'  # Force the class to 0 (representing "blockage")
                        bounding_boxes.append({
                            'x_center': float(x_center),
                            'y_center': float(y_center),
                            'width': width,
                            'height': height
                        })
            # If valid bounding boxes exist, assign the label as "blockage"
            if bounding_boxes:
                label = "blockage"

    # Add the image along with its label and bounding boxes to the mapping list
    image_label_mapping.append((image_file, label, bounding_boxes))

# Check and print the assigned labels and number of bounding boxes
for img_path, label, objects in image_label_mapping:
    print(f"{os.path.basename(img_path)} - {label} - {len(objects)} bounding boxes")

# Create directories for images and labels
os.makedirs('images', exist_ok=True)
os.makedirs('labels', exist_ok=True)

# Mapping of class labels to numeric IDs
class_map = {
    'blockage': 0,
    'no_blockage': 1  # "no_blockage" now has its associated class ID
}

# Save images and their label files in the corresponding folders
for img_path, label, objects in image_label_mapping:
    if label not in class_map or not objects:
        continue  # Skip if label is invalid or no objects are present

    # Extract the image filename without extension to use as the label filename
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join('labels', f'{img_name}.txt')

    # Write bounding box information to the label file
    with open(label_path, 'w') as label_file:
        for obj in objects:
            label_file.write(f"{class_map[label]} {obj['x_center']} {obj['y_center']} {obj['width']} {obj['height']}\n")

    # Copy the image to the 'images' folder
    shutil.copy(img_path, os.path.join('images', os.path.basename(img_path)))

# Create directories for train and test splits
os.makedirs('images/train', exist_ok=True)
os.makedirs('images/test', exist_ok=True)
os.makedirs('labels/train', exist_ok=True)
os.makedirs('labels/test', exist_ok=True)

# List all images and labels (assuming .jpg and .txt extensions)
image_paths = [os.path.join('images', f) for f in os.listdir('images') if f.endswith('.jpg')]
label_paths = [os.path.join('labels', f.split('.')[0] + '.txt') for f in os.listdir('images') if f.endswith('.jpg')]

# Split the images and labels into train and test sets (80% train, 20% test)
train_images, test_images, train_labels, test_labels = train_test_split(image_paths, label_paths, test_size=0.2, random_state=42)

# Move the training images and labels to their respective folders
for img_path, label_path in zip(train_images, train_labels):
    shutil.move(img_path, os.path.join('images/train', os.path.basename(img_path)))
    shutil.move(label_path, os.path.join('labels/train', os.path.basename(label_path)))

# Move the testing images and labels to their respective folders
for img_path, label_path in zip(test_images, test_labels):
    shutil.move(img_path, os.path.join('images/test', os.path.basename(img_path)))
    shutil.move(label_path, os.path.join('labels/test', os.path.basename(label_path)))

"""## **Model Development**"""

import yaml

# Create the data configuration file for YOLO
data_config = {
    'train': '/content/images/train',
    'val': '/content/images/test',
    'nc': 2,
    'names': ['blockage', 'no_blockage']
}

# Write the data configuration to a YAML file
with open('data.yaml', 'w') as f:
    yaml.dump(data_config, f)

"""**Training**

*   **.yaml** = from 0
*  **.pt** = pre-trained

## **YOLO V5 small**
"""

modelv5 = YOLO('yolov5s.pt')  # pre-trained YOLO
modelv5.train(data="data.yaml", epochs=30, imgsz=640, augment=True)
# - epochs: number of training epochs
# - imgsz: image size (resize input images to 640x640)
# - augment: enable data augmentation during training for better generalization

results = modelv5.predict(
    source="images/test",
    save=True,
    save_txt=True,      # Save prediction results as .txt files (YOLO format)
    save_conf=True,     # Include confidence scores in the saved labels
    conf=0.25,          # Confidence threshold to filter predictions
    iou=0.5,            # IOU threshold for non-max suppression
    verbose=True        # Print detailed results per image
)

# Quantitative evaluation on the validation/test set (mAP, precision, recall)
metrics = modelv5.val()

"""## **YOLO V5 nano**"""

modelv5n = YOLO('yolov5n.pt')  # pre-trained YOLO
modelv5n.train(data="data.yaml", epochs=30, imgsz=640, augment=True)
# - epochs: number of training epochs
# - imgsz: image size (resize input images to 640x640)
# - augment: enable data augmentation during training for better generalization

results = modelv5n.predict(
    source="images/test",
    save=True,
    save_txt=True,      # Save prediction results as .txt files (YOLO format)
    save_conf=True,     # Include confidence scores in the saved labels
    conf=0.25,          # Confidence threshold to filter predictions
    iou=0.5,            # IOU threshold for non-max suppression
    verbose=True        # Print detailed results per image
)

# Quantitative evaluation on the validation/test set (mAP, precision, recall)
metrics = modelv5.val()

"""## **YOLO V8 small**"""

model = YOLO('yolov8s.pt')  # pre-trained YOLO
model.train(data="data.yaml", epochs=30, imgsz=640, augment=True)
# - epochs: number of training epochs
# - imgsz: image size (resize input images to 640x640)
# - augment: enable data augmentation during training for better generalization

# The best model is already saved in the 'runs/train/expX/weights' folder as 'best.pt'

results = model.predict(
    source="images/test",
    save=True,
    save_txt=True,      # Save prediction results as .txt files (YOLO format)
    save_conf=True,     # Include confidence scores in the saved labels
    conf=0.25,          # Confidence threshold to filter predictions
    iou=0.5,            # IOU threshold for non-max suppression
    verbose=True        # Print detailed results per image
)

# Quantitative evaluation on the validation/test set (mAP, precision, recall)
metrics = model.val()

"""## **YOLO V8 nano**"""

modelv8n = YOLO('yolov8n.pt')# pre-trained YOLO
modelv8n.train(data="data.yaml",epochs=30, imgsz=640, augment=True)

# - epochs: number of training epochs
# - imgsz: image size (resize input images to 640x640)
# - augment: enable data augmentation during training for better generalization

results = modelv8n.predict(
    source="images/test",
    save=True,
    save_txt=True,      # Save prediction results as .txt files (YOLO format)
    save_conf=True,     # Include confidence scores in the saved labels
    conf=0.25,          # Confidence threshold to filter predictions
    iou=0.5,            # IOU threshold for non-max suppression
    verbose=True        # Print detailed results per image
)

# Quantitative evaluation on the validation/test set (mAP, precision, recall)
metrics = modelv8n.val()
