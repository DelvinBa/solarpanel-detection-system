from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the trained YOLOv8 model
model = YOLO('best5.pt')  # Load your trained model (Replace 'best.pt' with your trained weights)

# Function to draw bounding boxes and labels on an image
def draw_boxes(image, boxes, labels, confidences):
    for box, label, confidence in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{label} {confidence:.2f}', 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                    (0, 255, 0), 2)
    return image

# Load an image
image_path = './mydata/image2.jpg'
image = cv2.imread(image_path)

# Ensure the image is loaded correctly
if image is None:
    raise ValueError(f"Error loading image: {image_path}")

# Run inference
results = model.predict(image, device='cpu', conf=0.15)

# Extract results from the first image
result = results[0]  # Access the first result in the list

# Extract bounding boxes, class labels, and confidence scores
boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
class_indices = result.boxes.cls.cpu().numpy().astype(int)  # Class indices
confidences = result.boxes.conf.cpu().numpy()  # Confidence scores

# Map class indices to class names (labels)
labels = [model.names[i] for i in class_indices]

# Draw boxes and labels on the image
labeled_image = draw_boxes(image.copy(), boxes, labels, confidences)

# Display the labeled image
plt.imshow(cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
