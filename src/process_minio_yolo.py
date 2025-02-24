from minio import Minio
import cv2
import numpy as np
import os
from ultralytics import YOLO

# MinIO connection details
MINIO_ENDPOINT = "http://localhost:9000"  # Update if running remotely
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "mybucket"

# YOLO model path (update with your model file path)
YOLO_MODEL_PATH = "best5.pt"

# Initialize MinIO client
client = Minio(
    endpoint=MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False  # Set True if using HTTPS
)

# Initialize YOLO model
model = YOLO(YOLO_MODEL_PATH)

# List objects in the MinIO bucket
objects = client.list_objects(BUCKET_NAME, recursive=True)

for obj in objects:
    file_name = obj.object_name
    local_file_path = f"temp_{file_name}"  # Temporary storage for the image

    # Download image from MinIO
    client.fget_object(BUCKET_NAME, file_name, local_file_path)

    # Read image using OpenCV
    img = cv2.imread(local_file_path)

    if img is None:
        print(f"Failed to load image: {file_name}")
        continue

    # Run YOLO inference
    results = model(img)

    # Draw results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = result.names[int(box.cls[0].item())]

            # Draw bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with detections
    cv2.imshow("YOLO Detection", img)
    cv2.waitKey(1000)  # Display each image for 1 second

    # Cleanup
    os.remove(local_file_path)

cv2.destroyAllWindows()
