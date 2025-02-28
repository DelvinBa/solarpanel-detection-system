from minio import Minio
import cv2
import numpy as np
import os
from ultralytics import YOLO

from minio import Minio
import cv2
import numpy as np
import os
from ultralytics import YOLO

# ----------------------------
# CONFIGURATION
# ----------------------------
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "mybucket"
YOLO_MODEL_PATH = "models/best5.pt"  # Ensure model is stored in `models/`

# Initialize MinIO client
client = Minio(
    endpoint=MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False  # Change to True if using HTTPS
)

# Initialize YOLO model
model = YOLO(YOLO_MODEL_PATH)


# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def download_image(file_name, local_dir="temp_images/"):
    """Downloads an image from MinIO and returns the local file path."""
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    local_file_path = os.path.join(local_dir, file_name)
    client.fget_object(BUCKET_NAME, file_name, local_file_path)
    return local_file_path


def run_yolo(image):
    """Runs YOLO model on the given image and returns the detection results."""
    return model(image)


def draw_boxes(image, results):
    """Draws bounding boxes and labels on the image."""
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = result.names[int(box.cls[0].item())]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    objects = client.list_objects(BUCKET_NAME, recursive=True)

    for obj in objects:
        file_name = obj.object_name
        local_file_path = download_image(file_name)

        # Read image using OpenCV
        img = cv2.imread(local_file_path)

        if img is None:
            print(f"Failed to load image: {file_name}")
            continue

        # Run YOLO inference
        results = run_yolo(img)

        # Draw bounding boxes
        img = draw_boxes(img, results)

        # Display results
        cv2.imshow("YOLO Detection", img)
        cv2.waitKey(1000)

        # Cleanup
        os.remove(local_file_path)

    cv2.destroyAllWindows()
