#!/usr/bin/env python
# YOLO Solar Panel Detection Prediction Script using MLflow models

import os
import cv2
import mlflow
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Configure MLflow tracking"""
    try:
        # Get MLflow configuration
        mlflow_port = os.getenv("MLFLOW_PORT", "5001")
        
        # Set up the tracking URI to use the MLflow tracking server
        tracking_uri = f"http://localhost:{mlflow_port}"
        logger.info(f"Setting MLflow tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set the experiment name
        experiment_name = "solar_panel_detection"
        mlflow.set_experiment(experiment_name)
        
    except Exception as e:
        logger.error(f"Error setting up MLflow: {str(e)}")
        raise

def get_latest_model_version(model_name="solar_panel_yolo"):
    """Get the latest model from MLflow registry"""
    client = mlflow.tracking.MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    
    if not model_versions:
        logger.error(f"No model versions found for {model_name}")
        return None
    
    # Get latest model version
    latest_version = max([int(mv.version) for mv in model_versions])
    logger.info(f"Using model {model_name} version {latest_version}")
    
    return f"models:/{model_name}/{latest_version}"

def load_model(model_uri=None, run_id=None, local_path=None):
    """Load a YOLO model from MLflow or local path"""
    from ultralytics import YOLO
    
    if model_uri:
        logger.info(f"Loading model from MLflow registry: {model_uri}")
        try:
            # Try to load directly if it's a YOLO compatible model
            yolo_model = mlflow.pytorch.load_model(model_uri)
            return yolo_model
        except Exception as e:
            logger.warning(f"Failed to load as PyTorch model: {e}")
            # Fall back to downloading artifacts and loading as YOLO model
            client = mlflow.tracking.MlflowClient()
            model_versions = client.search_model_versions(f"name='solar_panel_yolo'")
            run_id = model_versions[0].run_id
            return load_model(run_id=run_id)
    
    elif run_id:
        logger.info(f"Loading model from MLflow run: {run_id}")
        try:
            # Try to get best model from artifacts
            model_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="best_model/best.pt"
            )
            return YOLO(model_path)
        except Exception as e:
            logger.error(f"Failed to load model from run {run_id}: {e}")
            raise
    
    elif local_path:
        logger.info(f"Loading model from local path: {local_path}")
        return YOLO(local_path)
    
    else:
        raise ValueError("Either model_uri, run_id, or local_path must be provided")

def draw_boxes(image, boxes, labels, confidences):
    """Draw bounding boxes and labels on an image"""
    for box, label, confidence in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{label} {confidence:.2f}', 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                    (0, 255, 0), 2)
    return image

def predict_image(model, image_path, conf_threshold=0.25, save_dir=None):
    """Run inference on an image and visualize results"""
    # Load image
    logger.info(f"Loading image from {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    # Run inference
    logger.info("Running inference...")
    results = model.predict(image, conf=conf_threshold)
    
    # Extract result from the first image
    result = results[0]
    
    # Extract boxes, class labels, and confidence scores
    boxes = result.boxes.xyxy.cpu().numpy()
    class_indices = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    
    # Map class indices to class names
    class_names = {0: 'solar_panel', 1: 'solar_array', 2: 'roof_array'}
    labels = [class_names.get(i, f"class_{i}") for i in class_indices]
    
    # Draw boxes on the image
    labeled_image = draw_boxes(image.copy(), boxes, labels, confidences)
    
    # Convert to RGB for matplotlib
    labeled_image_rgb = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
    
    # Create a figure
    plt.figure(figsize=(12, 8))
    plt.imshow(labeled_image_rgb)
    plt.axis('off')
    
    # Save the image if a directory is provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        output_path = save_dir / f"prediction_{Path(image_path).stem}.jpg"
        plt.savefig(output_path)
        logger.info(f"Saved prediction to {output_path}")
    
    # Display the image
    plt.tight_layout()
    plt.show()
    
    # Return detection data
    return {
        'boxes': boxes,
        'classes': class_indices,
        'labels': labels,
        'confidences': confidences
    }

def main():
    """Main function to run inference"""
    parser = argparse.ArgumentParser(description="YOLO Solar Panel Detection Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--run_id", type=str, help="MLflow run ID to load model from")
    parser.add_argument("--local_model", type=str, help="Path to local YOLO model file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output_dir", type=str, default="reports/predictions", help="Directory to save predictions")
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup MLflow
    setup_mlflow()
    
    # Get model
    if args.run_id:
        model = load_model(run_id=args.run_id)
    elif args.local_model:
        model = load_model(local_path=args.local_model)
    else:
        # Get latest model from registry
        model_uri = get_latest_model_version()
        if model_uri:
            model = load_model(model_uri=model_uri)
        else:
            # Try to find a model in the models directory
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
            if os.path.exists(models_dir):
                # Look for best.pt in any subdirectory
                model_files = list(Path(models_dir).glob('**/weights/best.pt'))
                if model_files:
                    logger.info(f"Loading most recent local model: {model_files[0]}")
                    model = load_model(local_path=str(model_files[0]))
                else:
                    raise ValueError("No models found. Train a model first or provide a run_id or local_model.")
            else:
                raise ValueError("No model available in MLflow registry or locally. Train a model first.")
    
    # Run prediction
    results = predict_image(
        model=model,
        image_path=args.image,
        conf_threshold=args.conf,
        save_dir=output_dir
    )
    
    # Print summary of detections
    print("\nDetection Summary:")
    for label, conf in zip(results['labels'], results['confidences']):
        print(f"  - {label}: confidence {conf:.2f}")

if __name__ == "__main__":
    main() 