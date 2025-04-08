#!/usr/bin/env python
# YOLO Solar Panel Detection Training Script with MLflow Integration

import os
import yaml
import mlflow
import argparse
from ultralytics import YOLO
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Configure MLflow tracking"""
    try:
        # Get MLflow configuration
        mlflow_port = os.getenv("MLFLOW_PORT", "5000")
        
        # Set up the tracking URI to use the MLflow tracking server
        tracking_uri = f"http://localhost:{mlflow_port}"
        logger.info(f"Setting MLflow tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set the experiment name
        experiment_name = "solar_panel_detection"
        logger.info(f"Setting experiment name: {experiment_name}")
        mlflow.set_experiment(experiment_name)
        
    except Exception as e:
        logger.error(f"Error setting up MLflow: {str(e)}")
        raise

def create_data_yaml(base_dir, output_path="src/traintest/data.yaml"):
    """Create or update the data.yaml file for YOLO training"""
    data_config = {
        'path': base_dir,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {
            0: 'solar_panel',
            1: 'solar_array',
            2: 'roof_array'
        },
        'nc': 3  # Number of classes
    }
    
    # Write data.yaml
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    logger.info(f"Created data configuration at {output_path}")
    return output_path

def train_model(data_yaml_path, model_name="yolov8n.pt", epochs=50, batch_size=16, img_size=832, **kwargs):
    """Train the YOLO model with MLflow tracking"""
    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run with ID: {run_id}")
        
        # Log parameters
        params = {
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
        }
        params.update(kwargs)
        mlflow.log_params(params)
        logger.info(f"Logged parameters: {params}")
        
        # Initialize YOLO model
        model = YOLO(model_name)
        
        # Train the model
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            patience=20,  # Early stopping patience
            project="solar_panel_detection",
            name=f"yolo_run_{run_id}",
            exist_ok=True,
            pretrained=True,
            **kwargs
        )
        
        # Get the best model path
        best_model_path = Path(results.save_dir) / "weights/best.pt"
        
        # Log metrics from the training results
        metrics = {}
        if hasattr(results, "results_dict"):
            # Newer versions of YOLO have a results_dict attribute
            metrics = {
                "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
                "precision": results.results_dict.get("metrics/precision(B)", 0),
                "recall": results.results_dict.get("metrics/recall(B)", 0),
                "val_box_loss": results.results_dict.get("val/box_loss", 0),
                "val_cls_loss": results.results_dict.get("val/cls_loss", 0),
                "val_dfl_loss": results.results_dict.get("val/dfl_loss", 0)
            }
        else:
            # Fallback for cases where results_dict isn't available
            metrics = {
                "final_val_box_loss": model.trainer.metrics.get('val/box_loss', 0),
                "final_val_cls_loss": model.trainer.metrics.get('val/cls_loss', 0),
                "final_val_dfl_loss": model.trainer.metrics.get('val/dfl_loss', 0),
                "final_mAP50": model.trainer.metrics.get('metrics/mAP50(B)', 0),
                "final_mAP50-95": model.trainer.metrics.get('metrics/mAP50-95(B)', 0)
            }
        
        mlflow.log_metrics(metrics)
        logger.info(f"Logged metrics: {metrics}")
        
        # Log model to MLflow
        if best_model_path.exists():
            mlflow.pytorch.log_model(
                pytorch_model=model.model,
                artifact_path="model",
                registered_model_name="solar_panel_yolo"
            )
            logger.info(f"Logged model to MLflow Model Registry as 'solar_panel_yolo'")
            
            # Log additional artifacts
            mlflow.log_artifact(str(best_model_path), "best_model")
            
            # Log confusion matrix and other plots if they exist
            plots_dir = Path(results.save_dir) / "plots"
            if plots_dir.exists():
                mlflow.log_artifacts(str(plots_dir), "plots")
                logger.info(f"Logged plots and artifacts to MLflow")
        else:
            logger.warning(f"Best model file not found at {best_model_path}")
        
        return model, run_id

def main():
    """Main function to run the training process"""
    parser = argparse.ArgumentParser(description="Train YOLO model with MLflow tracking")
    parser.add_argument("--data_dir", type=str, default="SateliteData", help="Path to dataset directory")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model to start with")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img_size", type=int, default=832, help="Image size")
    args = parser.parse_args()
    
    # Setup MLflow
    setup_mlflow()
    
    # Create data.yaml
    base_dir = os.path.join(os.path.dirname(__file__), args.data_dir)
    data_yaml_path = create_data_yaml(base_dir)
    
    # Train model
    model, run_id = train_model(
        data_yaml_path=data_yaml_path,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size
    )
    
    logger.info(f"Training completed. MLflow run ID: {run_id}")
    logger.info(f"View run details at: http://localhost:{os.getenv('MLFLOW_PORT', '5000')}/experiments")

if __name__ == "__main__":
    main() 