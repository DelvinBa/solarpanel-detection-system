#!/usr/bin/env python
# YOLO Solar Panel Detection Training Script with MLflow Integration

import os
# Set OpenMP environment variable to avoid duplicate library warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set MLflow timeout and retry configurations
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "120"  # 120 seconds timeout
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "5"  # 5 retries
os.environ["MLFLOW_CHUNK_SIZE"] = "5242880"  # 5MB for large file uploads

import yaml
import mlflow
import argparse
import torch
from ultralytics import YOLO
from pathlib import Path
import logging
import inspect
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add required safe globals for PyTorch 2.6+ security measures
try:
    from torch.serialization import add_safe_globals
    # Allow loading YOLO detection model class
    add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
    logger.info("Added safe globals for PyTorch model loading")
except ImportError:
    logger.warning("Could not import add_safe_globals, this might cause issues with model loading")
    # Try alternative by setting an environment variable for PyTorch
    os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "False"

# Patch ultralytics strip_optimizer function to handle PyTorch 2.6 changes
try:
    import ultralytics.utils.torch_utils as torch_utils
    
    # Store original strip_optimizer function
    original_strip_optimizer = torch_utils.strip_optimizer
    
    # Create patched version
    def patched_strip_optimizer(f=None, *args, **kwargs):
        try:
            return original_strip_optimizer(f, *args, **kwargs)
        except Exception as e:
            logger.warning(f"Error in strip_optimizer: {e}")
            logger.info("Using patched version of strip_optimizer")
            
            # If the function has a 'torch.load' call that's failing, we'll implement
            # a simplified version that works with PyTorch 2.6+
            try:
                # Simple version: just copy the file without stripping
                if f is not None and Path(f).exists():
                    # Get file size
                    filesize = Path(f).stat().st_size
                    logger.info(f"File size before stripping: {filesize/1e6:.2f} MB")
                    
                    # Copy file to a temp location
                    temp_file = f"{f}.temp"
                    import shutil
                    shutil.copy2(f, temp_file)
                    
                    # Return the file path
                    return f
            except Exception as inner_e:
                logger.error(f"Patched strip_optimizer also failed: {inner_e}")
                # Just return the file path
                return f
    
    # Apply the patch
    torch_utils.strip_optimizer = patched_strip_optimizer
    logger.info("Patched ultralytics.utils.torch_utils.strip_optimizer function to handle PyTorch 2.6 changes")
except Exception as patch_e:
    logger.warning(f"Failed to patch strip_optimizer: {patch_e}")
    
def setup_mlflow():
    """Configure MLflow tracking"""
    try:
        # Set up the tracking URI to use the MLflow server with the correct port
        mlflow_port = os.getenv("MLFLOW_PORT", "5001")
        tracking_uri = f"http://localhost:{mlflow_port}"
        logger.info(f"Setting MLflow tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create a new experiment
        experiment_name = "solar_panel_detection"
        print(f"Creating experiment: {experiment_name}")
        
        # Try to get existing experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"Experiment already exists with ID: {experiment_id}")
        else:
            # Create the experiment
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created experiment with ID: {experiment_id}")
        
        return experiment_id
    except Exception as e:
        print(f"Error setting up MLflow: {str(e)}")
        raise

def create_data_yaml(base_dir, output_path=None):
    """Create or update the data.yaml file for YOLO training"""
    if output_path is None:
        # Create in the models directory for better organization
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
        os.makedirs(models_dir, exist_ok=True)
        output_path = os.path.join(models_dir, 'data.yaml')
    
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

def train_model(data_yaml_path, experiment_id, model_name="yolov8n.pt", epochs=50, batch_size=16, img_size=832, **kwargs):
    """Train the YOLO model with MLflow tracking"""
    # Get models directory for storing outputs
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    
    # Start MLflow run with explicit experiment_id
    print(f"Starting MLflow run with experiment ID: {experiment_id}")
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"Started MLflow run with ID: {run_id}")
        
        # Log parameters
        params = {
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
        }
        params.update(kwargs)
        mlflow.log_params(params)
        print(f"Logged parameters: {params}")
        
        # Initialize YOLO model
        try:
            model = YOLO(model_name)
            
            # Disable MLflow in YOLO's internal settings to prevent conflicts
            try:
                # Try newer ultralytics structure first
                from ultralytics.utils.callbacks.mlflow import _log_to_mlflow
                import types
                # Replace the log function with a no-op function
                _log_to_mlflow.__code__ = (lambda *args, **kwargs: None).__code__
                logger.info("Disabled MLflow in YOLO by patching _log_to_mlflow")
            except ImportError:
                try:
                    # Fallback for older versions
                    from ultralytics.utils.settings import Settings
                    settings = Settings()
                    settings.mlflow = False
                    logger.info("Disabled MLflow in YOLO internal settings")
                except ImportError:
                    logger.warning("Could not find ultralytics MLflow settings to disable. This is normal for newer versions.")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Trying alternative loading approach...")
            
            # Alternative approach: set environment variable and try again
            os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "False"
            
            # For PyTorch 2.6+, we may need to use a context manager
            try:
                from torch.serialization import safe_globals
                with safe_globals(['ultralytics.nn.tasks.DetectionModel']):
                    model = YOLO(model_name)
                logger.info("Successfully loaded model with safe_globals context manager")
            except Exception as nested_error:
                logger.error(f"Context manager approach failed: {nested_error}")
                
                # Final approach: try to monkey patch torch.load
                try:
                    logger.info("Trying direct torch.load approach...")
                    # Store original torch.load
                    original_torch_load = torch.load
                    
                    # Define patched version that forces weights_only=False
                    def patched_torch_load(*args, **kwargs):
                        kwargs['weights_only'] = False
                        return original_torch_load(*args, **kwargs)
                    
                    # Replace torch.load temporarily
                    torch.load = patched_torch_load
                    
                    # Try to load model with patched function
                    model = YOLO(model_name)
                    
                    # Restore original torch.load
                    torch.load = original_torch_load
                    
                    logger.info("Successfully loaded model with patched torch.load")
                except Exception as final_error:
                    logger.error(f"All loading approaches failed: {final_error}")
                    raise
        
        # Train the model
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            patience=20,  # Early stopping patience
            project=models_dir,
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
            try:
                mlflow.pytorch.log_model(
                    pytorch_model=model.model,
                    artifact_path="model",
                    registered_model_name="solar_panel_yolo"
                )
                logger.info(f"Logged model to MLflow Model Registry as 'solar_panel_yolo'")
                
                # Log additional artifacts
                try:
                    mlflow.log_artifact(str(best_model_path), "best_model")
                    logger.info(f"Logged best model artifact to MLflow")
                except Exception as artifact_e:
                    logger.warning(f"Failed to log model artifacts to MLflow: {artifact_e}")
                    logger.warning("Model still saved locally at: {best_model_path}")
                
                # Log confusion matrix and other plots if they exist
                plots_dir = Path(results.save_dir) / "plots"
                if plots_dir.exists():
                    try:
                        mlflow.log_artifacts(str(plots_dir), "plots")
                        logger.info(f"Logged plots and artifacts to MLflow")
                    except Exception as plots_e:
                        logger.warning(f"Failed to log plots to MLflow: {plots_e}")
                        logger.warning(f"Plots still available locally at: {plots_dir}")
            except Exception as mlflow_e:
                logger.warning(f"Failed to log model to MLflow: {mlflow_e}")
                logger.warning(f"Model still saved locally at: {best_model_path}")
        else:
            logger.warning(f"Best model file not found at {best_model_path}")
        
        return model, run_id

def main():
    """Main function to run the training process"""
    parser = argparse.ArgumentParser(description="Train YOLO model with MLflow tracking")
    parser.add_argument("--data_dir", type=str, default="data/processed/SateliteData", help="Path to dataset directory")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model to start with")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img_size", type=int, default=832, help="Image size")
    args = parser.parse_args()
    
    # Setup MLflow
    experiment_id = setup_mlflow()
    
    # Create data.yaml
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', args.data_dir))
    print(f"Using data directory: {base_dir}")
    data_yaml_path = create_data_yaml(base_dir)
    
    # Train model - pass experiment_id explicitly
    model, run_id = train_model(
        data_yaml_path=data_yaml_path,
        experiment_id=experiment_id,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size
    )
    
    print(f"Training completed. MLflow run ID: {run_id}")
    print(f"View run details at: http://localhost:{os.getenv('MLFLOW_PORT', '5001')}/experiments")

if __name__ == "__main__":
    main() 