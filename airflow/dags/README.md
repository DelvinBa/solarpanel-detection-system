# Airflow DAGs for YOLO Model Training and Deployment

This directory contains Airflow DAGs for automating the training, evaluation, and deployment of YOLO models for solar panel detection.

## Overview

The system consists of two main DAGs:

1. **train_yolo_solar_panel_detection**: Handles training and evaluation of YOLO models, with tracking in MLflow
2. **deploy_yolo_model**: Manages deployment of models that have been promoted to production

## Prerequisites

- Docker and Docker Compose
- MLflow tracking server (configured at http://mlflow_server:5001)
- YOLO training data in the expected format
- Airflow environment (provided by the docker-compose.yml)

## Training DAG

The training DAG (`train_yolo_solar_panel_detection`) automates the following steps:

1. **Data Validation**: Ensures the required training data exists
2. **Model Training**: Trains a YOLO model using the specified parameters
3. **Model Evaluation**: Evaluates the model's performance on the validation dataset
4. **Model Registration**: Registers the model to MLflow Model Registry as staging or production

### Configuration

You can configure the training process by setting Airflow Variables:

- `yolo_epochs`: Number of training epochs (default: 5)
- `yolo_batch_size`: Batch size for training (default: 16)
- `yolo_img_size`: Image size for training (default: 832)
- `yolo_model_name`: Base model to use (default: yolov8n.pt)

### Triggering Training

To trigger a training run:

1. Navigate to the Airflow UI (http://localhost:8080)
2. Go to DAGs > train_yolo_solar_panel_detection
3. Click "Trigger DAG" button

## Deployment DAG

The deployment DAG (`deploy_yolo_model`) automates the following steps:

1. **Check for Production Models**: Looks for models that have been promoted to production
2. **Download Model**: Downloads the model artifacts from MLflow
3. **ONNX Conversion**: Converts the PyTorch model to ONNX format for faster inference
4. **Deployment**: Copies the models to the deployment directory
5. **Service Restart**: Restarts the inference service to use the new model

### Scheduling

The deployment DAG runs automatically every day at midnight, but will only deploy new models if:

1. There is at least one model in the "Production" stage
2. The latest production model has not been deployed yet

### Manual Triggering

You can also trigger the deployment DAG manually through the Airflow UI.

## Viewing MLflow Experiments

To view training results and model performance:

1. Navigate to the MLflow UI (http://localhost:5001)
2. Go to the "Experiments" tab to view training runs
3. Go to the "Models" tab to view registered models

## Advanced Configuration

For more advanced configuration, you can modify:

- The DAG files directly
- Environment variables in the docker-compose.yml file
- The MLflow tracking server configuration 