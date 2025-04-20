#!/bin/bash

# This script copies the training data to the Airflow container

# Set the destination directory in the Airflow container
DATA_DIR="/opt/airflow/mlflow/data/processed/SateliteData"

# Find the Airflow scheduler container ID
CONTAINER_ID=$(docker ps | grep "airflow-scheduler" | awk '{print $1}')

if [ -z "$CONTAINER_ID" ]; then
    echo "Error: Airflow scheduler container not found. Is Airflow running?"
    exit 1
fi

echo "Found Airflow container with ID: $CONTAINER_ID"

# Create the destination directory in the container
echo "Creating destination directory at $DATA_DIR"
docker exec $CONTAINER_ID mkdir -p $DATA_DIR/train/images $DATA_DIR/train/labels $DATA_DIR/val/images $DATA_DIR/test/images

# Check if the source data directory exists
if [ ! -d "./solarpanel_detection_service/data/processed/SateliteData" ]; then
    echo "Error: Source data directory not found. Expected at: ./solarpanel_detection_service/data/processed/SateliteData"
    exit 1
fi

# Copy the data from the source to the container
echo "Copying data to Airflow container..."
docker cp ./solarpanel_detection_service/data/processed/SateliteData/. $CONTAINER_ID:$DATA_DIR/

# Set the Airflow variable for the data directory
echo "Setting Airflow variable yolo_data_dir to $DATA_DIR"
docker exec $CONTAINER_ID airflow variables set yolo_data_dir $DATA_DIR

echo "Data copying completed!"
echo "The Airflow variable 'yolo_data_dir' has been set to '$DATA_DIR'"
echo "You can now trigger the Airflow DAG 'train_yolo_solar_panel_detection' to start the training pipeline." 