# Automatic Solar Panel Detection

## Table of Contents
- [Automatic Solar Panel Detection](#automatic-solar-panel-detection)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Setup Instructions](#setup-instructions)
    - [Cloud Deployment](#cloud-deployment)
    - [Local Setup](#local-setup)
  - [Usage Guide](#usage-guide)
    - [Run Data Collection](#run-data-collection)
    - [Run Training Pipeline](#run-training-pipeline)
    - [Run Inference Process](#run-inference-process)
  - [Known Issues](#known-issues)
  - [AWS Static IPs](#aws-static-ips)
  - [Accessing Services](#accessing-services)
    - [Portainer](#portainer)
    - [FastAPI Gateway](#fastapi-gateway)
    - [MinIO Storage](#minio-storage)
    - [Airflow](#airflow)

## Overview
This project automatically detects rooftop solar panels on Dutch homes using aerial imagery.

Key features:
- **Automated Data Collection**: Retrieves fresh, geo-referenced house-level images across the Netherlands
- **YOLO-based Inference Service**: Runs object detection per House ID and returns presence/absence plus confidence scores
- **Continuous MLOps Pipeline**: Retrains and redeploys the model whenever new labeled data arrive, keeping accuracy up-to-date

## Prerequisites
- Docker
- Docker Compose
- IDE (e.g., VSCode)

## Setup Instructions

### Cloud Deployment
1. Access the cloud environment at [Portainer Dashboard](https://3.88.102.215:9443/#!/auth)
2. Login with credentials: 
   - Username: `admin`
   - Password: `1qaz!QAZ123456`
3. View all services from the Portainer dashboard by clicking on the Containers tab in the left sidebar

> **Note**: Data collection has known issues in the cloud environment. See [Known Issues](#known-issues) section.

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://gitlab.com/saxionnl/master-ict-se/dataops/2024-2025/group-02/02.git
   cd 02
   ```
   > **Note**: The .env file is included in the repository for ease of setup. In a production environment, this file should not be committed to version control.

2. Start the services using Docker Compose:
   ```bash
   docker-compose up --build -d
   ```

## Usage Guide

### Run Data Collection
1. Access FastAPI Gateway:
   - URL: http://localhost:8000/docs
   - Use the Swagger UI to run the data collection endpoint
   - For collection via city code:
     - Find city codes [here](https://public.opendatasoft.com/explore/dataset/georef-netherlands-gemeente/table/?disjunctive.prov_code&disjunctive.prov_name&disjunctive.gem_code&disjunctive.gem_name&sort=year)
     - Input the Gemeentecode with prefix
   - For collection via VIDs:
     - Transform VID format from `153010000328605.0` to `0153010000328605` (add leading 0, remove decimal)

2. View collected data in MinIO:
   - URL: http://localhost:9001
   - Credentials: `minioadmin:minioadmin`
   - Navigate to the `inference-data` bucket to see scraped data

### Run Training Pipeline
1. Access Airflow Webserver:
   - URL: http://localhost:8080
   - Credentials: `admin:admin`

2. Execute `1-split_traintest` DAG:
   - This DAG splits the dataset into training and testing sets, ensuring that the model is trained on a diverse set of data.

3. Execute `2-train_yolo` DAG:
   - This DAG trains the YOLO model using the training dataset, updating the model parameters to improve accuracy.

### Run Inference Process
1. Access Airflow Webserver:
   - URL: http://localhost:8080
   - Credentials: `admin:admin`
   - Trigger the `batch_detection` DAG manually

4. View results in MinIO:
   - URL: http://localhost:9001
   - Credentials: `minioadmin:minioadmin`
   - Navigate to the `inference-data` bucket
   - Result images are in the `detection-results` folder
   - Confidence results, house IDs, and image URLs are stored in `house_id_results.csv`

## Known Issues
- **Cloud Data Collection**: The FastAPI service in the cloud environment cannot send requests to the `solarpanel_detection_service` container. It is recommended to run data collection and inference processes locally.
- **Portainer First-Time Access**: On first local access, Portainer may fail to start. Restart the container through your Docker dashboard if needed.

## AWS Static IPs
The project is deployed on AWS with the following static IP configuration:
- **Public IP**: 3.88.102.215
- **Private IP**: 172.31.21.44

## Accessing Services

### Portainer
Monitor container status through Portainer:
- **Cloud**: [https://3.88.102.215:9443](https://3.88.102.215:9443/#!/auth) (`admin:1qaz!QAZ123456`)
- **Local**: http://localhost:9443
  - First-time local setup requires creating a new user and password

### FastAPI Gateway
- URL: http://localhost:8000/docs

### MinIO Storage
- URL: http://localhost:9001
- Credentials: `minioadmin:minioadmin`

### Airflow
- URL: http://localhost:8080
- Credentials: `admin:admin`


