# Automatic Solarpanel Detection 

## Table of Contents
- [Automatic Solarpanel Detection](#automatic-solarpanel-detection)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Setup Instructions](#setup-instructions)
    - [Cloud Deployment](#cloud-deployment)
    - [Local Setup](#local-setup)
  - [Usage Guide](#usage-guide)
    - [Portainer](#portainer)
    - [MLflow](#mlflow)
    - [Airflow](#airflow)
    - [FastAPI Gateway](#fastapi-gateway)
    - [MinIO](#minio)

## Overview
Brief description of your project, its purpose, and main capabilities.


## Prerequisites
- Docker
- Docker Compose
- Access credentials (for cloud deployment)
- IDE e.g. VSCode

## Setup Instructions

### Cloud Deployment
1. Access the cloud environment at `[CLOUD_URL]`
2. Authentication details and access methods
3. Configuration steps if any

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://gitlab.com/saxionnl/master-ict-se/dataops/2024-2025/group-02/02.git
   cd 02
   ```
   We pushed .env file to the repository, for ease of setting up the project. Normally this file should not be pushed to the repository.


2. Start the services using Docker Compose:
   ```bash
   docker-compose build
   docker-compose up -d
   ```

3. Verify that all containers are running:
   ```bash
   docker-compose ps
   ```

## Usage Guide

### Portainer
- Access: http://localhost:9000 (or cloud URL)
- Login with provided credentials
- Main functionalities:
  - Container management
  - Volume management
  - Network settings

### MLflow
- Access: http://localhost:5000 (or cloud URL)
- Purpose: Tracking experiments and managing models
- Basic operations:
  - View experiments
  - Compare runs
  - Register models

### Airflow
- Access: http://localhost:8080 (or cloud URL)
- Purpose: Managing and scheduling data workflows
- Available DAGs:
  - (List of your main DAGs and their purposes)
- How to trigger DAGs manually

### FastAPI Gateway
- Access: http://localhost:8000 (or cloud URL)
- API documentation: http://localhost:8000/docs
- Main endpoints:
  - (List of key endpoints)

### MinIO
- Access: http://localhost:9001 (or cloud URL)
- Purpose: Object storage (similar to AWS S3)
- Basic operations:
  - Creating buckets
  - Uploading files
  - Downloading files
  - Setting access policies
- Integration with other services:
  - How MLflow uses MinIO for artifact storage
  - How Airflow accesses data from MinIO


