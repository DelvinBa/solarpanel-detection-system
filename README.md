# Automatic Solar Panel Detection

## 📌 Project Background

The energy transition is not just a technical challenge but a societal one. The **NOWATT** project aims to involve residents, SMEs, and government entities in accelerating energy neutrality at the neighborhood level using **Artificial Intelligence (AI)**. One critical aspect is **predicting the energy labels** of homes in the Netherlands.

A major factor influencing a home's energy efficiency is **solar panel installation**. By **detecting solar panels on rooftops**, we can improve **energy label prediction models** and help housing associations meet sustainability requirements by **2030**. Organizations like **Nijhuis Bouw** are actively working on these improvements.

This project leverages **computer vision** to detect solar panels using satellite imagery, forming a crucial step in energy label classification.

---

## 🛠️ Project Scope & Goals

In this project, we focus on:
- 📡 **Satellite Image Processing** – Detect solar panels from aerial/satellite images.
- 🏠 **Data Integration** – Align detected solar panels with **open-source** data like **Kadaster**.
- 🤖 **Object Detection Algorithms** – Experiment with **YOLO, Faster R-CNN, SSD**, etc.
- 🔄 **Automated ML Pipeline** – Develop a **data pipeline** that automates preprocessing, model training, and evaluation.
- 🚀 **API Deployment** – Convert the trained model into an **API** that accepts images and returns solar panel predictions.

You will work with:
✅ **Raw & annotated satellite image data**  
✅ **Preprocessing scripts & model training code**  
✅ **Object detection models (YOLO, etc.)**  

---

## 🏗️ Architecture

This project consists of several components:

1. **MLflow** - For model tracking, versioning, and serving
2. **Airflow** - For automated workflow orchestration
3. **MinIO** - For object storage (images and model artifacts)
4. **PostgreSQL** - For metadata storage
5. **YOLO** - For object detection

### System Architecture Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │
│   Images    │────▶│   Airflow   │────▶│  Database   │
│  (MinIO)    │     │  Workflow   │     │ (PostgreSQL)│
│             │     │             │     │             │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │                           
                           ▼                           
                    ┌─────────────┐                    
                    │             │                    
                    │  YOLO Model │                    
                    │             │                    
                    └──────┬──────┘                    
                           │                           
                           ▼                           
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │
│  MLflow     │◀────│  Processed  │────▶│  Labeled    │
│  Tracking   │     │   Results   │     │   Images    │
│             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## Prerequisites

Ensure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Python 3.10+](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/)

## ⚙️ Setup Guide

### **1️⃣ Clone the Repository**
```bash
git clone <repository-url>
cd AutomaticSolarPanelDetection
```

### **2️⃣ Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Configure Environment Variables**
The project uses environment variables for configuration. You can either:

1. Use the default values in the `.env` file
2. Create your own `.env` file based on the provided template

```bash
# Copy the template and modify as needed
cp .env.template .env
```

### **5️⃣ Prepare YOLO Model**
Ensure your YOLO model is correctly placed in the appropriate directory:

```bash
# Create directory if it doesn't exist
mkdir -p airflow/dags/models

# Place your YOLO model file (e.g., best5.pt) in this directory
# cp /path/to/your/model.pt airflow/dags/models/best5.pt
```

### **6️⃣ Start the Services**
```bash
docker-compose up -d
```

This command will start all necessary services:
- **MLflow** server on port 5000
- **Airflow** webserver on port 8080
- **MinIO** server on port 9000 (API) and 9001 (Console)
- **PostgreSQL** databases for both MLflow and Airflow

### **7️⃣ Verify Services**

- **MLflow UI**: http://localhost:5000
- **Airflow UI**: http://localhost:8080 (Username: admin, Password: admin)
- **MinIO Console**: http://localhost:9001 (Username: minioadmin, Password: minioadmin)

### **8️⃣ Set Up MinIO Bucket**

1. Open the MinIO Console: http://localhost:9001
2. Log in using the credentials (default: minioadmin/minioadmin)
3. Create a bucket called `mybucket` for storing images
4. Upload your images to this bucket

## 🚀 Usage

### Training YOLO Models with MLflow Integration

We've integrated YOLO model training with MLflow for experiment tracking and model management. This allows you to:

1. **Track Experiments**: Log parameters, metrics, and artifacts during training
2. **Compare Models**: Easily compare different model configurations
3. **Version Control**: Register and version trained models
4. **Load for Inference**: Seamlessly load models for inference

#### Training a YOLO Model

To train a YOLO model with MLflow tracking:

```bash
# Basic training with default parameters
python traintest/train_yolo.py

# Custom training with specific parameters
python traintest/train_yolo.py --model yolov8m.pt --epochs 50 --batch 16 --img_size 832
```

#### Running Inference with MLflow-tracked Models

```bash
# Using the latest model from the registry
python traintest/predict_mlflow.py --image path/to/image.jpg

# Using a specific MLflow run
python traintest/predict_mlflow.py --image path/to/image.jpg --run_id <mlflow_run_id>
```

For more details on the MLflow integration, see the [MLflow YOLO Integration Documentation](docs/mlflow_yolo_integration.qmd).

### Running the Airflow DAG

The YOLO detection pipeline runs automatically every 5 minutes through an Airflow DAG. The DAG:

1. Fetches images from the MinIO bucket
2. Processes them with the YOLO model
3. Saves the labeled images back to MinIO in the "labeled-images" folder
4. Logs detection results to the PostgreSQL database

You can manually trigger the DAG from the Airflow UI:

1. Open http://localhost:8080
2. Navigate to the DAGs page
3. Find the "yolo_minio_airflow" DAG
4. Click the "Trigger DAG" button

### Tracking Model Performance with MLflow

MLflow can be used to track model performance, versions, and experiments:

1. Open http://localhost:5000
2. View experiments, runs, and metrics
3. Compare different model versions

## 📁 Project Organization

```
├── .env                    <- Environment variables for local development
├── config.env              <- Environment variables for Docker containers
├── docker-compose.yml      <- Docker Compose configuration
├── airflow/                <- Airflow DAGs and configuration
│   └── dags/               <- Airflow DAG definitions
│       ├── models/         <- YOLO model files
│       └── process_minio_yolo.py <- Main DAG for image processing
├── mlflow/                 <- MLflow server configuration
│   └── Dockerfile          <- Dockerfile for MLflow server
├── data/                   <- Data files
│   ├── external/           <- Data from third party sources
│   ├── interim/            <- Intermediate processed data
│   ├── processed/          <- Final processed data for modeling
│   └── raw/                <- Original, immutable data
├── models/                 <- Trained and serialized models
├── notebooks/              <- Jupyter notebooks for exploration
├── traintest/              <- YOLO model training and testing
│   ├── train_yolo.py       <- MLflow-integrated YOLO training script
│   ├── predict_mlflow.py   <- Prediction script using MLflow models
│   ├── data.yaml           <- YOLO dataset configuration
│   └── yolo_SolarPanel.ipynb <- Original YOLO training notebook
├── src/                    <- Source code
│   ├── __init__.py         <- Makes src a Python module
│   ├── config.py           <- Configuration settings
│   ├── dataset.py          <- Dataset handling
│   ├── features.py         <- Feature engineering
│   └── modeling/           <- Model training and prediction
├── docs/                   <- Documentation
│   └── mlflow_yolo_integration.qmd <- MLflow integration documentation
├── requirements.txt        <- Python dependencies
└── README.md               <- Project documentation
```

## 🛠️ Configuration Options

### Environment Variables

Key environment variables that can be configured:

| Variable | Description | Default |
|----------|-------------|---------|
| PG_USER | PostgreSQL username for MLflow | mlflow |
| PG_PASSWORD | PostgreSQL password for MLflow | mlflow |
| PG_DATABASE | PostgreSQL database for MLflow | mlflow |
| MLFLOW_PORT | Port for MLflow server | 5000 |
| MLFLOW_BUCKET_NAME | MinIO bucket for MLflow artifacts | mlflow |
| MINIO_ROOT_USER | MinIO root username | minioadmin |
| MINIO_ROOT_PASSWORD | MinIO root password | minioadmin |
| MINIO_PORT | MinIO API port | 9000 |
| MINIO_CONSOLE_PORT | MinIO Console port | 9001 |

## 🧹 Cleanup

To stop and remove all containers:

```bash
docker-compose down
```

To also remove volumes (warning: this will delete all data):

```bash
docker-compose down -v
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in the repository.

## Platform-Specific Setup

### Windows
1. Install Docker Desktop for Windows
2. Make sure WSL2 is enabled (required for Docker Desktop)
3. Clone this repository
4. Open PowerShell or Command Prompt in the project directory
5. Run:
   ```powershell
   docker-compose up -d
   ```

### macOS
1. Install Docker Desktop for Mac
2. Clone this repository
3. Open Terminal in the project directory
4. Run:
   ```bash
   docker-compose up -d
   ```

### Linux
1. Install Docker and Docker Compose
2. Clone this repository
3. Open Terminal in the project directory
4. Run:
   ```bash
   docker-compose up -d
   ```

## Accessing the Services

- Airflow Web Interface: http://localhost:8080
  - Username: admin (default)
  - Password: admin (default)
- MLflow UI: http://localhost:5000
- MinIO Console: http://localhost:9001
  - Username: minioadmin (default)
  - Password: minioadmin (default)

## Project Structure

```
.
├── airflow/
│   ├── dags/              # Airflow DAG files
│   ├── Dockerfile         # Custom Airflow image
│   └── start-airflow.sh   # Airflow startup script
├── mlflow/                # MLflow configuration
├── docker-compose.yml     # Docker Compose configuration
└── .env                   # Environment variables (create from .env.example)
```

## Troubleshooting

If you encounter any issues:

1. Make sure all required ports are available
2. Check if Docker is running properly
3. Try cleaning up Docker resources:
   ```bash
   docker-compose down -v
   docker system prune -f
   ```
4. Restart Docker Desktop (Windows/macOS) or Docker service (Linux)
5. Check the logs:
   ```bash
   docker-compose logs -f
   ```

## Development

To modify the project:

1. Edit the DAG files in `airflow/dags/`
2. Rebuild the containers:
   ```bash
   docker-compose build
   docker-compose up -d
   ```

## License

[Your License Here]

