# YOLOv8 Object Detection with MinIO

This project fetches images from a MinIO bucket and runs object detection using a trained YOLOv8 model in Python.

## **Prerequisites**

Ensure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/)
- [Python 3.8+](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/)

## **Setup and Run the Project**

### **1. Clone the Repository**

```sh
git clone <repository-url>
cd <repository-folder>
```

### **2. Create and Use a Virtual Environment**

It is recommended to use a virtual environment to manage dependencies.

#### **Create a Virtual Environment**
```sh
python -m venv venv
```

#### **Activate the Virtual Environment**
- **For Linux/macOS:**
  ```sh
  source venv/bin/activate
  ```
- **For Windows:**
  ```sh
  venv\Scripts\activate
  ```

#### **Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3. Start the Required Services with Docker**

Run the following command to start MinIO and Airflow:

```sh
docker-compose up -d
```

This will start:

- **MinIO** (for object storage)
- **Apache Airflow** (if used for orchestration)
- **PostgreSQL** (for Airflow's database)

### **4. Verify MinIO is Running**

Access the MinIO web UI at:

```
http://localhost:9001
```

Login with:

- **Username:** `minioadmin`
- **Password:** `minioadmin`

Ensure an **images** bucket exists with images uploaded.

### **5. Place Python File in Local DAGs Folder**

Instead of copying the DAG file manually, you can mount a local folder in `docker-compose.yml`.

#### **Update `docker-compose.yml` to Include the DAGs Volume**

Modify the `airflow-webserver` and `airflow-scheduler` services:

```yaml
  airflow-webserver:
    volumes:
      - ./dags:/opt/airflow/dags
  
  airflow-scheduler:
    volumes:
      - ./dags:/opt/airflow/dags
```

This ensures that all DAG files placed in the `./dags` folder on your local machine will automatically be available inside the Airflow container.

#### **Place the DAG File in the Local Folder**

Move your DAG file (e.g., `my_dag.py`) into the `dags` folder in your project directory:

```sh
mkdir -p dags
mv my_dag.py dags/
```

Restart Airflow services to detect the new DAG:

```sh
docker-compose restart airflow-webserver airflow-scheduler
```

### **6. Run the YOLOv8 Detection Script**

Ensure your virtual environment is activated, then execute the Python script:

```sh
python process_minio_yolo.py
```

This will:

1. Fetch images from the `images` bucket in MinIO.
2. Run YOLOv8 object detection.
3. Display the detected objects with bounding boxes.

### **7. (Optional) Stop the Services**

To stop and remove all containers:

```sh
docker-compose down
```

## **Project Structure**

```
.
├── dags  # Folder containing Airflow DAGs
│   ├── my_dag.py  # DAG script
├── docker-compose.yml  # Docker configuration
├── process_minio_yolo.py  # Main script to run YOLO detection
├── requirements.txt  # Python dependencies
├── README.md  # This file
├── venv  # Virtual environment (if created)
└── yolo_model.pt  # Trained YOLOv8 model (ensure it's present)
```

## **Troubleshooting**

- If Airflow doesn't work, create an admin user manually:
  ```sh
  docker exec -it airflow_webserver bash
  airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin
  ```
- If MinIO isn’t accessible, check the container logs:
  ```sh
  docker logs minio
  ```

## **Credits**

This project integrates **YOLOv8** with **MinIO** for image processing and object detection.

---