FROM apache/airflow:2.7.2-python3.11

# Switch to root to install system dependencies
USER root

# Install system dependencies required for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the airflow user
USER airflow

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt