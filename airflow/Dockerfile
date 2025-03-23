# Use the official Airflow image with Python 3.11
FROM apache/airflow:2.7.3-python3.11

# Switch to root user to install system dependencies
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

# Switch back to airflow user
USER airflow

# Copy Python dependencies
COPY requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt
# After your main pip install
RUN pip install --no-cache-dir minio==7.2.3
# Set the default entrypoint to run Airflow
ENTRYPOINT ["airflow"]
