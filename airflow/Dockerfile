# Use the official Airflow image with Python 3.7
FROM apache/airflow:2.7.3-python3.7

# Switch to root user to install system dependencies
USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy startup script
COPY start-airflow.sh /opt/airflow/start-airflow.sh
RUN chmod +x /opt/airflow/start-airflow.sh

# Switch back to airflow user
USER airflow

# Install Python dependencies
RUN pip install --no-cache-dir \
    minio==7.2.3 \
    opencv-python==4.11.0.86 \
    ultralytics==8.1.32

# Set the default entrypoint to run Airflow
ENTRYPOINT ["/opt/airflow/start-airflow.sh"]
