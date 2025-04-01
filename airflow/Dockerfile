FROM apache/airflow:2.10.50-python3.8


USER root

# Install system dependencies if needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*



# Install specific Python packages directly
RUN pip install --no-cache-dir \
    minio \
    opencv-python \
    ultralytics \
    sqlalchemy \
    psycopg2-binary

# Copy startup script
COPY start-airflow.sh /opt/airflow/start-airflow.sh
RUN chmod +x /opt/airflow/start-airflow.sh

# Switch back to airflow user
USER airflow
# If you have additional requirements in a file, uncomment the following:
# COPY requirements.txt /tmp/requirements.txt
# RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set the default entrypoint to run Airflow
ENTRYPOINT ["/opt/airflow/start-airflow.sh"]