# Use the official Airflow image as your base
FROM apache/airflow:2.10.5

USER root
# Install any system dependencies you need
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch to airflow user to install Python packages into Airflow's environment
USER airflow

# (Optional) Copy your requirements.txt if you want to install from there
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Switch back to root only if you need to adjust folder permissions
USER root
RUN mkdir -p /opt/airflow/dags /opt/airflow/logs /opt/airflow/config /opt/airflow/plugins \
    && chown -R airflow:root /opt/airflow \
    && chmod -R 775 /opt/airflow

# Finally, switch back to airflow user and copy your DAGs/code
USER airflow
COPY --chown=airflow:root solarpanel_detection_system/ /opt/airflow/dags/
