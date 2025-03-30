# syntax=docker/dockerfile:1
FROM apache/airflow:2.10.5 as builder

USER root
# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment and set PATH
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final stage: use the base Airflow image
FROM apache/airflow:2.10.5
USER root
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create necessary directories with proper permissions
RUN mkdir -p /opt/airflow/dags /opt/airflow/logs /opt/airflow/config /opt/airflow/plugins \
    && chown -R airflow:root /opt/airflow \
    && chmod -R 775 /opt/airflow

# Switch back to airflow user and copy application code
USER airflow
COPY --chown=airflow:root src/ /opt/airflow/dags/