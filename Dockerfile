# Use multi-stage build for optimization
FROM apache/airflow:2.10.5 as builder

USER root

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages with caching and increased timeout
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 1000 -r requirements.txt || \
    (pip install --no-cache-dir --timeout 1000 torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
     pip install --no-cache-dir --timeout 1000 -r requirements.txt)

# Final stage
FROM apache/airflow:2.10.5

USER root

# Copy system dependencies from builder
COPY --from=builder /usr/lib/x86_64-linux-gnu/libglib-2.0.so* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libglib-2.0.so* /usr/lib/x86_64-linux-gnu/

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create necessary directories with proper permissions
RUN mkdir -p /opt/airflow/dags /opt/airflow/logs /opt/airflow/config /opt/airflow/plugins \
    && chown -R airflow:root /opt/airflow \
    && chmod -R 775 /opt/airflow

# Switch back to airflow user
USER airflow

# Copy application code, excluding problematic directories
COPY --chown=airflow:root src/ /opt/airflow/dags/ 