FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages with retry logic
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wheel setuptools

# Install packages with retry logic (simplified approach)
RUN for i in {1..3}; do \
    pip install --no-cache-dir \
    mlflow[extras]==2.8.1 \
    psycopg2-binary==2.9.9 \
    'pydantic<2.0.0' \
    boto3==1.34.63 \
    cryptography==42.0.5 \
    pymysql==1.1.0 && break; \
    if [ $i -lt 3 ]; then \
    echo "Attempt $i failed, retrying..." && \
    sleep 5; \
    else \
    echo "All attempts failed"; \
    exit 1; \
    fi; \
    done

EXPOSE 5000