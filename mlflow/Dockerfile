FROM python:3.10-slim

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up pip and wheel first - this is crucial
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Then install the rest of the packages
RUN pip install --no-cache-dir \
    pysftp \
    mlflow[extras]==2.8.1 \
    psycopg2-binary==2.9.9 \
    'pydantic<2.0.0' \
    boto3==1.34.63 \
    cryptography==42.0.5 \
    pymysql==1.1.0

EXPOSE 5000