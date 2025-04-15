FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install MinIO client
RUN pip install --no-cache-dir minio

# Copy the initialization script
COPY minio-init.sh /app/
RUN chmod +x /app/minio-init.sh

# Set the entrypoint
ENTRYPOINT ["/app/minio-init.sh"] 