# AWS Deployment Environment Configuration

When deploying the solar panel detection system to AWS, ensure the following environment variables are properly configured:

## Essential Airflow Configuration

```bash
# Airflow security keys
AIRFLOW__CORE__FERNET_KEY=k8IfvPBpKOoDZSBbqHOQCya9gshVXcZ5-apiPfVHfB8=
AIRFLOW__WEBSERVER__SECRET_KEY=a-very-secret-key-that-should-be-changed-in-production
```

### About Airflow Secret Keys

- **AIRFLOW__WEBSERVER__SECRET_KEY**: This is used for signing session cookies and CSRF tokens. It **MUST** be the same across all Airflow components (webserver, scheduler, worker, triggerer) to prevent 403 FORBIDDEN errors when accessing logs. Generate a strong random key for production.

- **AIRFLOW__CORE__FERNET_KEY**: Used to encrypt passwords in the connection and variable configurations. Generate a secure key using:
  ```
  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
  ```

```bash
# Airflow database connection
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://postgres:postgres@postgres/postgres
```

## Service Communication in Cloud Environments

When deploying microservices to AWS, service-to-service communication should follow these best practices:

### 1. Environment-based Service Discovery

Service URLs should be configured via environment variables instead of hardcoded values:

```bash
# For FastAPI Gateway
DETECTION_SERVICE_URL=http://solarpanel-detection-service:8000  # In ECS/K8s
# or
DETECTION_SERVICE_URL=http://internal-alb-name.region.elb.amazonaws.com  # Using ALB
```

### 2. AWS-specific Service Communication Options

Depending on your deployment model, use these patterns:

- **ECS with Service Discovery**: Use service discovery namespaces with DNS
  ```
  DETECTION_SERVICE_URL=http://detection-service.internal:8000
  ```

- **ECS with Application Load Balancer**:
  ```
  DETECTION_SERVICE_URL=http://internal-detection-alb-12345.us-east-1.elb.amazonaws.com
  ```

- **Kubernetes**: Use Kubernetes service names
  ```
  DETECTION_SERVICE_URL=http://solarpanel-detection-svc.default.svc.cluster.local:8000
  ```

- **API Gateway and Lambda**: Use API Gateway URLs
  ```
  DETECTION_SERVICE_URL=https://api-id.execute-api.region.amazonaws.com/stage
  ```

Using environment variables for service URLs allows your application to be:
- Environment agnostic (works locally, in staging, and production)
- Cloud provider agnostic
- Easier to scale and reconfigure without code changes

## MinIO/S3 Configuration for AWS

The system supports both local MinIO and AWS S3. When deploying to AWS, you can use either:

### Option 1: AWS S3 (Recommended for Production)

```bash
# AWS S3 configuration
MINIO_ENDPOINT=s3.amazonaws.com
MINIO_SECURE=True
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1
MINIO_BUCKET=your-bucket-name
```

### Option 2: Self-hosted MinIO

```bash
# Self-hosted MinIO configuration
MINIO_ENDPOINT=your-minio-server  # Without http:// prefix
MINIO_PORT=9000
MINIO_ACCESS_KEY=your_minio_access
MINIO_SECRET_KEY=your_minio_secret
MINIO_SECURE=False  # Set to True if using HTTPS
MINIO_BUCKET=your-bucket-name
```

The application code prioritizes AWS credentials when both are present. For local development with MinIO, you can leave AWS credential variables empty.

## Logging Configuration

If you're seeing permissions issues with logs (403 errors), ensure that:

1. All Airflow components (webserver, scheduler, worker, triggerer) use the EXACT SAME `AIRFLOW__WEBSERVER__SECRET_KEY`
2. Container time is synchronized across all machines (consider adding an NTP service)
3. Log storage is properly configured and accessible to all containers

## Troubleshooting Connection Issues

If you're experiencing MinIO/S3 connection issues, check:

1. Network connectivity between Airflow and S3/MinIO service
2. VPC Security Group rules (if applicable)
3. IAM permissions for the AWS credentials (need s3:GetObject, s3:PutObject, s3:ListBucket)
4. Bucket existence and proper permissions

### Test S3 Connectivity

Run a test container with the same environment variables to verify connectivity:

```bash
docker run --rm -it \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  python:3.10-slim \
  /bin/bash -c "pip install minio && python -c 'from minio import Minio; client=Minio(\"s3.amazonaws.com\", access_key=\"$AWS_ACCESS_KEY_ID\", secret_key=\"$AWS_SECRET_ACCESS_KEY\", secure=True); print(client.list_buckets())'"
```

### Test MinIO Connectivity

```bash
docker run --rm -it \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin \
  python:3.10-slim \
  /bin/bash -c "pip install minio && python -c 'from minio import Minio; client=Minio(\"minio:9000\", access_key=\"$MINIO_ACCESS_KEY\", secret_key=\"$MINIO_SECRET_KEY\", secure=False); print(client.list_buckets())'"
```

## Container DNS Resolution

If containers can't resolve each other by name (like the "759b6305fa8c" error in your logs), try these solutions:

1. Use fixed container names in the docker-compose file and reference those names in connection strings
2. Set up a proper DNS service if running in Kubernetes or complex environments
3. Use container IP addresses if necessary (though names are preferred)

## Checking Logs

To debug Airflow logs access issues, check if you can access the logs directly:

```bash
# Get container ID
docker ps | grep airflow-worker
# Check webserver config
docker exec -it airflow-webserver cat /opt/airflow/airflow.cfg | grep secret_key
# Check worker config 
docker exec -it airflow-worker cat /opt/airflow/airflow.cfg | grep secret_key
```

The secret_key values should match exactly across all containers. 