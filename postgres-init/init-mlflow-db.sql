-- Create mlflow user and database
CREATE USER mlflow WITH PASSWORD 'mlflow';
CREATE DATABASE mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;

-- Connect to mlflow database to set up extensions
\c mlflow
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
ALTER DATABASE mlflow OWNER TO mlflow;

-- Grant additional permissions needed for MLflow
GRANT ALL PRIVILEGES ON SCHEMA public TO mlflow;
ALTER ROLE mlflow SUPERUSER; 