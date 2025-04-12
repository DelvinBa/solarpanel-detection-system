CREATE DATABASE mlflow;
CREATE USER mlflow WITH PASSWORD 'mlflow';
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;

-- Connect to the mlflow database to set permissions inside it
\c mlflow

-- Give the mlflow user permission to create schemas and tables in the mlflow database
ALTER SCHEMA public OWNER TO mlflow;
GRANT ALL ON SCHEMA public TO mlflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mlflow;
