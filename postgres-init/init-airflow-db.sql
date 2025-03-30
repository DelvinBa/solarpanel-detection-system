-- initairflow.sql

-- Create airflow user and database
CREATE USER airflow WITH PASSWORD 'airflow';
CREATE DATABASE airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;

-- Connect to the newly created airflow database and fix schema ownership
\connect airflow;
ALTER SCHEMA public OWNER TO airflow;