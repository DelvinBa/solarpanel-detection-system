#!/bin/bash

# Initialize the database
airflow db init

# Upgrade the database
airflow db upgrade

# Create default connections
airflow connections create-default-connections

# Create admin user
airflow users create \
    --username "${AIRFLOW_USERNAME:-admin}" \
    --password "${AIRFLOW_PASSWORD:-admin}" \
    --firstname "${AIRFLOW_FIRSTNAME:-Admin}" \
    --lastname "${AIRFLOW_LASTNAME:-User}" \
    --role "${AIRFLOW_ROLE:-Admin}" \
    --email "${AIRFLOW_EMAIL:-admin@example.com}"

# Handle different commands
case "$1" in
  webserver)
    airflow webserver
    ;;
  scheduler)
    airflow scheduler
    ;;
  worker)
    airflow celery worker
    ;;
  *)
    # Start the scheduler in the background
    airflow scheduler &
    # Start the webserver
    airflow webserver
    ;;
esac 