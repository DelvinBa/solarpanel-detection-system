#!/bin/bash
set -e

echo "Creating Airflow admin user..."
docker exec -i solar_panel_detection-airflow-webserver-1 /bin/bash <<EOF
airflow db init
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
EOF

echo "Airflow user created. You can now login with:"
echo "Username: admin"
echo "Password: admin" 