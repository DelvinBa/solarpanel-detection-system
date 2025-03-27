from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def simple_task():
    """Very simple task that just returns a string."""
    return "Done!"

with DAG(
    dag_id="quick_test",
    start_date=datetime(2024, 3, 26),
    schedule_interval=None,
    catchup=False
) as dag:
    
    task = PythonOperator(
        task_id="simple_task",
        python_callable=simple_task
    ) 