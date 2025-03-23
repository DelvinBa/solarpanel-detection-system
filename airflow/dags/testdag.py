from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def hello_airflow():
    print("Hello, Airflow!")

# Define the DAG
with DAG(
    dag_id="simple_dag",
    schedule_interval="@daily",
    start_date=datetime(2024, 3, 21),
    catchup=False,
) as dag:
    
    task_hello = PythonOperator(
        task_id="say_hello",
        python_callable=hello_airflow
    )

    task_hello
