import pytest
from airflow.models import DagBag

@pytest.fixture(scope="session")
def dagbag():
    # adjust the path if your DAGs live somewhere else
    return DagBag(dag_folder="airflow/dags", include_examples=False)

def test_no_import_errors(dagbag):
    # Makes sure every .py in `dags/` parses
    assert dagbag.import_errors == {}

def test_my_dag_loaded(dagbag):
    dag = dagbag.get_dag("my_dag_id")        # replace with your actual dag_id
    assert dag is not None
    assert dag.schedule_interval == "@daily"  # or whatever you expect
    # check you have the right number of tasks
    assert len(dag.tasks) == 5
