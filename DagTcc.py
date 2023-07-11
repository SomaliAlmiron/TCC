from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.dummy import DummyOperator
from airflow.operators.bash_operator import BashOperator
from airflow.utils.task_group import TaskGroup

pathScript = "/home/somali/airflow/dags/etl_scripts"
pathTcc =  "/home/somali/airflow/dags/etl_scripts/featurestore/tcc.csv"
pathEncoder = "/home/somali/airflow/dags/etl_scripts/featurestore/tcc_proc.csv"

default_args = {
   'owner': 'teste',
   'depends_on_past': False,
   'start_date': datetime(2019, 1, 1),
   'retries': 0,
   }

with DAG(
   'dag-pipeline-tcc',
   schedule_interval=timedelta(minutes=10),
   catchup=False,
   default_args=default_args
   ) as dag:

    start = DummyOperator(task_id="start")

    with TaskGroup("preProcessing", tooltip="preProcessing") as preProcessing:
        t1 = BashOperator(
            dag=dag,
            task_id='encoder_dataset',
            bash_command="""
            cd {0}
            python3 preprocessing_tcc.py {1} {2}
            """.format(pathScript, pathTcc, pathEncoder)
        )
        [t1]

    with TaskGroup("model", tooltip="model") as model:
        t2 = BashOperator(
            dag=dag,
            task_id='modelo',
            bash_command="""
            cd {0}
            python3 ml_sklearn.py {1} {2} {3} {4} {5} {6}
            """.format(pathScript,pathEncoder, "TccClassificacao", "ModeloTcc", 100, 2, 0)
        )
        [t2]


    end = DummyOperator(task_id='end')
    start >> preProcessing >> model >> end