from datetime import datetime, timedelta
from kubernetes import client

from airflow import DAG
from airflow.contrib.operators.kubernetes_pod_operator import KubernetesPodOperator

default_args = {
    "depends_on_past": False,
    "email": ["test@test.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "end_date": datetime(2022, 5, 16),
}

#if retrieving image from ECR
image = "000000000000.dkr.ecr.us-east-1.amazonaws.com/mlops_project:latest"
#if local image
#image = "mlops_project:latest"

#if running using aws instances
resources = client.V1ResourceRequirements(
    requests={"cpu": "1", "memory": "2000Mi"},
    limits={"cpu": "1", "memory": "3500Mi"}
)
node = 't3.medium'

resources_train = client.V1ResourceRequirements(
    requests={"cpu": "3", "memory": "10000Mi"},
    limits={"cpu": "7", "memory": "15000Mi"}
)
node_train = 'c5.2xlarge'

env_vars = {
    "AWS_ACCESS_KEY_ID": "{{ var.value.AWS_ACCESS_KEY_ID }}",
    "AWS_SECRET_ACCESS_KEY": "{{ var.value.AWS_SECRET_ACCESS_KEY }}"
}

#if running locally
#env_vars = None

with DAG(
    "DAG_name",
    description="DAG description",
    catchup=False,
    schedule_interval=None,#only on demand
    start_date=datetime(2022, 5, 15),
    dagrun_timeout=timedelta(minutes=120),
    tags=["ML pipeline"],
) as dag:

    step = {}
    
    step['make_data'] = {
        'name':'create_dataset',
        'resources':resources,
        'node_selector':node
    }
    
    step['feat_data'] = {
        'name':'build_features',
        'resources':resources,
        'node_selector':node
    }
    
    step['train_model'] = {
        'name':'train_model',
        'resources':resources_train,
        'node_selector':node_train
    }
    
    step['predict'] = {
        'name':'perform_predictions',
        'resources':resources,
        'node_selector':node
    }

    for s in step:
        step[s]['task'] = KubernetesPodOperator(
            task_id=s,
            name=step[s]['name'],
            env_vars=env_vars,
            cmds=["python"],
            arguments=["main.py","--step",s,"--environment","dev"],
            namespace="airflow",
            image=image,
            image_pull_policy="Always",
            labels={"airflow": "2.2.5"},
            is_delete_operator_pod=True,
            get_logs=True,
            resources=step[s]['resources'],
            node_selector={'node.kubernetes.io/flavor':step[s]['node_selector']},
            startup_timeout_seconds=600
        )
    
    step['make_data']['task'] >> step['feat_data']['task'] >> step['train_model']['task'] >> step['pred_data']['task']
