import numpy as np
import pandas as pd
import random
import string
import os
import sys
import requests
import boto3
import s3fs
from datetime import datetime
from pyathena import connect
from pyathena.pandas.cursor import PandasCursor
import sklearn.metrics as metrics

F1_THRESHOLD = 0.8

def get_bucket_from_env(environment):
    bucket = ""
    if (environment != 'local'):
        bucket = f"s3://bucket-{environment}/"
    return bucket

def get_pyathena_connector(bucket,region,pd_cursor=False):
    conn = connect(
        s3_staging_dir=bucket,
        region_name=region,
        cursor_class=PandasCursor if pd_cursor else None)
    return conn.cursor() if pd_cursor else conn

def get_df_from_query(file,conn,cursor=False):
    query = open(file, 'r')
    if cursor:
        df = conn.execute(query.read()).as_pandas()
    else:
        df = pd.read_sql(query.read(), conn)
    query.close()
    return df

def create_data_version_control(df,name=""):
    dvc_file_name = f"{name}_{gen_dataset_dvc(8)}.csv"
    df.to_csv(dvc_file_name,index=False)
    return dvc_file_name

def eval_classification_metrics(real, pred):
    report = metrics.classification_report(real, pred)
    accuracy = metrics.accuracy_score(real, pred)
    f1 = metrics.f1_score(real, pred, average='micro')
    precision = metrics.precision_score(real, pred, average='micro')
    recall = metrics.recall_score(real, pred, average='micro')
    conf_mat = metrics.confusion_matrix(real, pred)
    return report, accuracy, f1, precision, recall, conf_mat

def is_classification_model_approved(f1):
    return f1 > F1_THRESHOLD

def deploy_model(bucket="",run="",location=""):
    folder = f"mlruns/0/{run}/"

    if (bucket == ""):
        os.system(f"cp -r {folder} {location}")
        return
    
    s3_fs = s3fs.S3FileSystem()
    s3_path = f"{bucket}{location}"
    s3_fs.put(folder, s3_path, recursive=True)

def get_model(bucket='', folder='', model_name='', use='latest'):
    all_models = []
    if bucket == '': #local
        all_models = [f for f in os.listdir(folder if folder != '' else None) if model_name in f]
    else:
        if (bucket[-1] == '/'):
            bucket = bucket[:-1]
        client = boto3.client('s3')
        result = client.list_objects(Bucket=bucket[5:], Prefix=folder, Delimiter='/')
        print(result)
        to_rm = len(folder)
        all_models = [o.get('Prefix')[to_rm:-1] for o in result.get('CommonPrefixes') if model_name in o.get('Prefix')]
        
    print(f"- Model requested:\n{use}")
    print("- Models found:")
    print(all_models)
    if (use == 'latest'):
        all_models.sort(reverse=True)
        return all_models[0]
    else:
        f = f"{use}-{model_name}"
        return f if f in all_models else None

def gen_dataset_dvc(length=8):
    return f'{get_current_date()}_{get_random_string(length)}'

def get_current_date():
    return f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

def get_random_string(length=8,str_list=string.ascii_letters):
    return ''.join(random.choice(str_list) for i in range(length))
