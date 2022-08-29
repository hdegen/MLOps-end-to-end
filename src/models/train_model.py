import mlflow
import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import lib.utils as ut

@click.command(help="Model training")
@click.option("--name", default = "", help = "Model name")
@click.option("--environment", default = "local", help = "Production (prod) or Developer (dev) environment")
def train_model(environment,name):
    
    # - Set the bucket variable depending on the environment option
    bucket = ut.get_bucket_from_env(environment)
    path = 'data'
    print(f"- Using bucket as:\n{bucket if bucket != '' else 'local'}")

    # - Get the dataset
    print("- Loading dataset")
    feat_data = "featurized_data"
    df = pd.read_csv(f"{bucket}{path}/{feat_data}.csv")
    model_name = name if name != "" else "Tree_model"

    # - Create a dataset version for this training 
    print("- Versioning dataset")
    dvc_name = ut.create_data_version_control(df,name=f"{bucket}{path}/{model_name}")
    
    # - Prepare and train
    print("- Training model")
    target = 'specie'
    features = df.columns[:-1]
    X = df[features]
    y = df[target]
    rndm = 101
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=rndm)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # - Check metrics
    print("- Checking metrics")
    pred = clf.predict(X_test)
    report, accuracy, f1, precision, recall, conf_matrix = ut.eval_classification_metrics(y_test, pred)

    print(f"{model_name}:")
    print(f"-> rndm: {rndm}")
    print(f"-> report:\n{report}")
    print(f"-> conf_matrix:\n{conf_matrix}\n")
    print(f"-> conf_matrix (%):\n{100*conf_matrix/conf_matrix.astype(float).sum(axis=1,keepdims=True)}\n")

    # - MLFlow logs
    mlflow.log_param("name", model_name)
    mlflow.log_param("data", dvc_name)
    mlflow.log_param("data shape", f'{df.shape}')
    mlflow.log_metric("rndm", rndm)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    print("- Store model experiment")
    mlflow.sklearn.log_model(clf, "Tree_model")
    
    #----------- Deploy Model ----------------
    if (ut.is_classification_model_approved(f1)):
        print("- Deploy model")
        #The followint deploy should be made to a given uri, it does not work locally
        #mlflow.sklearn.log_model(sk_model=lr, artifact_path="enade", registered_model_name=f"sk-learn-ampli-{model}")
        
        #So we just copy it to S3 manually
        path = 'models'
        folder = f"{path}/{ut.get_current_date()}-{model_name}"
        run_id = mlflow.active_run().info.run_id
        ut.deploy_model(bucket,run_id,folder)
        mlflow.log_param("deploy", folder)
        
if __name__ == '__main__':
    train_model()