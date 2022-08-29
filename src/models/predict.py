import mlflow
import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import lib.utils as ut

@click.command(help="Model prediction")
@click.option("--name", default = "", help = "Name of model")
@click.option("--model", default = "latest", help = "Model to be used")
@click.option("--environment", default = "local", help = "Production (prod) or Developer (dev) environment")
def create_prediction(environment,name,model):
        
    #Set the bucket variable depending on the environment option
    bucket = ut.get_bucket_from_env(environment)
    path = 'data'
    print(f"- Using bucket as:\n{bucket if bucket != '' else 'local'}")

    #Get the dataset
    print("- Loading dataset")
    feat_data = "featurized_data"
    df = pd.read_csv(f"{bucket}{path}/{feat_data}.csv")
    target = 'specie'
    features = df.columns[:-1]
    y = df[target]
    X = df[features]

    #Get trained model
    print("- Load model")
    path = 'models/'
    run_model = ut.get_model(bucket=bucket, folder=path, model_name=name, use=model)
    loaded_model = mlflow.pyfunc.load_model(f"{bucket}{path}{run_model}/artifacts/{name}/")
    print(loaded_model)
    
    print("- Perform prediction")
    pred = loaded_model.predict(X)
    diff = y - pred
    df[target] = pred

    # - Save dataset with prediction
    print("- Saving dataset:")
    path = 'data'
    output_file = f"{bucket}{path}/prediction.csv"
    df.to_csv(output_file,index=False)

if __name__ == '__main__':
    create_prediction()
