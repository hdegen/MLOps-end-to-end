import mlflow
import click
import pandas as pd

import lib.utils as ut
from sklearn import datasets

@click.command(help="Create and process dataset")
@click.option("--environment", default = "local", help = "Production (prod) or Developer (dev) environment")
def process_data(environment):

    # - Set the bucket variable depending on the environment option
    bucket = ut.get_bucket_from_env(environment)
    path = 'data'
    print(f"- Using bucket as: {bucket if bucket != '' else 'local'}")

    # - Get the dataset
    print("- Obtaining dataset")
    #from a csv:
    raw_data = "iris"
    try:
        df = pd.read_csv(f"{bucket}{path}/{raw_data}.csv")
    except:
        iris = datasets.load_iris()
        df = pd.DataFrame(iris.data)
        df.columns = iris.feature_names
        target = 'specie'
        df[target] = iris.target
        df.to_csv(f"{bucket}{raw_data}.csv",index=False)
    
    #or from a SQL query:
    #conn = ut.get_pyathena_connector(bucket=bucket,region='us-east-1',pd_cursor=True)
    #query_file = "data/query.sql"
    #df = ut.get_df_from_query(file=query_file,conn=conn,cursor=True)

    # - Cleaning code...
    print("- Cleaning dataset")
    #df.drop_duplicates(inplace=True)
    #...

    # - Save cleaned dataset
    print("- Saving dataset:")
    output_file = f"{bucket}{path}/processed_data.csv"
    df.to_csv(output_file,index=False)

if __name__ == '__main__':
    process_data()