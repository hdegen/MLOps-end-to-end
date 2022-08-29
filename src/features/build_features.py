import mlflow
import click
import pandas as pd

import lib.utils as ut

@click.command(help="Featurize data")
@click.option("--environment", default = "local", help = "Production (prod) or Developer (dev) environment")
def featurize_data(environment):

    # - Set the bucket variable depending on the environment option
    bucket = ut.get_bucket_from_env(environment)
    path = 'data'
    print(f"- Using bucket as:\n{bucket if bucket != '' else 'local'}")

    # - Get the dataset
    print("- Loading dataset")
    proc_data = "processed_data"
    df = pd.read_csv(f"{bucket}{path}/{proc_data}.csv")

    # - Create features
    print("- Creating features")
    #df['feat_name'] = df.apply(lambda row: row['col1']*row['col2'],axis=1)
    #...
    
    # - Save featurized dataset
    print("- Saving dataset:")
    output_file = f"{bucket}{path}/featurized_data.csv"
    df.to_csv(output_file,index=False)

if __name__ == '__main__':
    featurize_data()