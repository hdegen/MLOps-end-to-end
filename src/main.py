import mlflow
import click

WORKFLOW_STEPS = ['test', 'make_data', 'feat_data', 'train_model', 'predict', 'full']
ENVIRONMENT_OPTIONS = ['local', 'dev', 'prod']

@click.command(help = "MLOps - ML Pipeline")
@click.option("--step", default = "full", help = "Pipeline step")
@click.option("--environment", default = "local", help = "Production (prod) or Developer (dev) environment")
def workflow(step,environment):
    assert (step in WORKFLOW_STEPS), "The step provided is not contained in the pipeline."
    assert (environment in ENVIRONMENT_OPTIONS), "The environment provided is not contained in the possible options."
    
    with mlflow.start_run() as active_run:

        if (step == 'test'):
            print("\n-----------------------------------------")
            print("--- Test MLFlow and environment")
            mlflow.run(".", "run_test")
            return
        
        if (step in ['make_data','full']):
            print("\n-----------------------------------------")
            print("--- LAUNCHING 'make_data'")
            make_data_run = mlflow.run(".", "make_data", parameters={"environment": environment})
            make_data_run = mlflow.tracking.MlflowClient().get_run(make_data_run.run_id)

        if (step in ['feat_data','full']):
            print("\n-----------------------------------------")
            print("--- LAUNCHING 'feat_data'")
            feat_data_run = mlflow.run(".", "build_features", parameters={"environment": environment})
            feat_data_run = mlflow.tracking.MlflowClient().get_run(feat_data_run.run_id)
        
        if (step in ['train_model','full']):
            print("\n-----------------------------------------")
            print("--- LAUNCHING 'train_model'")
            train_model_run = mlflow.run(".", "train_model", parameters={"name":"Tree_model", "environment": environment})
            train_model_run = mlflow.tracking.MlflowClient().get_run(train_model_run.run_id)
        
        if (step in ['predict','full']):
            print("\n-----------------------------------------")
            print("--- LAUNCHING 'predict'")
            pred_data_run = mlflow.run(".", "predict", 
                parameters={
                    "name": "Tree_model",
                    "model": "latest",
                    "environment": environment})
            pred_data_run = mlflow.tracking.MlflowClient().get_run(pred_data_run.run_id)

if __name__ == '__main__':
    workflow()
