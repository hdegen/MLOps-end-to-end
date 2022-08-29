# MLOps project - AWS
Due to necessity of automate ML pipelines and best practices I decided to create a repository with a template that I developed and that can be quickly used to create a full ML pipeline from scratch easily. You can further adapt this code to the necessities of your projects/teams, but you should have everything needed to understand the principles of MLOps and to create it in your own cloud environment.

## Intro to MLOps
Since Machine Learning is currently being used everywhere to solve different data-related problems, an organizational issue rose. How to keep track of all of the experiments and models that were trained? Which one was the best? Which data had been used to create them? What happened to a new model created this year that has worse metrics than the others created last year? What had been done differently back then? Where are these models stored? Does anyone remember where that awesome model developed a month ago is and know why it has not been possible to reproduce it again?
All of these issues made it necessary to create an environment where the full ML lifecycle is automated and tracked, making it possible to adress solutions to all of these problems. This is known as Machine Learning “Operations” (MLOps), similarly to what is done in the combination of  Software Development and Operations (DevOps). The purpose of this repository is not to explain all of the faces of MLOps but to show how to create a full template based on the best practices of MLOps applied to a cloud environment. Although this is focused on the Amazon Web Services, all of the solutions used here are open source, making it possible to apply this template to any cloud environment and not become locked in just using their prompt solutions.
For this purpose, there are some steps that are needed:
- Standardize the project structure: Imagine that you are a member of a huge team of data scientists, all working on shared projects. What would happen if each one just decided how the structure of each project will follow? It will be a mess and it will be hard to work across different projects. But if the common template and structure is known, it's easy to start a new project, correct a bug in another one or finish a project that you did not start.
- Create a full reproductible ML pipeline: Using MLFlow you can simply create your full pipeline defining environment to be used, which steps to follow, and which parameters should be stored, alongside with your trained model.
- Use data versioning for model training: As you can end up producing many different models when experimenting the best approach/features/method/algorithm/model to obtain the best metrics, it is a good practice to save the data you used to train your model, as this could be a game changer in future questions.
- Containerization of the pipeline: Instead of trying to make your project work in a different OS, understanding why your project works on your computer but not on the computer of your team member, would it not be better to run this in a container, which does not depend on the OS of the host and solve all of the possible compatibility problems of the environment?
- Use a job scheduler to run the pipeline on the cloud: Instead of running these training procedures locally, would it not be awesome to use cloud computing to run your jobs whenever you want, with much more computational resources?

## Project structure
A first good procedure is to standardizate your project following the best development practices, and for this purpose there is no better structure than cookiecutter, which has a proper branch for data science projects cookiecutter-data-science (github). This is an excellent start if you still do not have your own structure and it is easy to set up a data science project. The template described in this article is based on cookiecutter but does not use all of the functionalities as written in cookiecutter and modify several others. This template is the most general as possible for different possible projects. Feel free to adapt the code to have the best functionalities that will be needed by your team. The way it is implemented here is possible for you to use different AWS accounts and buckets for the same project, since you will probably want to test your code in a development environment before really automating everything to be deployed to your production environment. This template is not focused on running a data science project with local resources, but on the cloud, although you still can perform tests in your own machine. 
What you will find is the folders in the root directory:
- data: as the focus is to run the project in a container consuming data directly from S3, this is mainly focused in retrieving data to test your code
- notebooks: this folder is built to store your analysis and notebooks
- src: this is your project source code, containing all of the code/pipeline, having subdirectories:
  - data: modules needed for obtaining and cleaning your dataset, and you could also store SQL queries if you will be consulting a database (Athena or other)
  - features: modules needed to featurize your cleaned data
  - models: modules needed for training models and using them to predictions
  - lib: your own library to make it easier to use factorize functionalities in different parts of your pipeline/code
  - examples: this is mainly for documentation and usability, you can show examples of how to use and call your pipeline

## Environment
To avoid package compatibility problems you can create a conda environment from the conda.yaml file to centralize all of the packages used and make sure all of the members can work withouth any problems. Just remember that to run this template on the cloud you will need to export your aws credentials to your terminal before and make sure to install [aws-cli 2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html). If you just want to play around just use the environment as "local" and you will be fine!

## ML Pipeline
Para evitar ficar preso nas soluções já desenhadas da AWS, decidiu-se criar as pipelines deste template baseado no MLFlow. O MLFlow nos fornece um ambiente centralizado e fácil de registro, manutenção e reprodução de treinos e resultados de experimentos de ML, permitindo um deploy de modelos de forma muito simples. Ele permite uma fácil leitura e visualizacao de todos os treinos realizados, parâmetros e organização de forma que fica fácil encontrar e determinar os melhores modelos. Ele também já possui nativamente uma ótima integração com o conda.
A estrutura de arquivos e códigos para utilização do MLFlow é bem clara e objetiva. Existe um arquivo chamado MLproject onde você configura o nome do arquivo do conda para ser utilizado, e todos os entry points que o MLFlow vai ter, juntamente com seus parâmetros. Aqui são definidos todos os passos da pipeline e a funcao principal, responsavel por estruturar os chamados de cada passo.
 
O arquivo main.py possui novamente todos os passos dessa pipeline e chama esses comandos na sequência em que a pipeline deve ser rodada utilizando as funcionalidades de chamada do MLFlow:
Cada chamada dentro de um job ativo cria um subjob, e todos os parâmetros que forem salvos são acoplados ao job principal. Para rodar, basta chamar a função main do python com os parâmetros desejados. Para o ENADE existem inúmeros parâmetros de entradas, mas para um projeto generalista não necessariamente, por isso apenas alguns casos de parâmetros foram incluídos.
Para iniciar, voce precisa instalar o mlflow via pip:
pip install mlflow
E tambem na sua base do conda:
conda install -c conda-forge mlflow
Isso deve ser o suficiente, visto que o proprio mlflow ira criar um environment do conda para rodar o codigo. Para rodar o seu projeto, basta entrar na pasta src e executar a chamada da funcao main:
python main.py --parameter {parameter}
Todos os parametros possuem um valor default que foi setado no arquivo MLproject, portanto caso queira utilizar com todos os valores default basta efetuar a chamada limpa da funcao main:
python main.py
Caso queira rodar passos separadamente, basta chamar cada passo separadamente:
python main.py --step "make_data"
python main.py --step "feat_data"
python main.py --step "train_model"

python main.py --step "predict"

Para setar em qual ambiente o script deve rodar, apenas defina a variavel de ambiente:
python main.py --step "make_data" --environment "dev"
Lembre-se de que para rodar no ambiente da aws as suas credenciais daquele ambiente ja devem ter sido exportadas para o seu terminal.
Ao rodar a pipeline, o MLFlow criara uma pasta chamada mlruns dentro de src para armazenar todos os parametros que foram definidos no codigo. Para visualizar os indicadores, apenas rode:
mlflow ui
E acesse seu localhost:5000 para visualizar a pagina:
Note que conseguimos visualizar as metricas:
O dado versionado (explicacao no bloco seguinte):
E o modelo:
Ao clicar no link do modelo do sklearn, voce sera redirecionado para a parte de gerenciamento daquele modelo:
E aqui voce pode registrar um modelo para deploy, mas neste caso voce precisa que o MLFlow esteja conectado a algum banco de dados, caso contrario o mlflow nao conseguira registrar o modelo e consequentemente fazer o deploy. Para possiveis testes locais, foi adicionado uma funcao que copia o modelo para sua pasta local ou para o s3 de forma que ele pode ser obtido quando queremos utiliza-lo para previsoes. Quando o MLFlow estiver online atualizaremos esse tutorial para tambem receber o modelo via model registry.

## Data versioning
[DVC](https://dvc.org/) is currently the best solution to data versioning since it has a great integration with code versioning tools as github. The whole idea behing data versioning is the fact that sometimes you will need to check your original training data to check differences with other models and other problems. If you do not have your original data it could make it impossible to find out why that was a great model and why you are not being able to reproduce those results, sometimes bringing different results/metrics and sometimes could be hard to understand why. However, if the dataset is always stored, it is easy to check if that could be something related to featurization or some procedure that you had applied to the dataset.
In that way, DVC is a great tool but serves maily to versionate your data in a storage, keeping tracking of the models and data. However, if you are using MLFlow and you can create a copy of your data and save it to the model parameters, we can mantain the tracking just using MLFlow, making it not completely necessary to use DVC. We can simply create a directory in the Simple Storage Service (S3) and save the trained data there, keeping track of the models and data by using MLFlow, without the need of aditional packages.

## Containerization
The idea of creating images is that the project can be run in any OS without further compatibility problems. That means that you can run in your linux distribution and get the same results of your team member that uses another distribution or use windows. For this purpose, this template contains a DockerFile to create images of the project, allowing you to create images of the project and store in repositories such as Elastic Container Registry ([ECR](https://aws.amazon.com/ecr/)) and making it easier to use these images in a task scheduler, such as airflow or other built in tools. The base of the project is a conda image based on alpine distribution to be a little bit lightweigh. The process copy the src folder that contains everything needed to run the full pipeline. Just note that busybox 1.35 has critical vulnerabilities in its stable version ([1](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2018-25032),[2](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-28928)), having the need to update it to the [latest unstable version](https://pkgs.alpinelinux.org/package/edge/main/x86/busybox).
The idea is that for each open PR, you have a jenkins pipeline that build this image and deploy it in your ECR, in a way that as soon as it is done it is already ready to be consumed by the job scheduler.
If you want to test locally and are not sure how to use docker, just run the docker build script:
```
./build_docker.sh
```
And run it, locally:
```
./run_docker.sh
```
or using your aws credentials to run using aws storage and data:
```
./run_docker_cloud.sh
```

## Scheduler: [Apache Airflow](https://airflow.apache.org/)
Inside the root folder of this template there is a DAG example to be executed on airflow (dag.py). You can [install airflow locally](https://www.astronomer.io/events/recaps/official-airflow-helm-chart/) to test this template and all of its features but you will really want to set up [kubernetes on the cloud](https://aws.amazon.com/kubernetes/?nc1=h_ls) and make airflow the scheduler of the cloud jobs. This will help you to be able to use cloud machines that will give you the best performance, especially if you have a large dataset that can not be handled locally or need to train neural networks using GPUs.
Each step of the pipeline can be run within an airflow job, but you can also just run the full pipeline in one call, that is up to you. If you had large datasets and each step of your pipeline have different computational needs, you should run your pipeline by steps since you can setup different [EC2 instances](https://aws.amazon.com/ec2/instance-types/) and resources needed by each step. You may need more memory to load, clean and transform the first dataset but you may need CPUs/GPUs when training or performing another action. This allows you to choose the best resources for each step, increasing performance and reducing costs. Of course, you should always improve your algorithms and methods instead of just selecting more resources to your project, and there is no sense in increasing CPUs or adding GPUs if your code is not handling parallelization. You can use python multiprocessing tools to perform CPU parallelization in your own functions or methods or if you were just using simple pandas functions on dataframes you can use packages such as [dask](https://www.dask.org/) to parallelize some actions.

