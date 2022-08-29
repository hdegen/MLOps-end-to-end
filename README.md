# MLOps project
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
To avoid package compatibility problems you  conda.yaml para centralizar os pacotes necessários para cada projeto, sendo que esse arquivo já contém os pacotes gerais necessários para rodar o template na aws. Lembre-se de que você precisará ter instalado o conda/anaconda e também o aws-cli. Para rodar a pipeline na cloud voce tambem precisara exportar as suas credenciais da aws no terminal que for utilizar para testes.
Pipeline de ML (MLOps)
Para evitar ficar preso nas soluções já desenhadas da AWS, decidimos criar as pipelines deste template baseado no MLFlow. O MLFlow nos fornece um ambiente centralizado e fácil de registro, manutenção e reprodução de treinos e resultados de experimentos de ML, permitindo um deploy de modelos de forma muito simples. Ele permite uma fácil leitura e visualizacao de todos os treinos realizados, parâmetros e organização de forma que fica fácil encontrar e determinar os melhores modelos. Ele também já possui nativamente uma ótima integração com o conda.
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
Data versioning
O DVC lidera atualmente como melhor solução para versionamento de dados por possuir uma ótima integração com ambientes de versionamento de código como o github. A ideia por trás de versionamento de dados ocorre para validação e entendimento quando experimentos futuros começam a trazer resultados/métricas diferentes ou até mesmo muito inferiores do que modelos do passado, muitas vezes sem razão aparente. Se o dataset utilizado para treinar modelos está sempre sendo guardado, fica fácil encontrar se houve alguma mudança na faturização ou em algo de código que diminui a performance de um modelo de ML.
Sendo assim, o DVC é uma ferramenta que serve em sua base para versionar o seu dado em algum ambiente de storage, como o próprio s3 da aws e consegue criar tags no versionamento do código. Ou seja, nada mais faz do que salvar um dado em um storage e manter o tracking automático para você. A utilização de uma ferramenta inteira apenas para isso se torna desnecessária quando já estamos utilizando um ambiente de nuvem e o MLFlow. A cada modelo treinado, podemos salvar o dado novamente em um diretório próprio para versionamento de dados de modelos no s3 e linkar o nome desse dataset nos parâmetros do MLFlow, de forma que fica fácil acessar o dado original de algum modelo, sem precisar adicionar softwares externos.
Containerization
The idea of aComo a ideia dos projetos não é rodar na máquina de cada um dos membros do time mas em algum lugar centralizado, sem depender de sistema operacional específico e ambientes, o template já contempla um DockerFile para criação de imagens do projeto, facilitando o uso dele por membros com diferentes OS e em agendador de tarefas, como o airflow. O processo parte de uma imagem base do anaconda no alpine, copia os arquivos de pacotes necessários e faz as atualizações e instalações necessárias para montar a imagem final. Note que o busybox possui vulnerabilidades críticas em sua versão atualmente estável, e, portanto, deve ser utilizado sua última versão instável de release.
Assim que um PR é aberto, a ideia é ter uma pipeline no jenkins que builda uma nova imagem e disponibiliza no ECR da conta de produção, de forma que assim que o jenkins faz o deploy da imagem ela já pode ser testada no airflow, que deve apontar para o ECR.
Scheduler: Apache Airflow
Inside this template there is a DAG example to be executed on airflow. You can install airflow locally to test this template on all its features but you will really want to set up kubernetes on the cloud and make airflow the scheduler of the cloud jobs. This will help you to be able to use cloud machines that will give you the best performance, especially if you have a large dataset that can not be handled locally or need to train neural networks using GPUs.
Each step of the pipeline can be run via airflow, or you can run the end to end version of your pipeline, that is up to you. If you had large datasets and each step of your pipeline have different computational needs, you should run your pipeline by steps since you can setup different EC2 instances and resources needed by each step. You may need more memory to load, clean and transform the first dataset but you may need CPUs/GPUs when training or performing another action. This allows you to choose the best resources for each step, increasing performance and reducing costs. Of course, you should always improve your algorithms and methods instead of just selecting more resources to your project, and there is no sense in increasing CPUs or adding GPUs if your code is not handling parallelization. You can use python multiprocessing tools to perform parallelization in your own functions or methods or if you were just using simple pandas functions on dataframes you can use packages such as dask to parallelize some actions.

