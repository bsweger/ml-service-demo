# Forecasting Service (Demo)

This repo is a modified version of the Chapter 8 project in [_Machine Learing Engineering with Python_](https://bookshop.org/p/books/machine-learning-engineering-with-python-second-edition-manage-the-lifecycle-of-machine-learning-models-using-mlops-with-practical-examples-andrew-mcm/20564864?ean=9781837631964) by Andrew P. McMahon (second edition). The original code is available at https://github.com/PacktPublishing/Machine-Learning-Engineering-with-Python-Second-Edition.

## Background

This code is a simplified representation of the FastAPI-based forecasting microservice project from Chapter 8 of MLEP. The objective of that project is to build a system that can predict future sales for a given retail store and date range.

This modified version includes only the code that is necessary to:

* Train Prophet-based forecasting models for a series of retail stores
* Log the model artifacts to MlFlow
* Create a FastAPI-based microservice to serve the model
* Use Kubernetes (locally, via minikube) to deploy the microservice

The goal is piecing together the components of a machine learning project, not to build the best possible model or demonstrate engineering best practices.

## Overview

This repo contains two individual Python applications:

1. `train`: A script that trains Prophet-based forecasting models (one for each store in the dataset) and logs their artifacts to MLflow (this virtual environment is also where you'll spin up a local MLflow server).
2. `serve`: API that serves predictions from the trained models.

The setup instructions below are for running both the MLflow tracking server and the API locally.


## Prerequisites

To run this code, you will need the following installed on your local machine:

* Python 3.10
* minikube: https://minikube.sigs.k8s.io/docs/start/
* kubectl, the Kubernetes command-line tool: https://kubernetes.io/docs/tasks/tools/#kubectl
* Docker: https://docs.docker.com/get-docker/

In addition, you will need:

* A Kaggle account (so the training script can download the [dataset](https://www.kaggle.com/c/rossmann-store-sales/data))
* Access to a docker registry (so you can push the API image to a registry that minikube can access)


## Setup

Once all the above prerequisites are satistifed, you're ready to run this proejct: clone the repo, open a terminal, make sure you're using Python 3.10, and follow the instructions below.

### Install the `train` dependencies

1. From the root of the repo:

    ```bash
    cd train
    ```

2. Create a Python virtual environment:

    ```bash
    python3 -m venv .venv --prompt train
    ```

3. Activate the virtual environment:

    ```bash
    source .venv/bin/activate
    ```

4. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Install the `serve` dependencies

1. Open a new terminal window

2. From the root of the repo:

    ```bash
    cd serve
    ```

3. Create a Python virtual environment:

    ```bash
    python3 -m venv .venv --prompt serve
    ```

4. Activate the virtual environment:

    ```bash
    source .venv/bin/activate
    ```

5. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Start the local MLflow tracking server

MLflow stores two types of data:

* Metadata about experiments, runs, and models
* Model artifacts (_e.g._, the trained model parameters)

For this project, which runs everything locally, the metadata is stored in a local sqlite database that MLflow creates automatically (`mlflow.db`). The model artifacts are stored in a directory called `.mlruns` (also created automatically by MLflow).

1. From the root of the repo:

    ```bash
    cd train
    ```

2. Make sure you're in the `train` virtual environment created above:

    ```bash
    source .venv/bin/activate
    ```

3. Start the MLflow tracking server. The `--backend-store-uri` and `--artifacts-destination` directives tell MLflow where to store the metadata and model artifacts described above:

    ```bash
    mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --artifacts-destination .mlruns \
    --host 0.0.0.0:7777
    ```
4. You should now be able to access the MLflow UI at http://localhost:7777. If you're prompted to opt-in to the new model registry UI, say yes.

## Training the models and logging their artifacts on MLflow

This project uses a dataset of retail store sales from Kaggle. The final goal is a system that predicts future sales for a given store and date range. To accomplish this, we first need to train one model for each store.

**Note:** In keeping with this repo's "simplest possible" ethos, the script below trains each store model sequentially (the book has instructions for using Ray to train them in parallel).

1. Open a new terminal window

2. From the root of the repo:

    ```bash
    cd train
    ```

3. Make sure you're in the `train` virtual environment created above:

    ```bash
    source .venv/bin/activate
    ```

4. Run the script that downloads the retail dataset, creates a new MLflow experiment, and performs an MLflow run for each retail store:

    ```bash
    python src/train_forecasters_mlflow.py
    ```

5. Once the above script completes, you can view the output in the `Experiments` section of your local MLflow server: http://localhost:7777. You should see an experiment called `retail-forecaster`. Click that link to see a list of MLflow runs (one for each store in the retail dataset). Clicking on a run's link will display metadata and a series of related artifacts (including the trained model).

## Registering and tagging models

The models created by `train_forecaster_mlflow.py` need to be _registered_ before the API can reference them.

You can use the MLFlow UI (http://localhost:7777) to register models:

1. Register models from individual MLflow runs: https://mlflow.org/docs/latest/model-registry.html#register-a-model. Each store should have a corresponding model with the following naming convention: `prophet-retail-forecaster-store-<store id>` (_e.g._, `prophet-retail-forecaster-store-2`)
2. Each registered model can contain one or more more versions. Make sure one model version for each store has an assigned alias called `production` (which is the default alias used by the API). To assign an alias via the MLflow UI: https://mlflow.org/docs/latest/model-registry.html#deploy-and-organize-models

**Note:** Because this process is manual, you don't need to create a model for each of the 1000+ stores. Just do a few to get started and make a note of the corresponding store ids (which you'll need when using the API).

## Accessing forecast data via the API

Once the local MLFlow tracking server is running, you're ready to package and deploy the API service that will serve the forecasts.

### Creating and pushing a Docker image for the API

To run in a Kubernetes cluster, the API application will need to be packaged as a Docker image and pushed to a registry that the cluster can access.

These instructions assume you're using Docker Hub as your registry and will be pushing to a public repository (if you're using a private registry, there will an extract change to your Kubernetes manifeset, noted below).

Make sure that Docker is running on your machine and follow the steps below.

1. From the root of the repo:

    ```bash
    cd serve
    ```

2. Create a Docker [image](https://docs.docker.com/get-started/overview/#images) (essentially, a "template" that describes how to build and run the API application). The `build` command references the project's [Dockerfile](Dockerfile) to create the image and store it locally.

    ```bash
    docker build -t <your docker registry>/forecasting-service:latest .
    ```

    For example: `docker build -t janeway/forecasting-service:latest .`
    
    Running this command for the first time will take a few minutes, as Docker pulls a Python base image from Docker Hub and installs the project's dependencies.

    When the build completes, you can see the image on your local machine: `docker images`.

3. Push a copy of the local image to a docker registry (_e.g._, Docker Hub):

    ```bash
    docker push <your docker registry>/forecasting-service:latest
    ```

    For example: `docker push janeway/forecasting-service:latest`

    The [registry](https://docs.docker.com/get-started/overview/#docker-registries) is where your local Kubernetes cluster will get the image and use it to deploy [containers](https://docs.docker.com/get-started/overview/#containers) that run the API service.

### Deploying the API service via a local Kubernetes cluster (using minikube)

Once the API's Docker image is available in a registry, you can use it to create a Kubernetes deployment. This demo project uses miniube to run a Kubernetes cluster on your local machine.

1. Modify the project's Kubernetes manifest [direct-kube-deploy.yaml](direct-kube-deploy.yaml):

    * Replace `<your image location>` with the location of you image you pushed to a Docker registry in the previous section.
    * **Optional**: if you're using a private Docker registry, [create a secret that allows Kubernetes/minikube to access it](https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/), and add that secret name to the `imagePullSecrets` section of the manifest.

2. Start minikube (this can take a few minutes):

    ```bash
    minikube start
    ```

3. Create the Kubernetes deployment and service as specified the project's Kubernetes manifest:

    ```bash
    kubectl apply -f direct-kube-deploy.yaml
    ```

8. Open a tunnel that will allow the Kubernetes service to access the host machine:

    ```bash
    minikube tunnel
    ```

9. You should now be able to access the API! In a browser, access the "healthcheck" route to verify that the API can access the MLflow tracking server: http://localhost:8080/health/

10. Request a forecast from one of the production models created above:

    ```bash
    curl --request POST --location \
      --url http://127.0.0.1:8080/forecast \
      --header 'content-type: application/json' \
      --data '[{
            "store_id": "1114",
            "begin_date": "2023-12-01T00:00:00Z",
            "end_date": "2023-12-03T00:00:00Z"
        }]'
    ```

11. A successful request will return a JSON response with the forecast data for the specified store and date range. For example:

    ```json
    [{"request":{"store_id":"1114","begin_date":"2023-12-01T00:00:00Z","end_date":"2023-12-03T00:00:00Z"},"forecast":[{"timestamp":"2023-12-01T00:00:00","value":24726},{"timestamp":"2023-12-02T00:00:00","value":26097},{"timestamp":"2023-12-03T00:00:00","value":25263}]}]
    ```