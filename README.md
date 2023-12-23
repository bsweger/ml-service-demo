# Forecasting Service (Demo)

This is a modified version of the Chapter 8 project in _Machine Learing Engineering with Python_ by Andrew P. McMahon (second edition). The original code is available at https://github.com/PacktPublishing/Machine-Learning-Engineering-with-Python-Second-Edition.

## Background

This code represents the FastAPI-based forecasting microservice project from Chapter 8 of MLEP. This modified version includes only the code that is necessary to:

* Train Prophet-based forecasting models for a series of retail stores
* Log the model artifacts to MlFlow
* Create a FastAPI-based microservice to serve the model

The primary focus is to work through the lifecycle of a machine learning project with an operational lens (_i.e._, not to build the best possible model or produce production-ready code).

## Overview

This repo contains two individual Python applications:

1. `train`: A script that trains Prophet-based forecasting models (one for each store in the dataset) and logs their artifacts to MLflow (this virtual environment is also where you'll spin up a local MLflow server).
2. `serve`: API that serves predictions from the trained models (this runs locally, optionally via a minikube cluster).

## Prerequisites

To run this code, you will need the following installed on your local machine:

* Python 3.10
* [minikube](https://minikube.sigs.k8s.io/docs/start/) (to run Kubernetes locally)

In addition, you will need:

* A Kaggle account (so the training script can download the [dataset](https://www.kaggle.com/c/rossmann-store-sales/data))
* Access to a Docker registry (e.g., [Docker Hub](https://hub.docker.com/)


## Setup

### Install the `train` dependencies

1. From the root of the repo: `cd train`
2. Create a Python virtual environment: `python3 -m venv .venv --prompt train`
3. Activate the virtual environment: `source .venv/bin/activate`
4. Install the dependencies: `pip install -r requirements.txt`

### Install the `serve` dependencies

1. From the root of the repo: `cd serve`
2. Create a Python virtual environment: `python3 -m venv .venv --prompt serve`
3. Activate the virtual environment: `source .venv/bin/activate`
4. Install the dependencies: `pip install -r requirements.txt`

### Start the local MLflow tracking server

1. From the root of the repo: `cd train`
2. Make sure you're in the `train` virtual environment created above: `source .venv/bin/activate`
3. Start the MLflow tracking server. Here we're storing the server's metadata in a local sqlite database and instructing the server to store the output of experiment runs in the `.mlruns` directory:
    ```
    mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --artifacts-destination .mlruns \
    --host 0.0.0.0:7777
    ```
4. You should be able to access the MLflow UI at http://localhost:7777

## Training the models and logging their artifacts on MLflow

1. From the root of the repo: `cd train`
2. Make sure you're in the `train` virtual environment created above: `source .venv/bin/activate`
3. Run the script that downloads the retail dataset, trains a model for each store, and logs model artifacts to the MLflow server you started above: `python src/train_forecasters_mlflow.py`
4. Once the script completes, you can view the logged model artifacts in the `Experiments` section of the MLflow UI at http://localhost:7777

## Creating production models

The API will be looking for production models. For now, you can use the MLFlow UI to create models from the logged artifacts and set their stage to `Production`.

**Note:** Because this process is currently manual, you don't need to create a model for each store. Just do a few to get started and make a note of the corresponding store id (which you'll need when using the API).

## Accessing forecast data via the API

### Running the API locally, without Kubernetes

1. From the root of the repo: `cd serve`
2. Make sure you're in the `serve` virtual environment created above: `source .venv/bin/activate`
3. Start the FastAPI server: app:app --host 127.0.01 --port 8080 --reload
4. Verify that the API can access the MLflow tracking server: `http://localhost:8080/health/`
5. Request a forecast from one of the production models created above:  
    ```
    curl --request POST \
    --url http://127.0.0.1:8080/forecast \
    --header 'content-type: application/json' \
    --data '[
        {
            "store_id": "<store id that corresponds to a production model>",
            "begin_date": "2023-12-01T00:00:00Z",
            "end_date": "2023-12-07T00:00:00Z"
        }
    ]'
    ```
6. A successful request will return a JSON response with the forecast data for the specified store and date range. For example:  
    ```
    [
        {
            "request": {
                "store_id": "2",
                "begin_date": "2023-12-01T00:00:00Z",
                "end_date": "2023-12-07T00:00:00Z"
            },
            "forecast": [
                {
                    "timestamp": "2023-12-01T00:00:00",
                    "value": 6090
                },
                {
                    "timestamp": "2023-12-02T00:00:00",
                    "value": 4316
                },
                {
                    "timestamp": "2023-12-03T00:00:00",
                    "value": 6277
                },
                {
                    "timestamp": "2023-12-04T00:00:00",
                    "value": 7745
                },
                {
                    "timestamp": "2023-12-05T00:00:00",
                    "value": 7062
                },
                {
                    "timestamp": "2023-12-06T00:00:00",
                    "value": 7683
                },
                {
                    "timestamp": "2023-12-07T00:00:00",
                    "value": 6951
                }
            ]
        }
    ]
    ```

### Running the API on a local kubernetes cluster managed by minikube

build docker image
push image to docker registry
modify kubernetes yaml to image you just pushed (and create a container secret if necessary)
kubectl incantations
healthcheck
request forecast
