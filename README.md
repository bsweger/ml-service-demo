# Forecasting Service (Demo)

This repo is a modified version of the Chapter 8 project in [_Machine Learing Engineering with Python_](https://bookshop.org/p/books/machine-learning-engineering-with-python-second-edition-manage-the-lifecycle-of-machine-learning-models-using-mlops-with-practical-examples-andrew-mcm/20564864?ean=9781837631964) by Andrew P. McMahon (second edition). The original code is available at https://github.com/PacktPublishing/Machine-Learning-Engineering-with-Python-Second-Edition.

## Background

This code is a simplified representation of the FastAPI-based forecasting microservice project from Chapter 8 of MLEP. The objective of that project is to build a system that can predict future sales for a given retail store and date range.

This modified version includes only the code that is necessary to:

* Train Prophet-based forecasting models for a series of retail stores
* Log the model artifacts to MlFlow
* Create a FastAPI-based microservice to serve the model

The goal is piecing together the components of a machine learning project, not to build the best possible model or demonstrate engineering best practices.

## Overview

This repo contains two individual Python applications:

1. `train`: A script that trains Prophet-based forecasting models (one for each store in the dataset) and logs their artifacts to MLflow (this virtual environment is also where you'll spin up a local MLflow server).
2. `serve`: API that serves predictions from the trained models.

The setup instructions below are for running both the MLflow tracking server and the API locally.


## Prerequisites

To run this code, you will need the following installed on your local machine:

* Python 3.10

In addition, you will need:

* A Kaggle account (so the training script can download the [dataset](https://www.kaggle.com/c/rossmann-store-sales/data))


## Setup

To run this project: clone the repo, open a terminal, make sure you're using Python 3.10, and follow the instructions below.

### Install the `train` dependencies

1. From the root of the repo: `cd train`
2. Create a Python virtual environment: `python3 -m venv .venv --prompt train`
3. Activate the virtual environment: `source .venv/bin/activate`
4. Install the dependencies: `pip install -r requirements.txt`

### Install the `serve` dependencies

1. Open a new terminal window
2. From the root of the repo: `cd serve`
3. Create a Python virtual environment: `python3 -m venv .venv --prompt serve`
4. Activate the virtual environment: `source .venv/bin/activate`
5. Install the dependencies: `pip install -r requirements.txt`

### Start the local MLflow tracking server

MLflow stores two types of data:

* Metadata about experiments, runs, and models
* Model artifacts (_e.g._, the trained model parameters)

For this project, which runs everything locally, the metadata is stored in a local sqlite database that MLflow creates automatically (`mlflow.db`). The model artifacts are stored in a directory called `.mlruns` (also created automatically by MLflow).

1. From the root of the repo: `cd train`
2. Make sure you're in the `train` virtual environment created above: `source .venv/bin/activate`
3. Start the MLflow tracking server. The `--backend-store-uri` and `--artifacts-destination` directives tell MLflow where to store the metadata and model artifacts described above:

    ```
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
2. From the root of the repo: `cd train`
3. Make sure you're in the `train` virtual environment created above: `source .venv/bin/activate`
4. Run the script that downloads the retail dataset, creates a new MLflow experiment, and performs an MLflow run for each retail store: `python src/train_forecasters_mlflow.py`
5. Once the above script completes, you can view the output in the `Experiments` section of your local MLflow server: http://localhost:7777. You should see an experiment called `retail-forecaster`. Click that link to see a list of MLflow runs (one for each store in the retail dataset). Clicking on a run's link will display metadata and a series of related artifacts (including the trained model).

## Registering and tagging models

The models created by `train_forecaster_mlflow.py` need to be _registered_ before the API can reference them.

You can use the MLFlow UI (http://localhost:7777) to register models:

1. Register models from individual MLflow runs: https://mlflow.org/docs/latest/model-registry.html#register-a-model. Each store should have a corresponding model with the following naming convention: `prophet-retail-forecaster-store-<store id>` (_e.g._, `prophet-retail-forecaster-store-2`)
2. Each registered model can contain one or more more versions. Make sure one model version for each store has an assigned alias called `production` (which is the default alias used by the API). To assign an alias via the MLflow UI: https://mlflow.org/docs/latest/model-registry.html#deploy-and-organize-models

**Note:** Because this process is manual, you don't need to create a model for each of the 1000+ stores. Just do a few to get started and make a note of the corresponding store ids (which you'll need when using the API).

## Accessing forecast data via the API

### Running the API locally

1. From the root of the repo: `cd serve`
2. Make sure you're in the `serve` virtual environment created above: `source .venv/bin/activate`
3. Navigate to the directory that contains the FastAPI application: `cd src`
3. Start the FastAPI server: `uvicorn app:app --host 127.0.01 --port 8080 --reload`
4. In a browser, access the "healthcheck" route to verify that the API can access the MLflow tracking server: http://localhost:8080/health/
5. Request a forecast from one of the production models created above:  
    ```
    curl --request POST --location \
      --url http://127.0.0.1:8080/forecast \
      --header 'content-type: application/json' \
      --data '[{
            "store_id": "1114",
            "begin_date": "2023-12-01T00:00:00Z",
            "end_date": "2023-12-03T00:00:00Z"
        }]'
    ```
6. A successful request will return a JSON response with the forecast data for the specified store and date range. For example:  
    ```
    [{"request":{"store_id":"1114","begin_date":"2023-12-01T00:00:00Z","end_date":"2023-12-03T00:00:00Z"},"forecast":[{"timestamp":"2023-12-01T00:00:00","value":24726},{"timestamp":"2023-12-02T00:00:00","value":26097},{"timestamp":"2023-12-03T00:00:00","value":25263}]}]
    ```
