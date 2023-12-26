import logging
import os
import sys
import time
import warnings

import kaggle
import mlflow
import pandas as pd
from prophet import Prophet, serialize
from sklearn.metrics import mean_absolute_error, mean_squared_error

EXPERIMENT_NAME = 'retail-forecaster'
MODEL_BASE_NAME = 'prophet-retail-forecaster-store-'
MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:7777')

mlflow.set_tracking_uri(MLFLOW_URI)


def download_kaggle_dataset(kaggle_dataset: str = 'pratyushakar/rossmann-store-sales') -> None:
    api = kaggle.api
    logger.info(f'Kaggle username = {api.get_config_value("username")}')
    kaggle.api.dataset_download_files(kaggle_dataset, path='./', unzip=True, quiet=False)


def extract_params(pr_model):
    return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}


def prep_store_data(df: pd.DataFrame, store_id: int = 4, store_open: int = 1) -> pd.DataFrame:
    df_store = df[(df['Store'] == store_id) & (df['Open'] == store_open)].reset_index(drop=True)
    df_store['Date'] = pd.to_datetime(df_store['Date'])
    df_store.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)
    return df_store.sort_values('ds', ascending=True)


def train_predict(
    store_id: int, df: pd.DataFrame, train_fraction: float, seasonality: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    # grab split data
    # df.shape[0] returns number of rows in the dataframe
    train_index = int(train_fraction * df.shape[0])
    # grab the first n rows of dataframe, as determined by train_index
    df_train = df.copy().iloc[0:train_index]
    # grab the remaining rows of dataframe
    df_test = df.copy().iloc[train_index:]

    # train and predict
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(description=f'Prophet experiment for store {store_id}'):
        # create Prophet model
        model = Prophet(
            yearly_seasonality=seasonality['yearly'],
            weekly_seasonality=seasonality['weekly'],
            daily_seasonality=seasonality['daily'],
            interval_width=0.95,
        )

        # fit the model using the training dataframe
        model.fit(df_train)

        # assign each row in the test dataframe a predicted value
        predicted = model.predict(df_test)

        # calculate metrics
        mse = mean_squared_error(df_test['y'], predicted['yhat'])
        mae = mean_absolute_error(df_test['y'], predicted['yhat'])

        # log metrics
        mlflow.log_metric('mse', mse)
        mlflow.log_metric('mae', mae)

        artifact_path = f'{MODEL_BASE_NAME}{store_id}'
        mlflow.prophet.log_model(model, artifact_path=artifact_path)

        params = extract_params(model)
        mlflow.log_params(params)

        model_uri = mlflow.get_artifact_uri(artifact_path=artifact_path)
        logger.info(f'Model artifact logged to: {model_uri}')

    return predicted, df_train, df_test, train_index


def main():
    # If data present, read it in, otherwise, download it
    file_path = 'train.csv'
    if os.path.exists(file_path):
        logger.info('Dataset found, reading into pandas dataframe.')
        df = pd.read_csv(file_path)
    else:
        logger.info('Dataset not found, downloading ...')
        download_kaggle_dataset()
        logger.info('Reading dataset into pandas dataframe.')
        df = pd.read_csv(file_path)

    # Get the unique store IDs
    # store_ids = dataset.unique("Store") # if you were using Ray DataFrame
    store_ids = df['Store'].unique()  # [0:50] #for testing

    # Define the parameters for the Prophet model
    seasonality = {'yearly': True, 'weekly': True, 'daily': False}

    start = time.time()
    predictions = []
    train_data = []
    test_data = []
    train_indices = []
    for store_id in store_ids:
        df_store = prep_store_data(df, store_id=store_id)
        predicted, df_train, df_test, train_index = train_predict(
            store_id=store_id,
            df=df_store,
            train_fraction=0.8,
            seasonality=seasonality,
        )
        predictions.append(predicted)
        train_data.append(df_train)
        test_data.append(df_test)
        train_indices.append(train_index)

    results = {
        'predictions': predictions,
        'train_data': train_data,
        'test_data': test_data,
        'train_indices': train_indices,
    }
    total_time = time.time() - start

    logger.info(
        {
            'models trained': len(store_ids),
            'time': f'{total_time/60:.2f} minutes',
        }
    )

    logger.info('Done!')


if __name__ == '__main__':
    # Set up logging
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Suppress warnings (some of which are important for real-world applications but noisy for a demo project)
    warnings.filterwarnings('ignore')

    # Suppress all Prophet logging except for critical errors
    prophet_logger = logging.getLogger('cmdstanpy')
    prophet_logger.addHandler(logging.NullHandler())
    prophet_logger.propagate = False
    prophet_logger.setLevel(logging.CRITICAL)

    main()
