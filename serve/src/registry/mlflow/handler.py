import logging
import os
from pprint import pprint

import mlflow
from mlflow.client import MlflowClient
from mlflow.pyfunc import PyFuncModel

# Tell the service where to find the MLflow tracking server
# If running the service via minikube, 'host.minikube.internal' will resolve to the host machine
MLFLOW_PORT = os.getenv('MLFLOW_TRACKING_PORT', '7777')
MLFLOW_TRACKING_URI = f'os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1"):{MLFLOW_PORT}'
MLFLOW_TRACKING_URI_MINIKUBE = f'http://host.minikube.internal:{MLFLOW_PORT}'


class MLFlowHandler:
    def __init__(self) -> None:
        # get URI of the MLflow tracking server (assumption: the tracking server is running locally)
        is_kubernetes = 'KUBERNETES_SERVICE_HOST' in os.environ and 'KUBERNETES_SERVICE_PORT' in os.environ
        if is_kubernetes:
            # if service is running on kubernetes, we'll assume it's on a local cluster managed by minikube
            tracking_uri = f'http://host.minikube.internal:{MLFLOW_PORT}'
        else:
            tracking_uri = f'http://127.0.0.1:{MLFLOW_PORT}'

        logging.info(f'Setting MLFlow tracking URI to {tracking_uri}')
        self.client = MlflowClient(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)

    def check_mlflow_health(self) -> None:
        try:
            experiments = self.client.search_experiments()
            for rm in experiments:
                pprint(dict(rm), indent=4)
                return 'Service returning experiments'
        except Exception:
            return f'Error calling MLFlow. Tracking URI={self.client.tracking_uri}'

    def get_production_model(self, store_id: str) -> PyFuncModel:
        model_name = f'prophet-retail-forecaster-store-{store_id}'
        model_uri = f'models:/{model_name}/production'
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        logging.info(f'Got model {model} from {model_uri}')
        return model
