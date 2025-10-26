# Customer churn prediction baseline models with MLflow integration.
# Author: Alan Perry C. Daen

# This script trains multiple baseline classifiers on the Bank Churn dataset,
# logs all metrics and models to an MLflow Tracking Server,
# and supports registering models for downstream analysis or serving.

# Disclaimer:
# This project is intended for educational and personal purposes only.
# While the model may achieve accurate predictions, it should not be used
# for real-world decision-making without validation from appropriate authorities.

import warnings
from pipeline.pipeline import ChurnPipeline
from config import mlflow_config

warnings.filterwarnings("ignore")

MLFLOW_TRACKING_URI = mlflow_config.MLFLOW_TRACKING_URI
DATA_PATH = "data/kaggle_hub/Churn_Modelling.csv"

def main():
    pipeline = ChurnPipeline(data_path=DATA_PATH, tracking_uri=MLFLOW_TRACKING_URI)
    pipeline.run()

if __name__ == "__main__":
    main()
