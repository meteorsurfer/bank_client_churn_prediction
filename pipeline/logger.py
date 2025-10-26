from mlflow.models import infer_signature
import mlflow

class MLflowLogger:
    def __init__(self, tracking_uri, experiment_name):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def log(self, model_name, model, X_test, y_pred, report_dict):
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params({"Model": model_name})
            mlflow.log_metrics({
                "Accuracy": report_dict["accuracy"],
                "Class 0 Recall": report_dict["0"]["recall"],
                "Class 1 Recall": report_dict["1"]["recall"],
                "Class 0 Precision": report_dict["0"]["precision"],
                "Class 1 Precision": report_dict["1"]["precision"],
                "Macro F1 Score": report_dict["macro avg"]["f1-score"],
            })

            if "XGB" in model_name:
                mlflow.xgboost.log_model(model, model_name)
            elif "CatBoost" in model_name:
                mlflow.catboost.log_model(model, model_name)
            elif "LGBM" in model_name:
                signature = infer_signature(X_test, y_pred)
                mlflow.lightgbm.log_model(
                    model,
                    artifact_path=model_name,
                    signature=signature,
                    registered_model_name="lgbm-model-for-bank-churn",
                )
            else:
                mlflow.sklearn.log_model(model, model_name)
