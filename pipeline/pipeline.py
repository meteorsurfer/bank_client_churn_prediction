import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .preprocess import DataPreprocessor
from .logger import MLflowLogger
from .trainer import ModelTrainer

class ChurnPipeline:
    def __init__(self, data_path, tracking_uri):
        self.preprocessor = DataPreprocessor(data_path)
        self.logger = MLflowLogger(tracking_uri, "Customer_Churn_Baseline")

    def run(self):
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess()
        X_train_scaled, X_test_scaled = self.preprocessor.scale(X_train, X_test)
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)

        models = {
            "LogReg": LogisticRegression(max_iter=500, class_weight="balanced"),
            "kNN": KNeighborsClassifier(),
            "SVM": SVC(probability=True, class_weight="balanced"),
            "RF": RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
            "LGBM": LGBMClassifier(random_state=42, class_weight="balanced", verbose=0),
            "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
            "CatBoost": CatBoostClassifier(class_weights=class_weights.tolist(), random_state=42, verbose=0),
        }

        for name, model in models.items():
            print(f"\nTraining and evaluating: {name}")
            trainer = ModelTrainer(name, model, class_weights)
            y_pred = trainer.train(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled)
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            self.logger.log(name, model, X_test, y_pred, report_dict)

        print("Training complete. All runs logged to server.")
