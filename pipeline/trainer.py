from sklearn.utils.class_weight import compute_sample_weight

class ModelTrainer:
    def __init__(self, model_name, model, class_weights):
        self.name = model_name
        self.model = model
        self.class_weights = class_weights

    def train(self, X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled):
        if self.name == "XGBoost":
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
            self.model.fit(X_train, y_train, sample_weight=sample_weights)
            y_pred = self.model.predict(X_test)
        elif self.name in ("SVM", "kNN"):
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
        else:
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

        return y_pred
