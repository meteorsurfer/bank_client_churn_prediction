import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def preprocess(self):
        categoricals = self.df.select_dtypes(include=["object"]).columns
        for col in categoricals:
            self.df[col] = self.label_encoder.fit_transform(self.df[col])
        X = self.df.drop(columns=["exited"])
        y = self.df["exited"]
        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    def scale(self, X_train, X_test):
        return self.scaler.fit_transform(X_train), self.scaler.transform(X_test)

