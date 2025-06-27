# src/data_processing.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# 1. Custom Transformer for Time Features
class TimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, time_column='TransactionStartTime'):
        self.time_column = time_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.time_column] = pd.to_datetime(X[self.time_column])
        X['transaction_hour'] = X[self.time_column].dt.hour
        X['transaction_day'] = X[self.time_column].dt.day
        X['transaction_month'] = X[self.time_column].dt.month
        return X.drop(columns=[self.time_column])

# 2. Preprocessing Function
def build_pipeline():
    numerical_features = ['Amount', 'Value', 'transaction_hour', 'transaction_day', 'transaction_month']
    categorical_features = ['ProductCategory', 'ChannelId', 'ProviderId', 'PricingStrategy']

    # Pipelines
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return Pipeline([
        ("time_features", TimeFeaturesExtractor()),
        ("preprocessor", full_pipeline)
    ])

if __name__ == "__main__":
    import pandas as pd

    # Sample test data (mocked)
    data = {
        "Amount": [1000, -500, 200],
        "Value": [1000, 500, 200],
        "ProductCategory": ["Electronics", "Fashion", "Electronics"],
        "ChannelId": ["web", "app", "web"],
        "ProviderId": ["P1", "P2", "P1"],
        "PricingStrategy": [2, 1, 2],
        "TransactionStartTime": ["2025-01-01 10:00:00", "2025-01-02 12:00:00", "2025-01-03 08:00:00"]
    }

    df = pd.DataFrame(data)
    print("Original Data:")
    print(df)

    # Build and run pipeline
    pipeline = build_pipeline()
    X_processed = pipeline.fit_transform(df)

    print("\nTransformed Feature Matrix:")
    print(X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed)
