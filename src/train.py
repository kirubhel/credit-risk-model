import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.data_processing import build_pipeline
from src.proxy_label import create_rfm_features, assign_risk_cluster
from sklearn.pipeline import Pipeline

# Load and preprocess data
df = pd.read_csv("data/raw/data.csv")
rfm = create_rfm_features(df)
risk_df = assign_risk_cluster(rfm)

# Merge target with main data
df = pd.merge(df, risk_df, on="CustomerId", how="inner")

# Drop duplicates and keep the latest record per customer
df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
df = df.sort_values("TransactionStartTime").drop_duplicates("CustomerId", keep="last")

# Select features and target
X = df[["Amount", "Value", "ProductCategory", "ChannelId", "ProviderId", "PricingStrategy", "TransactionStartTime"]]
y = df["is_high_risk"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Build pipeline
pipeline = build_pipeline()

# Train and evaluate models
def train_and_log(model, model_name):
    with mlflow.start_run(run_name=model_name):
        # Build a fresh pipeline each time
        base_pipeline = build_pipeline()
        clf = Pipeline(steps=base_pipeline.steps + [("classifier", model)])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("f1", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))
        mlflow.sklearn.log_model(clf, model_name)


# Run models
train_and_log(LogisticRegression(max_iter=1000), "LogisticRegression")
train_and_log(GradientBoostingClassifier(), "GradientBoosting")
