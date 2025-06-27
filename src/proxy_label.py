import pandas as pd
from sklearn.cluster import KMeans
from datetime import datetime

def create_rfm_features(df, snapshot_date=None):
    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max()

    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    rfm = df.groupby("CustomerId").agg({
        "TransactionStartTime": lambda x: (snapshot_date - x.max()).days,
        "TransactionId": "count",
        "Amount": "sum"
    }).rename(columns={
        "TransactionStartTime": "Recency",
        "TransactionId": "Frequency",
        "Amount": "Monetary"
    }).reset_index()

    return rfm

def assign_risk_cluster(rfm_df, random_state=42):
    rfm_scaled = rfm_df[["Recency", "Frequency", "Monetary"]].copy()
    from sklearn.preprocessing import StandardScaler
    rfm_scaled = StandardScaler().fit_transform(rfm_scaled)

    kmeans = KMeans(n_clusters=3, random_state=random_state)
    rfm_df["Cluster"] = kmeans.fit_predict(rfm_scaled)

    # Identify the high-risk cluster (typically: high Recency, low Frequency/Monetary)
    risk_order = rfm_df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    high_risk_cluster = risk_order["Frequency"].idxmin()

    rfm_df["is_high_risk"] = (rfm_df["Cluster"] == high_risk_cluster).astype(int)

    return rfm_df[["CustomerId", "is_high_risk"]]





# Sample mock data
data = {
    "CustomerId": ["C1", "C1", "C2", "C3", "C3", "C3"],
    "TransactionId": ["T1", "T2", "T3", "T4", "T5", "T6"],
    "Amount": [100, 200, 50, 20, 30, 25],
    "TransactionStartTime": [
        "2025-06-01", "2025-06-15", "2025-05-20",
        "2025-01-10", "2025-02-10", "2025-03-10"
    ]
}

df = pd.DataFrame(data)
rfm = create_rfm_features(df)
risk_df = assign_risk_cluster(rfm)

print(risk_df)
