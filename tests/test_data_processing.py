import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing import TimeFeaturesExtractor


def test_time_features_extractor():
    df = pd.DataFrame({
        "TransactionStartTime": ["2025-06-01 10:00:00", "2025-06-02 15:00:00"]
    })

    transformer = TimeFeaturesExtractor()
    transformed_df = transformer.transform(df)

    assert "transaction_hour" in transformed_df.columns
    assert "transaction_day" in transformed_df.columns
    assert "transaction_month" in transformed_df.columns
    assert "TransactionStartTime" not in transformed_df.columns
    assert transformed_df.shape[1] == 3
