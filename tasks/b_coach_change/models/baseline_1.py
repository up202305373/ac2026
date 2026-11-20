from __future__ import annotations

import pandas as pd

def train_and_predict(train_data: pd.DataFrame, test_data: pd.DataFrame, *_) -> pd.DataFrame:
    if train_data.empty or test_data.empty:
        raise ValueError("train_data and test_data must not be empty")

    required_columns = {"year", "tmID", "coachID", "stint"}
    for name, data in (("train_data", train_data), ("test_data", test_data)):
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    # Setup Test Predictions based on Starting Coaches (stint == 1)
    predictions = test_data[test_data["stint"] == 1][["year", "tmID", "coachID"]].copy()

    # Baseline prediction: No team changes coaches (0)
    predictions["predicted_change"] = 0

    return predictions[["year", "tmID", "predicted_change"]]