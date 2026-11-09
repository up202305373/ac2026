from __future__ import annotations

import pandas as pd

def train_and_predict(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    if train_data.empty:
        raise ValueError("train_data must not be empty")
    if test_data.empty:
        raise ValueError("test_data must not be empty")

    required_columns = {"year", "franchID", "Pts"}
    for name, data in (("train_data", train_data), ("test_data", test_data)):
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    previous_pts = (
        train_data[["year", "franchID", "Pts"]]
        .rename(columns={"year": "previous_year", "Pts": "predicted_pts"})
    )

    predictions = test_data[["year", "franchID"]].copy()
    predictions["previous_year"] = predictions["year"] - 1
    predictions = predictions.merge(
        previous_pts,
        on=["previous_year", "franchID"],
        how="left",
        validate="many_to_one",
    )

    return predictions[["year", "franchID", "predicted_pts"]]