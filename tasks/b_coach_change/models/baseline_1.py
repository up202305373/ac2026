from __future__ import annotations

import pandas as pd


def train_and_predict(
    train_data: pd.DataFrame, test_data: pd.DataFrame, *_args, **_kwargs
) -> pd.DataFrame:
    if train_data.empty or test_data.empty:
        raise ValueError("train_data and test_data must not be empty")

    required_columns = {"year", "tmID"}
    for name, data in (("train_data", train_data), ("test_data", test_data)):
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    filtered = test_data
    league_col = next(
        (col for col in ("lgID", "lgID_y", "lgID_x") if col in test_data.columns),
        None,
    )
    if league_col is not None:
        filtered = test_data.loc[test_data[league_col] == "NHL"]

    # Guarantee one unique row per (year, tmID) to satisfy validate="one_to_one"
    predictions = (
        filtered[["year", "tmID"]]
        .dropna(subset=["year", "tmID"])
        .drop_duplicates(subset=["year", "tmID"])
        .copy()
    )

    if predictions.empty:
        raise ValueError("No valid NHL team-seasons found in test_data")

    # Baseline prediction: No team changes coaches (0)
    predictions["predicted_change"] = 0

    return predictions[["year", "tmID", "predicted_change"]]