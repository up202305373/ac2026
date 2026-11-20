from __future__ import annotations

import pandas as pd


def train_and_predict(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    """Predict each player's PIM as his PIM from the previous season."""
    required_columns = {"year", "playerID", "PIM"}
    for name, data in (("train_data", train_data), ("test_data", test_data)):
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    previous = train_data[train_data["stint"] == 1]
    previous_pim = (
        previous[["year", "playerID", "PIM"]]
        .rename(columns={"year": "previous_year", "PIM": "predicted_pim"})
    )

    # Use the latest season available in training data.
    previous_year = int(train_data["year"].max())
    predictions = test_data[["year", "playerID"]].copy()
    predictions["previous_year"] = previous_year
    predictions = predictions.merge(
        previous_pim,
        on=["previous_year", "playerID"],
        how="left",
        validate="many_to_one",
    )

    return predictions[["year", "playerID", "predicted_pim"]]
