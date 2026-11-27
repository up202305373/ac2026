from __future__ import annotations

import pandas as pd


def train_and_predict(train_data: pd.DataFrame, test_data: pd.DataFrame, *_, **__) -> pd.DataFrame:
    """Predict each player's PIM as his total PIM from the previous season."""
    required_columns = {"year", "playerID", "PIM"}
    for name, data in (("train_data", train_data), ("test_data", test_data)):
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    # Last season's stints are observed, so use the player's full-season total.
    previous = train_data[["year", "playerID", "PIM"]].copy()
    previous["PIM"] = pd.to_numeric(previous["PIM"], errors="coerce")
    previous_pim = (
        previous.groupby(["year", "playerID"], as_index=False)["PIM"]
        .sum()
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

    # A player with no prior-season row is treated as having zero prior PIM.
    predictions["predicted_pim"] = pd.to_numeric(
        predictions["predicted_pim"], errors="coerce").fillna(0.0)
    return predictions[["year", "playerID", "predicted_pim"]]
