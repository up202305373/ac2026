"""Simple evaluation for predicted team rankings."""

from __future__ import annotations

import pandas as pd


def evaluate_ranking(predictions: pd.DataFrame, actual: pd.DataFrame) -> None:
    """Print the number of predictions that exactly match the actual rank.

    Args:
        predictions: Dataframe containing ``tmID`` and ``predicted_rank``.
        actual: Dataframe containing ``tmID`` and ``rank``.

    Raises:
        ValueError: If required columns are missing or no teams can be
            evaluated.
    """
    prediction_columns = {"tmID", "predicted_rank"}
    actual_columns = {"tmID", "rank"}

    missing_predictions = prediction_columns - set(predictions.columns)
    if missing_predictions:
        raise ValueError(
            f"predictions is missing columns: {sorted(missing_predictions)}"
        )

    missing_actual = actual_columns - set(actual.columns)
    if missing_actual:
        raise ValueError(f"actual is missing columns: {sorted(missing_actual)}")

    comparison = predictions[["tmID", "predicted_rank"]].merge(
        actual[["tmID", "rank"]],
        on="tmID",
        how="inner",
        validate="one_to_one",
    )
    comparison = comparison.dropna(subset=["predicted_rank", "rank"])

    if comparison.empty:
        raise ValueError("No teams with both predicted and actual ranks")

    correct = (comparison["predicted_rank"] == comparison["rank"]).sum()
    print(f"{correct}/{len(comparison)} correct")