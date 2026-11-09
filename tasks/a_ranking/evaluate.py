from __future__ import annotations
import logging
import pandas as pd
log = logging.getLogger("a_ranking")

def evaluate_ranking(predictions: pd.DataFrame, actual: pd.DataFrame) -> dict:
    prediction_columns = {"year", "franchID", "predicted_pts"}
    actual_columns = {"year", "franchID", "Pts", "divID", "rank"}

    missing_predictions = prediction_columns - set(predictions.columns)
    if missing_predictions:
        raise ValueError(
            f"predictions is missing columns: {sorted(missing_predictions)}"
        )

    missing_actual = actual_columns - set(actual.columns)
    if missing_actual:
        raise ValueError(f"actual is missing columns: {sorted(missing_actual)}")

    actual = actual[["year", "franchID", "Pts", "divID", "rank"]].copy()
    actual["Pts"] = actual["Pts"].astype(float)
    actual["rank"] = actual["rank"].astype(float)

    comparison = predictions[["year", "franchID", "predicted_pts"]].merge(
        actual,
        on=["year", "franchID"],
        how="inner",
        validate="one_to_one",
    )
    comparison = comparison.dropna(subset=["predicted_pts", "Pts"])

    if comparison.empty:
        raise ValueError("No teams with both predicted and actual points")

    # League-wide positions: both sides derived from points.
    comparison["predicted_pos"] = (
        comparison.groupby("year")["predicted_pts"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    comparison["actual_pos"] = (
        comparison.groupby("year")["Pts"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    league_correct = int((comparison["predicted_pos"] == comparison["actual_pos"]).sum())
    log.info(f"League positions: {league_correct}/{len(comparison)} correct")

    # Divisional positions: only team-seasons with a division can be compared.
    div = comparison[comparison["divID"].notna()].copy()
    div["predicted_div_pos"] = (
        div.groupby(["year", "divID"])["predicted_pts"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    div_correct = int((div["predicted_div_pos"] == div["rank"].astype(int)).sum())
    log.info(f"Divisional positions: {div_correct}/{len(div)} correct")

    return {
        "league_correct": league_correct,
        "league_total": len(comparison),
        "div_correct": div_correct,
        "div_total": len(div),
    }