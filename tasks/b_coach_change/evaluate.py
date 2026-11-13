from __future__ import annotations

import logging

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

log = logging.getLogger("b_coach_change")

def evaluate_coach_change(predictions: pd.DataFrame, actual: pd.DataFrame) -> dict:
    prediction_columns = {"year", "tmID", "predicted_change"}
    actual_columns = {"year", "tmID", "Coach_Changed"}

    missing_predictions = prediction_columns - set(predictions.columns)
    if missing_predictions:
        raise ValueError(
            f"predictions is missing columns: {sorted(missing_predictions)}"
        )

    missing_actual = actual_columns - set(actual.columns)
    if missing_actual:
        raise ValueError(f"actual is missing columns: {sorted(missing_actual)}")

    actual = actual[["year", "tmID", "Coach_Changed"]].copy()
    actual["Coach_Changed"] = actual["Coach_Changed"].astype(int)

    comparison = predictions[["year", "tmID", "predicted_change"]].merge(
        actual,
        on=["year", "tmID"],
        how="inner",
        validate="one_to_one",
    )
    comparison = comparison.dropna(subset=["predicted_change", "Coach_Changed"])

    if comparison.empty:
        raise ValueError("No teams with both predicted and actual coach changes")

    y_true = comparison["Coach_Changed"]
    y_pred = comparison["predicted_change"]

    accuracy = accuracy_score(y_true, y_pred)
    log.info("=== Coach Change Prediction ===")
    log.info(f"Accuracy: {accuracy:.2f}")

    # Force a 2x2 matrix even if only one class is present in the split
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    log.info("Confusion Matrix:\n(Rows: Actual 0/1 | Columns: Predicted 0/1)\n" + str(cm))

    # Map the labels explicitly and handle undefined metrics
    report = classification_report(
        y_true, 
        y_pred, 
        labels=[0, 1],
        target_names=["Kept Coach (0)", "Fired/Changed Coach (1)"],
        zero_division=0.0
    )
    log.info("Classification Report:\n" + report)

    return {
        "accuracy": float(accuracy),
        "total_evaluated": len(comparison),
    }