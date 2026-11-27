from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

log = logging.getLogger("c_pim")

OUTPUT_DIR = Path(__file__).resolve().parents[0] / "plots"


def evaluate_pim(predictions: pd.DataFrame, actual: pd.DataFrame, name: str, year: int, indent: str = "") -> dict:
    prediction_columns = {"year", "playerID", "predicted_pim"}
    actual_columns = {"year", "playerID", "PIM"}

    missing_predictions = prediction_columns - set(predictions.columns)
    if missing_predictions:
        raise ValueError(f"predictions is missing columns: {sorted(missing_predictions)}")

    missing_actual = actual_columns - set(actual.columns)
    if missing_actual:
        raise ValueError(f"actual is missing columns: {sorted(missing_actual)}")

    actual = actual[["year", "playerID", "PIM"]].copy()
    # A stint=1 row with no GP/PIM means the player did not play: zero PIM.
    actual["PIM"] = pd.to_numeric(actual["PIM"], errors="coerce").fillna(0.0)

    # Left join so players absent from the sampler's test rows are still scored.
    comparison = actual.merge(
        predictions[["year", "playerID", "predicted_pim"]],
        on=["year", "playerID"],
        how="left",
        validate="one_to_one",
    )
    comparison["predicted_pim"] = pd.to_numeric(
        comparison["predicted_pim"], errors="coerce").fillna(0.0)

    if comparison.empty:
        raise ValueError("No players to evaluate")

    comparison["error"] = comparison["predicted_pim"] - comparison["PIM"]
    comparison["abs_error"] = comparison["error"].abs()

    mae = comparison["abs_error"].mean()
    rmse = float(np.sqrt((comparison["error"] ** 2).mean()))
    spearman = spearmanr(comparison["predicted_pim"], comparison["PIM"]).statistic

    log.info(f"{indent}Players evaluated: {len(comparison)}")
    log.info(f"{indent}Mean absolute error: {mae:.2f} PIM")
    log.info(f"{indent}RMSE: {rmse:.2f} PIM")
    log.info(f"{indent}Spearman correlation: {spearman:.3f}")
    log.info(f"{indent}Mean actual PIM: {comparison['PIM'].mean():.2f}")
    log.info(f"{indent}Mean predicted PIM: {comparison['predicted_pim'].mean():.2f}")

    generate_graphs(comparison, name)

    return {
        "mae": mae,
        "rmse": rmse,
        "spearman": spearman,
        "evaluated": len(comparison),
        "total": len(actual),
        "comparison": comparison,
    }


def generate_graphs(comparison: pd.DataFrame, name: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(comparison["error"], bins=40, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title(f"{name}: PIM prediction error (predicted - actual)")
    ax.set_xlabel("Error (PIM)")
    ax.set_ylabel("Count")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / f"pim_error_{name}.png")
    plt.close(fig)
