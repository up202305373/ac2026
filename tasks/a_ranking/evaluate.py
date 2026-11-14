from __future__ import annotations
from pathlib import Path
from scipy.stats import spearmanr
import logging
import pandas as pd
log = logging.getLogger("a_ranking")

OUTPUT_DIR = Path(__file__).resolve().parents[0] / "plots"

def evaluate_ranking(predictions: pd.DataFrame, actual: pd.DataFrame, name: str, year: int) -> dict:
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

    # League-wide positions
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

    spearman = comparison.groupby("year").apply(lambda s: spearmanr(s["predicted_pos"], s["actual_pos"]).statistic).mean()
    comparison["pos_error"] = (comparison["predicted_pos"] - comparison["actual_pos"]).abs()
    mean_pos_error = comparison["pos_error"].mean()

    log.info(f"League spearman correlation: {spearman:.3f}")
    log.info(f"League mean absolute position error: {mean_pos_error:.2f} spots")

    # Divisional positions
    div = comparison[comparison["divID"].notna()].copy()
    div["predicted_div_pos"] = (
        div.groupby(["year", "divID"])["predicted_pts"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    div_correct = int((div["predicted_div_pos"] == div["rank"].astype(int)).sum())
    log.info(f"Divisional positions: {div_correct}/{len(div)} correct")

    div["div_pos_error"] = (div["predicted_div_pos"] - div["rank"]).abs()
    div_mean_pos_error = div["div_pos_error"].mean()

    log.info(f"Mean absolute divisional position error: {div_mean_pos_error:.2f} spots")

    generate_graphs(comparison, div, name, year)

    return {
        "league_correct": league_correct,
        "league_total": len(comparison),
        "spearman": spearman,
        "mean_pos_error": mean_pos_error,
        "div_mean_pos_error": div_mean_pos_error,
        "comparison": comparison,
        "div": div,
        "div_correct": div_correct,
        "div_total": len(div),
    }

def generate_graphs(comparison, div, name, year=None) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # League-wide position error distribution
    counts = comparison["pos_error"].value_counts().sort_index()
    axes[0].bar(counts.index, counts.values, width=1.0, edgecolor="black")
    axes[0].set_xticks(range(0, int(counts.index.max()) + 1))
    axes[0].set_title("League-wide Position Error Distribution")
    axes[0].set_xlabel("Absolute Position Error")
    axes[0].set_ylabel("Count")
    for p in axes[0].patches:
        axes[0].annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height() * 1.005), ha="center", va="bottom")

    # Divisional position error distribution
    div_counts = div["div_pos_error"].value_counts().sort_index()
    axes[1].bar(div_counts.index, div_counts.values, width=1.0, edgecolor="black")
    axes[1].set_xticks(range(0, int(div_counts.index.max()) + 1))
    axes[1].set_title("Divisional Position Error Distribution")
    axes[1].set_xlabel("Absolute Divisional Position Error")
    axes[1].set_ylabel("Count")
    for p in axes[1].patches:
        axes[1].annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height() * 1.005), ha="center", va="bottom")

    plt.tight_layout()
    if year is not None:
        #plt.savefig(OUTPUT_DIR / f"ped_overall_{name}_{year}.png")
        pass
    else:
        plt.savefig(OUTPUT_DIR / f"ped_overall_{name}.png")
    plt.close(fig)