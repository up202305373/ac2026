from __future__ import annotations
from pathlib import Path
from scipy.stats import spearmanr
import logging
import numpy as np
import pandas as pd
log = logging.getLogger("a_ranking")

OUTPUT_DIR = Path(__file__).resolve().parents[0] / "plots"

def _year_spearman(frame: pd.DataFrame, left: str, right: str) -> float:
    """Mean per-season Spearman correlation between two aligned columns."""
    return frame.groupby("year").apply(
        lambda s: spearmanr(s[left], s[right]).statistic).mean()


def _seed_positions(df: pd.DataFrame, score_col: str, winner_col: str,
                    tiebreak_cols: tuple[str, ...]) -> pd.Series:
    """Map franchID to a league-wide position using conference seed rules.

    Division winners are ordered before other teams inside each conference.
    Each group is ordered by ``score_col`` and then ``tiebreak_cols``. The
    conference lists are then interleaved (E1, W1, E2, W2, ...) so the result
    is a single 1..N league-wide ranking.
    """
    out = df.copy()
    out["_conf"] = out["confID"].fillna("__NA__").astype(str)
    out["_conf_order"] = out["_conf"].map({"EC": 0, "WC": 1}).fillna(9)
    out["_conf_seed"] = np.nan

    sort_cols = [score_col, *tiebreak_cols]
    for conf in out["_conf"].unique():
        group = out[out["_conf"] == conf]
        winners = group[group[winner_col].eq(1)].sort_values(
            sort_cols, ascending=False)
        others = group[group[winner_col].eq(0)].sort_values(
            sort_cols, ascending=False)
        ordered = pd.concat([winners, others])
        out.loc[ordered.index, "_conf_seed"] = range(1, len(ordered) + 1)

    out = out.sort_values(["_conf_seed", "_conf_order", "franchID"],
                          ascending=[True, True, True])
    return pd.Series(range(1, len(out) + 1), index=out["franchID"].values)


def evaluate_ranking(predictions: pd.DataFrame, actual: pd.DataFrame, name: str, year: int, indent: str = "") -> dict:
    prediction_columns = {"year", "franchID", "predicted_pts"}
    actual_columns = {"year", "franchID", "Pts", "divID", "rank",
                      "confID", "W", "GF", "GA"}

    missing_predictions = prediction_columns - set(predictions.columns)
    if missing_predictions:
        raise ValueError(
            f"predictions is missing columns: {sorted(missing_predictions)}"
        )

    missing_actual = actual_columns - set(actual.columns)
    if missing_actual:
        raise ValueError(f"actual is missing columns: {sorted(missing_actual)}")

    actual = actual[["year", "franchID", "Pts", "divID", "rank",
                     "confID", "W", "GF", "GA"]].copy()
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

    # League-wide point order: the existing headline league-wide ranking.
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
    comparison["pos_error"] = (comparison["predicted_pos"] - comparison["actual_pos"]).abs()
    points_spearman = _year_spearman(comparison, "predicted_pos", "actual_pos")
    league_correct = int((comparison["predicted_pos"] == comparison["actual_pos"]).sum())
    league_mae = comparison["pos_error"].mean()

    points_error = comparison["predicted_pts"] - comparison["Pts"]
    points_mae = points_error.abs().mean()
    points_rmse = np.sqrt((points_error ** 2).mean())

    # Divisional positions, including the first place in each division.
    div = comparison[comparison["divID"].notna()].copy()
    div["predicted_div_pos"] = (
        div.groupby(["year", "divID"])["predicted_pts"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    div["div_pos_error"] = (div["predicted_div_pos"] - div["rank"]).abs()
    div_mean_pos_error = div["div_pos_error"].mean()
    div_spearman = _year_spearman(div, "predicted_div_pos", "rank")

    div_correct = int((div["predicted_div_pos"] == div["rank"].astype(int)).sum())
    winners = div[div["rank"] == 1]
    div_winners_correct = int((winners["predicted_div_pos"] == 1).sum())

    # Seed-style league order: division winners first inside each conference,
    # then points. This is reconstructed because the DB has no seed column.
    div["_gd"] = div["GF"] - div["GA"]
    div["pred_div_winner"] = div["predicted_div_pos"].eq(1).astype(int)
    div["actual_div_winner"] = div["rank"].eq(1).astype(int)
    pred_seed = _seed_positions(div, "predicted_pts", "pred_div_winner", ("franchID",))
    actual_seed = _seed_positions(
        div, "Pts", "actual_div_winner", ("W", "_gd", "GF", "franchID"))

    seed = div[["year", "franchID"]].copy()
    seed["predicted_seed_pos"] = seed["franchID"].map(pred_seed).astype(float)
    seed["actual_seed_pos"] = seed["franchID"].map(actual_seed).astype(float)
    seed["seed_pos_error"] = (
        seed["predicted_seed_pos"] - seed["actual_seed_pos"]).abs()
    seed_spearman = _year_spearman(seed, "predicted_seed_pos", "actual_seed_pos")
    seed_correct = int((seed["predicted_seed_pos"] == seed["actual_seed_pos"]).sum())
    seed_mae = seed["seed_pos_error"].mean()

    log.info(f"{indent}Divisional positions: {div_correct}/{len(div)} correct")
    log.info(f"{indent}Division winners: {div_winners_correct}/{len(winners)} correct")
    log.info(f"{indent}Mean absolute divisional position error: {div_mean_pos_error:.2f} spots")
    log.info(f"{indent}Divisional spearman correlation: {div_spearman:.3f}")
    log.info(f"{indent}League point-order positions: {league_correct}/{len(comparison)} correct")
    log.info(f"{indent}Mean absolute league point-order position error: {league_mae:.2f} spots")
    log.info(f"{indent}League point-order spearman correlation: {points_spearman:.3f}")
    log.info(f"{indent}League seed-style positions: {seed_correct}/{len(seed)} correct")
    log.info(f"{indent}Mean absolute league seed-style position error: {seed_mae:.2f} spots")
    log.info(f"{indent}League seed-style spearman correlation: {seed_spearman:.3f}")
    log.info(f"{indent}Points MAE: {points_mae:.2f} (RMSE {points_rmse:.2f})")

    return {
        "spearman": points_spearman,
        "league_pts_spearman": points_spearman,
        "league_correct": league_correct,
        "league_total": len(comparison),
        "league_mae": league_mae,
        "league_seed_spearman": seed_spearman,
        "seed_correct": seed_correct,
        "seed_total": len(seed),
        "seed_mae": seed_mae,
        "div_spearman": div_spearman,
        "points_mae": points_mae,
        "points_rmse": points_rmse,
        "div_mean_pos_error": div_mean_pos_error,
        "div_correct": div_correct,
        "div_total": len(div),
        "div_winners_correct": div_winners_correct,
        "div_winners_total": len(winners),
        "comparison": comparison,
        "div": div,
        "seed": seed,
    }

HEADLINE_MODELS = ("baseline", "roster_base", "roster_all",
                   "roster_base_age", "roster_all_age")
ERROR_MODELS = ("roster_base", "roster_all")
ESTIMATOR_MODELS = ("roster_base", "roster_base_elasticnet", "roster_base_ols",
                    "roster_all", "roster_all_elasticnet", "roster_all_ols")


def _estimator_label(name: str) -> str:
    if name.endswith("_elasticnet"):
        return "ElasticNet"
    if name.endswith("_ols"):
        return "OLS"
    return "Ridge"

def _metric_frame(results_by_model: dict, names: list) -> pd.DataFrame:
    """Flatten per-year result dicts into a tidy metrics table."""
    rows = []
    for name in names:
        for r in results_by_model[name]:
            rows.append({
                "model": name,
                "year": int(r["year"]),
                "div_correct": r["div_correct"],
                "div_winners_correct": r["div_winners_correct"],
                "div_mae": r["div_mean_pos_error"],
                "div_spearman": r["div_spearman"],
                "league_pts_correct": r["league_correct"],
                "league_pts_mae": r["league_mae"],
                "league_pts_spearman": r["league_pts_spearman"],
                "league_seed_correct": r["seed_correct"],
                "league_seed_mae": r["seed_mae"],
                "league_seed_spearman": r["league_seed_spearman"],
                "points_mae": r["points_mae"],
                "points_rmse": r["points_rmse"],
            })
    return pd.DataFrame(rows)


def _plot_model_comparison(plt, metrics: pd.DataFrame, labels: list) -> None:
    panels = (
        (("div_correct",), "Divisional positions correct per season"),
        (("div_winners_correct",), "Division winners correct per season"),
        (("div_mae",), "Mean divisional position error (spots)"),
        (("points_mae", "points_rmse"), "Points error per season"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    grouped = metrics.groupby("model")
    for ax, (columns, title) in zip(axes.ravel(), panels):
        width = 0.8 / len(columns)
        for i, column in enumerate(columns):
            means = grouped[column].mean().reindex(labels)
            sds = grouped[column].std().reindex(labels).fillna(0)
            x = np.arange(len(labels)) + (i - (len(columns) - 1) / 2) * width
            ax.bar(x, means, width=width, yerr=sds, capsize=3,
                   color=colors[i], label=column)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set(title=title, ylim=(0, None))
        ax.legend(fontsize=8)
    fig.suptitle("Headline models over all test seasons (bars: mean, error bars: sd)", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

def _plot_per_year(plt, metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (column, title) in zip(axes, (
        ("div_mae", "Mean divisional position error (spots)"),
        ("points_mae", "Points MAE"),
    )):
        for name, group in metrics.groupby("model"):
            yearly = group.groupby("year")[column].mean()
            ax.plot(yearly.index, yearly.values, marker="o", markersize=4, label=name)
        ax.axvline(2004, color="red", linestyle=":", linewidth=1.5)
        ax.axvline(2005, color="gray", linestyle="--", linewidth=1.2)
        ax.set(title=title, xlabel="Season (starting year)", ylabel=column)
        ax.legend(fontsize=8)
    fig.suptitle("Per-season performance (dotted: 2004 lockout, dashed: 2005 held out)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_year_trends.png", dpi=150)
    plt.close(fig)

def _concat(results_by_model: dict, name: str, key: str) -> pd.DataFrame:
    return pd.concat([r[key] for r in results_by_model[name]], ignore_index=True)

def _plot_error_distributions(plt, results_by_model: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, name in enumerate(ERROR_MODELS):
        if not results_by_model.get(name):
            continue
        comparison = _concat(results_by_model, name, "comparison")
        div = _concat(results_by_model, name, "div")
        for column, ax in (("pos_error", axes[row, 0]), ("div_pos_error", axes[row, 1])):
            source = comparison if column == "pos_error" else div
            counts = source[column].value_counts().sort_index()
            ax.bar(counts.index, counts.values, width=1.0, edgecolor="black")
            ax.set_xticks(range(0, int(counts.index.max()) + 1))
            scope = "League-wide" if column == "pos_error" else "Divisional"
            ax.set(title=f"{name}: {scope} position error",
                   xlabel="Absolute position error", ylabel="Count")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_distributions.png", dpi=150)
    plt.close(fig)

def _plot_predictions(plt, results_by_model: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    for ax, name in zip(axes[0], ERROR_MODELS):
        if not results_by_model.get(name):
            continue
        comparison = _concat(results_by_model, name, "comparison")
        ax.scatter(comparison["Pts"], comparison["predicted_pts"], alpha=0.5, s=25)
        low = min(comparison["Pts"].min(), comparison["predicted_pts"].min())
        high = max(comparison["Pts"].max(), comparison["predicted_pts"].max())
        ax.plot([low, high], [low, high], "k--", linewidth=1)
        ax.set(title=name, xlabel="Actual points", ylabel="Predicted points")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)

def _plot_estimator_comparison(plt, metrics: pd.DataFrame, labels: list) -> None:
    panels = (
        ("div_correct", "Divisional positions correct / season"),
        ("div_winners_correct", "Division winners correct / season"),
        ("div_mae", "Divisional position error (spots)"),
        ("points_mae", "Points MAE"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for row, prefix in enumerate(("roster_base", "roster_all")):
        names = [n for n in labels if n == prefix or n.startswith(prefix + "_")]
        short = [_estimator_label(n) for n in names]
        for col, (column, title) in enumerate(panels):
            ax = axes[row, col]
            grouped = metrics[metrics["model"].isin(names)].groupby("model")[column]
            means = grouped.mean().reindex(names)
            sds = grouped.std().reindex(names).fillna(0)
            x = np.arange(len(names))
            ax.bar(x, means, yerr=sds, capsize=3, color=colors[:len(names)])
            ax.set_xticks(x)
            ax.set_xticklabels(short)
            ax.set(title=f"{'base' if row == 0 else 'all'}: {title}", ylim=(0, None))
    fig.suptitle(
        "Estimator comparison: Ridge vs ElasticNet vs OLS (mean ± sd over test seasons)",
        y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "estimator_comparison.png", dpi=150)
    plt.close(fig)


def _plot_league_comparison(plt, metrics: pd.DataFrame, labels: list) -> None:
    panels = (
        (("league_pts_correct",), "League point-order positions correct / season"),
        (("league_pts_mae",), "League point-order position error (spots)"),
        (("league_seed_correct",), "Seed-style positions correct / season"),
        (("league_seed_mae",), "Seed-style position error (spots)"),
        (("league_pts_spearman", "league_seed_spearman"), "League Spearman: point order vs seed style"),
        (("div_spearman",), "Divisional Spearman"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    grouped = metrics.groupby("model")
    for ax, (columns, title) in zip(axes.ravel(), panels):
        width = 0.8 / len(columns)
        for i, column in enumerate(columns):
            means = grouped[column].mean().reindex(labels)
            sds = grouped[column].std().reindex(labels).fillna(0)
            x = np.arange(len(labels)) + (i - (len(columns) - 1) / 2) * width
            ax.bar(x, means, width=width, yerr=sds, capsize=3,
                   color=colors[i], label=column)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set(title=title, ylim=(0, None))
        ax.legend(fontsize=8)
    fig.suptitle("League-wide and divisional ranking metrics (bars: mean, error bars: sd)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "league_comparison.png", dpi=150)
    plt.close(fig)


def generate_graphs(results_by_model: dict) -> None:
    """Write the curated evaluation figures once, from completed results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = [n for n in HEADLINE_MODELS if results_by_model.get(n)]
    if not labels:
        return
    metrics = _metric_frame(results_by_model, labels)
    _plot_model_comparison(plt, metrics, labels)
    _plot_league_comparison(plt, metrics, labels)
    _plot_per_year(plt, metrics)
    _plot_error_distributions(plt, results_by_model)
    _plot_predictions(plt, results_by_model)

    est_labels = [n for n in ESTIMATOR_MODELS if results_by_model.get(n)]
    if est_labels:
        _plot_estimator_comparison(
            plt, _metric_frame(results_by_model, est_labels), est_labels)