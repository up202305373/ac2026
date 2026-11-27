from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots in scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

log = logging.getLogger("b_coach_change")

METRICS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1_score", "F1 Score"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
]


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

    has_prob = "change_probability" in predictions.columns
    pred_cols = ["year", "tmID", "predicted_change"] + (
        ["change_probability"] if has_prob else []
    )

    comparison = predictions[pred_cols].merge(
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
    y_score = (
        comparison["change_probability"].astype(float).fillna(y_pred.astype(float))
        if has_prob
        else y_pred.astype(float)
    )

    accuracy = accuracy_score(y_true, y_pred)
    if y_true.nunique() >= 2:
        roc_auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    log.info("=== Coach Change Prediction ===")
    log.info(f"Accuracy: {accuracy:.2f}")
    log.info(
        f"ROC-AUC: {roc_auc:.2f} | PR-AUC: {pr_auc:.2f}"
        if y_true.nunique() >= 2
        else "ROC-AUC: N/A | PR-AUC: N/A (single class in split)"
    )

    # Force a 2x2 matrix even if only one class is present in the split
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    log.info("Confusion Matrix:\n(Rows: Actual 0/1 | Columns: Predicted 0/1)\n" + str(cm))

    target_names = ["Kept Coach (0)", "Fired/Changed Coach (1)"]

    # Map the labels explicitly and handle undefined metrics
    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=target_names,
        zero_division=0.0,
    )
    log.info("Classification Report:\n" + report_text)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=target_names,
        zero_division=0.0,
        output_dict=True,
    )
    changed_stats = report_dict["Fired/Changed Coach (1)"]

    return {
        "accuracy": float(accuracy),
        "precision": float(changed_stats["precision"]),
        "recall": float(changed_stats["recall"]),
        "f1_score": float(changed_stats["f1-score"]),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "total_evaluated": len(comparison),
    }


def _plot_metrics_vs_variable(
    results_df: pd.DataFrame,
    x_col: str,
    x_label: str,
    title: str,
    model_colors: dict,
    output_path: Path,
    use_markers: bool = False,
) -> Path:
    nrows = math.ceil(len(METRICS) / 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4.2 * nrows), sharex=True, squeeze=False)
    axes_flat = axes.ravel()
    models = list(model_colors.keys())
    x_ticks = sorted(results_df[x_col].unique())

    for ax, (metric_col, metric_label) in zip(axes_flat, METRICS):
        for model in models:
            sub = results_df[results_df["model"] == model]
            stats = (
                sub.groupby(x_col)[metric_col]
                .agg(["mean", "std"])
                .sort_index()
            )
            mean = stats["mean"]
            std = stats["std"].fillna(0.0)

            ax.plot(
                stats.index,
                mean,
                label=model,
                color=model_colors[model],
                linewidth=1.8,
                marker="o" if use_markers else None,
                markersize=4 if use_markers else None,
            )
            ax.fill_between(
                stats.index,
                (mean - std).clip(0.0, 1.0),
                (mean + std).clip(0.0, 1.0),
                color=model_colors[model],
                alpha=0.12,
            )

        ax.set_title(metric_label)
        ax.set_ylabel("Score")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.4)
        if use_markers:
            ax.set_xticks(x_ticks)

    for ax in axes[-1]:
        ax.set_xlabel(x_label)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(models), 4),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_year_by_diff_grid(
    results_df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    model_colors: dict,
    output_path: Path,
) -> Path:
    test_years = sorted(results_df["test_year"].unique())
    models = list(model_colors.keys())
    ncols = min(3, len(test_years))
    nrows = math.ceil(len(test_years) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * ncols, 4 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, year in zip(axes_flat, test_years):
        year_df = results_df[results_df["test_year"] == year]
        for model in models:
            sub = year_df[year_df["model"] == model].sort_values("diff")
            ax.plot(
                sub["diff"],
                sub[metric_col],
                label=model,
                color=model_colors[model],
                linewidth=1.5,
            )
        ax.set_title(f"Test Year: {year}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Hide unused subplot panels if test_years doesn't fill the grid
    for ax in axes_flat[len(test_years):]:
        ax.set_visible(False)

    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Training Window (diff years)")
    for row in axes:
        row[0].set_ylabel(metric_label)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(models), 4),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.suptitle(
        f"Model Comparison: {metric_label} vs. Training Window by Test Year",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_evaluation_graphs(results_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Generate comparison charts across models, metrics, diff, and test_year."""
    if results_df.empty:
        log.warning("No evaluation results to plot.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(results_df["model"].unique())
    cmap = plt.get_cmap("tab10")
    model_colors = {model: cmap(i % 10) for i, model in enumerate(models)}

    saved_paths = [
        _plot_metrics_vs_variable(
            results_df=results_df,
            x_col="diff",
            x_label="Training Window (diff years)",
            title="Model Comparison Across Training Window (Mean ± 1 Std across Test Years)",
            model_colors=model_colors,
            output_path=output_dir / "comparison_by_diff.png",
            use_markers=False,
        ),
        _plot_metrics_vs_variable(
            results_df=results_df,
            x_col="test_year",
            x_label="Test Year",
            title="Model Comparison Across Test Years (Mean ± 1 Std across diff)",
            model_colors=model_colors,
            output_path=output_dir / "comparison_by_test_year.png",
            use_markers=True,
        ),
        _plot_year_by_diff_grid(
            results_df=results_df,
            metric_col="accuracy",
            metric_label="Accuracy",
            model_colors=model_colors,
            output_path=output_dir / "accuracy_by_year_and_diff.png",
        ),
    ]

    for path in saved_paths:
        log.info(f"Saved plot: {path}")

    return saved_paths


def plot_flag_fraction_sweep(
    actual_by_year: dict[int, pd.DataFrame],
    output_dir: Path,
    values: list[int] | None = None,
) -> tuple[pd.DataFrame, Path | None]:
    """Evaluate and plot model_14 sensitivity across candidate BASE_FLAG_FRACTION values."""
    from models import model_14_2

    if values is None:
        values = list(range(0, 101, 5))

    shared_years = sorted(set(model_14_2._SWEEP_CACHE).intersection(actual_by_year))
    if not shared_years:
        log.warning("No cached model_14 splits available for BASE_FLAG_FRACTION sweep.")
        return pd.DataFrame(), None

    rows = []
    for pct in values:
        base_frac = pct / 100.0
        accs, macro_f1s, c1_f1s, c1_precs, c1_recs, flagged_rates = [], [], [], [], [], []

        for year in shared_years:
            actual_df = actual_by_year[year]
            if actual_df.empty:
                continue

            preds = model_14_2.predict_for_base_fraction(year, base_frac)
            actual = actual_by_year[year][["year", "tmID", "Coach_Changed"]].copy()
            actual["Coach_Changed"] = actual["Coach_Changed"].astype(int)

            comp = preds.merge(actual, on=["year", "tmID"], how="inner", validate="one_to_one")
            y_true = comp["Coach_Changed"].to_numpy(dtype=int)
            y_pred = comp["predicted_change"].to_numpy(dtype=int)

            accs.append(accuracy_score(y_true, y_pred))
            macro_f1s.append(f1_score(y_true, y_pred, average="macro", zero_division=0))
            c1_f1s.append(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
            c1_precs.append(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
            c1_recs.append(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
            flagged_rates.append(float(y_pred.mean()))

        rows.append(
            {
                "base_pct": pct,
                "base_flag_fraction": base_frac,
                "actual_flagged_rate": float(np.mean(flagged_rates)),
                "accuracy": float(np.mean(accs)),
                "macro_f1": float(np.mean(macro_f1s)),
                "class1_f1": float(np.mean(c1_f1s)),
                "class1_precision": float(np.mean(c1_precs)),
                "class1_recall": float(np.mean(c1_recs)),
            }
        )

    sweep_df = pd.DataFrame(rows)
    log.info("\n=== BASE_FLAG_FRACTION Sweep (Mean Across Test Years) ===")
    log.info(sweep_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    best_macro = sweep_df.loc[sweep_df["macro_f1"].idxmax()]
    best_c1 = sweep_df.loc[sweep_df["class1_f1"].idxmax()]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "flag_fraction_sweep.png"

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.plot(sweep_df["base_pct"], sweep_df["accuracy"], marker="o", linewidth=2, label="Mean Accuracy")
    ax.plot(sweep_df["base_pct"], sweep_df["macro_f1"], marker="s", linewidth=2, label="Mean Macro F1")
    ax.plot(
        sweep_df["base_pct"],
        sweep_df["class1_f1"],
        marker="^",
        linewidth=2.2,
        label="Mean Class 1 F1 (Coach Change)",
    )
    ax.plot(
        sweep_df["base_pct"],
        sweep_df["class1_precision"],
        linestyle="--",
        alpha=0.75,
        label="Class 1 Precision",
    )
    ax.plot(
        sweep_df["base_pct"],
        sweep_df["class1_recall"],
        linestyle=":",
        alpha=0.75,
        label="Class 1 Recall",
    )

    ax.axvline(best_macro["base_pct"], color="tab:orange", linestyle="--", alpha=0.6)
    ax.annotate(
        f"Best Macro F1: {best_macro['macro_f1']:.3f} (@ {int(best_macro['base_pct'])}%)",
        xy=(best_macro["base_pct"], best_macro["macro_f1"]),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=9,
    )

    ax.axvline(best_c1["base_pct"], color="tab:green", linestyle="--", alpha=0.6)
    ax.annotate(
        f"Best Class 1 F1: {best_c1['class1_f1']:.3f} (@ {int(best_c1['base_pct'])}%)",
        xy=(best_c1["base_pct"], best_c1["class1_f1"]),
        xytext=(6, -14),
        textcoords="offset points",
        fontsize=9,
    )

    ax.set_title("Model 14 Sensitivity to BASE_FLAG_FRACTION (Mean Across Test Years)")
    ax.set_xlabel("BASE_FLAG_FRACTION (%)")
    ax.set_ylabel("Mean Score")
    ax.set_xticks(values)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Saved plot: {output_path}")
    return sweep_df, output_path