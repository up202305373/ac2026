"""Shared dynamic thresholding and ensembling utilities for coach-change models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score

Strategy = Literal[
    "fixed",
    "implied_count",
    "era_base_rate",
    "walk_forward_f1",
    "walk_forward_accuracy",
]


def _clamp_bounds(
    n_teams: int, min_flags: int = 3, max_fraction: float = 0.60
) -> tuple[int, int]:
    if n_teams <= 1:
        return 0, n_teams
    low = min(min_flags, n_teams - 1)
    high = max(low, min(n_teams - 1, int(round(max_fraction * n_teams))))
    return low, high


def flag_top_k(
    probabilities: np.ndarray,
    k: float,
    min_flags: int = 3,
    max_fraction: float = 0.60,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    n_teams = len(probs)
    low, high = _clamp_bounds(n_teams, min_flags, max_fraction)
    n_flags = max(low, min(high, int(round(k))))
    predicted = np.zeros(n_teams, dtype=int)
    if n_flags > 0:
        predicted[np.argsort(-probs)[:n_flags]] = 1
    return predicted


def flag_by_threshold(
    probabilities: np.ndarray,
    threshold: float,
    min_flags: int = 3,
    max_fraction: float = 0.60,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    n_teams = len(probs)
    low, high = _clamp_bounds(n_teams, min_flags, max_fraction)
    raw_flags = probs >= threshold
    count = int(raw_flags.sum())
    if count < low:
        return flag_top_k(probs, low, min_flags, max_fraction)
    if count > high:
        return flag_top_k(probs, high, min_flags, max_fraction)
    return raw_flags.astype(int)


def tune_walk_forward_threshold(
    train_rows: pd.DataFrame,
    features: list[str],
    model_factory: Callable[[], BaseEstimator],
    metric: Literal["f1", "accuracy"] = "f1",
    val_years: int = 15,
    min_train_seasons: int = 10,
    lookback_years: int = 45,
    min_flags: int = 3,
    max_fraction: float = 0.60,
) -> float | None:
    """Tune a probability cutoff using expanding out-of-time historical seasons."""
    years = sorted(train_rows["year"].dropna().unique())
    if len(years) <= min_train_seasons:
        return None

    candidate_years = years[min_train_seasons:][-val_years:]
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for val_year in candidate_years:
        fold_train = train_rows[
            (train_rows["year"] < val_year)
            & (train_rows["year"] >= max(1960, int(val_year) - lookback_years))
        ]
        fold_val = train_rows[train_rows["year"] == val_year]
        if fold_val.empty or fold_train["Coach_Changed"].nunique() < 2:
            continue

        model = model_factory()
        model.fit(
            fold_train[features].astype(float),
            fold_train["Coach_Changed"].astype(int),
        )
        probs = model.predict_proba(fold_val[features].astype(float))[:, 1]
        folds.append((fold_val["Coach_Changed"].astype(int).to_numpy(), probs))

    if not folds:
        return None

    y_all = np.concatenate([y for y, _ in folds])
    base_rate = float(y_all.mean())
    thresholds = np.linspace(0.15, 0.75, 61)

    best_score = -1.0
    best_threshold = base_rate
    for threshold in thresholds:
        pred_all = np.concatenate(
            [
                flag_by_threshold(probs, float(threshold), min_flags, max_fraction)
                for _, probs in folds
            ]
        )
        if metric == "f1":
            score = float(f1_score(y_all, pred_all, zero_division=0.0))
        else:
            score = float(accuracy_score(y_all, pred_all))

        # Tie-break toward the empirical base rate to avoid extreme thresholds
        is_better = score > best_score or (
            np.isclose(score, best_score)
            and abs(threshold - base_rate) < abs(best_threshold - base_rate)
        )
        if is_better:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def select_coach_changes(
    probabilities: np.ndarray,
    train_rows: pd.DataFrame,
    strategy: Strategy = "implied_count",
    *,
    features: list[str] | None = None,
    model_factory: Callable[[], BaseEstimator] | None = None,
    fixed_fraction: float = 0.34,
    era_window: int = 15,
    val_years: int = 15,
    min_flags: int = 3,
    max_fraction: float = 0.60,
) -> tuple[np.ndarray, dict[str, float | str | None]]:
    """Convert test-season probabilities into binary flags using the chosen strategy."""
    probs = np.asarray(probabilities, dtype=float)
    n_teams = len(probs)

    if strategy == "fixed":
        predicted = flag_top_k(probs, fixed_fraction * n_teams, min_flags, max_fraction)
        return predicted, {
            "strategy": strategy,
            "effective_fraction": float(predicted.mean()) if n_teams else 0.0,
            "threshold": None,
        }

    if strategy == "implied_count":
        expected_changes = float(np.sum(probs))
        predicted = flag_top_k(probs, expected_changes, min_flags, max_fraction)
        return predicted, {
            "strategy": strategy,
            "effective_fraction": float(predicted.mean()) if n_teams else 0.0,
            "threshold": None,
        }

    if strategy == "era_base_rate":
        recent_years = sorted(train_rows["year"].dropna().unique())[-era_window:]
        rate = float(
            train_rows.loc[
                train_rows["year"].isin(recent_years), "Coach_Changed"
            ].mean()
        )
        predicted = flag_top_k(probs, rate * n_teams, min_flags, max_fraction)
        return predicted, {
            "strategy": strategy,
            "effective_fraction": float(predicted.mean()) if n_teams else 0.0,
            "threshold": None,
        }

    if strategy in ("walk_forward_f1", "walk_forward_accuracy"):
        if features is None or model_factory is None:
            raise ValueError(
                "features and model_factory are required for walk-forward strategies"
            )
        metric: Literal["f1", "accuracy"] = (
            "f1" if strategy == "walk_forward_f1" else "accuracy"
        )
        threshold = tune_walk_forward_threshold(
            train_rows=train_rows,
            features=features,
            model_factory=model_factory,
            metric=metric,
            val_years=val_years,
            min_flags=min_flags,
            max_fraction=max_fraction,
        )
        if threshold is None:
            # Safe fallback when history is too short for walk-forward folds
            return select_coach_changes(
                probs,
                train_rows,
                strategy="implied_count",
                min_flags=min_flags,
                max_fraction=max_fraction,
            )
        predicted = flag_by_threshold(probs, threshold, min_flags, max_fraction)
        return predicted, {
            "strategy": strategy,
            "effective_fraction": float(predicted.mean()) if n_teams else 0.0,
            "threshold": threshold,
        }

    raise ValueError(f"Unknown strategy: {strategy}")


def blend_predictions(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    *,
    weight_a: float = 0.5,
    threshold: float | None = None,
    fixed_fraction: float | None = None,
    min_flags: int = 3,
    max_fraction: float = 0.60,
) -> pd.DataFrame:
    """Blend probabilities from two models and convert them into binary flags.

    Defaults to the model-implied count (`sum(blended_prob)`) when neither
    `threshold` nor `fixed_fraction` is provided.
    """
    merged = pred_a[["year", "tmID", "change_probability"]].merge(
        pred_b[["year", "tmID", "change_probability"]],
        on=["year", "tmID"],
        how="inner",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    blended_prob = (
        weight_a * merged["change_probability_a"].to_numpy(dtype=float)
        + (1.0 - weight_a) * merged["change_probability_b"].to_numpy(dtype=float)
    )

    if threshold is not None:
        predicted = flag_by_threshold(blended_prob, threshold, min_flags, max_fraction)
    elif fixed_fraction is not None:
        predicted = flag_top_k(
            blended_prob, fixed_fraction * len(blended_prob), min_flags, max_fraction
        )
    else:
        predicted = flag_top_k(
            blended_prob, float(np.sum(blended_prob)), min_flags, max_fraction
        )

    result = merged[["year", "tmID"]].copy()
    result["change_probability"] = blended_prob
    result["predicted_change"] = predicted
    return result