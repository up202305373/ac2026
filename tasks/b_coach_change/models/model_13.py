"""Coach-change prediction model 13 (stacked rank ensemble with a decision layer).

Key improvements
----------------
* Reuses the complementary signals of the strongest existing models instead of
  training another tiny 2-season classifier from scratch:
  - ``model_2``  : high-recall LR + random-forest ensemble (best prior F1 / ROC / PR).
  - ``model_3``  : same family with a better-balanced dynamic rank threshold.
  - ``model_12`` : precision/accuracy specialist with the evaluation-aligned label
                   and a short-window heuristic fallback (also covers 2005).
  - ``baseline_3``: prior-season standings heuristic that wins in low-signal years.
  - ``baseline_2``: repeat-firing rule, used only as a last-resort fallback.
* Converts each component's probability into a within-season percentile rank so
  models with very different probability scales can be combined safely.
* Keeps two explicit layers:
  - a **risk score** (0.7 * model_2 + 0.3 * model_12 rank) used for the reported
    ``change_probability``, preserving the best ranking/PR-AUC signal;
  - a **decision score** that adds a small agreement bonus when model_2 and
    model_12 both rank a team high, plus model_3 and standings tie-breakers.
    This recovers precision without giving up model_2's recall.
* Flags a fixed 68% of teams (clamped to at least 3), a recall-oriented
  operating point that still keeps precision above every earlier model.
* Is robust to component failures (e.g. model_2/model_3 have no training rows in
  2005) by renormalizing over the components that are available.

The public contract matches every other model in this task directory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models import baseline_2, baseline_3, model_2, model_3, model_12

# Component names are kept in priority order for fallback purposes.
COMPONENTS = (
    ("model_2", model_2.train_and_predict),
    ("model_3", model_3.train_and_predict),
    ("model_12", model_12.train_and_predict),
    ("baseline_3", baseline_3.train_and_predict),
    ("baseline_2", baseline_2.train_and_predict),
)

RISK_WEIGHTS = {"model_2": 0.7, "model_12": 0.3}
DECISION_WEIGHTS = {
    "model_2": 0.6,
    "model_12": 0.3,
    "model_3": 0.1,
    "baseline_3": 0.1,
}
AGREEMENT_PRODUCT_WEIGHT = 0.4
AGREEMENT_MIN_WEIGHT = 0.2
FLAG_FRACTION = 0.68
MIN_FLAGS = 3
MAX_FLAG_FRACTION = 0.99
PROBABILITY_SLOPE = 4.0


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Within-split percentile rank; missing values become neutral 0.5."""
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rank(method="average", pct=True).fillna(0.5).to_numpy(dtype=float)


def _component_probabilities(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    kwargs: dict,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Run available component models and align their probabilities to test teams."""
    base = test_data.copy()
    league_col = next((c for c in ("lgID", "lgID_y", "lgID_x") if c in base.columns), None)
    if league_col is not None:
        base = base[base[league_col] == "NHL"]
    base = base[["year", "tmID"]].drop_duplicates(["year", "tmID"]).reset_index(drop=True)

    ranks: dict[str, np.ndarray] = {}
    n = len(base)
    for name, train_and_predict in COMPONENTS:
        try:
            clean_kwargs = {
                key: value.copy() if isinstance(value, pd.DataFrame) else value
                for key, value in kwargs.items()
            }
            prediction = train_and_predict(
                train_data=train_data.copy(),
                test_data=test_data.copy(),
                **clean_kwargs,
            )
            if "change_probability" not in prediction.columns:
                continue
            aligned = base.merge(
                prediction[["year", "tmID", "change_probability"]].drop_duplicates(
                    ["year", "tmID"]
                ),
                on=["year", "tmID"],
                how="left",
                validate="one_to_one",
            )["change_probability"]
            if aligned.notna().sum() < n:
                # A component that cannot score every team is not trustworthy here.
                continue
            ranks[name] = _percentile_rank(aligned.to_numpy(dtype=float))
        except Exception:  # noqa: BLE001 - a failed component must not break the ensemble
            continue
    return base, ranks


def _risk_score(ranks: dict[str, np.ndarray], n_teams: int) -> np.ndarray:
    """Probability-layer score: model_2 recall kept, model_12 precision added."""
    if "model_2" not in ranks:
        for name in ("model_12", "model_3", "baseline_3", "baseline_2"):
            if name in ranks:
                return ranks[name]
        return np.full(n_teams, 0.5)

    r2 = ranks["model_2"]
    r12 = ranks.get("model_12", np.full(n_teams, 0.5))
    return RISK_WEIGHTS["model_2"] * r2 + RISK_WEIGHTS["model_12"] * r12


def _decision_score(ranks: dict[str, np.ndarray], n_teams: int) -> np.ndarray:
    """Decision-layer score: weighted ranks plus an explicit agreement bonus."""
    if "model_2" not in ranks:
        for name in ("model_12", "model_3", "baseline_3", "baseline_2"):
            if name in ranks:
                return ranks[name]
        return np.full(n_teams, 0.5)

    neutral = np.full(n_teams, 0.5)
    r2 = ranks["model_2"]
    r12 = ranks.get("model_12", neutral)
    r3 = ranks.get("model_3", neutral)
    rb = ranks.get("baseline_3", neutral)

    active = {
        "model_2": r2,
        "model_12": r12,
        "model_3": r3,
        "baseline_3": rb,
    }
    weight_sum = sum(DECISION_WEIGHTS[name] for name in active)
    base = sum(
        DECISION_WEIGHTS[name] * values for name, values in active.items()
    ) / weight_sum

    # Teams rated highly by both the recall model and the precision model get a
    # boost; this is what converts model_2's broad recall into usable precision.
    return (
        base
        + AGREEMENT_PRODUCT_WEIGHT * r2 * r12
        + AGREEMENT_MIN_WEIGHT * np.minimum(r2, r12)
    )


def _flag_top(scores: np.ndarray) -> np.ndarray:
    n_teams = len(scores)
    if n_teams == 0:
        return np.zeros(0, dtype=int)
    k = max(MIN_FLAGS, int(round(FLAG_FRACTION * n_teams)))
    k = min(k, n_teams, int(np.ceil(MAX_FLAG_FRACTION * n_teams)))
    predicted = np.zeros(n_teams, dtype=int)
    if k > 0:
        predicted[np.argsort(-scores, kind="stable")[:k]] = 1
    return predicted


def train_and_predict(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    *args,
    **kwargs,
) -> pd.DataFrame:
    if test_data.empty or "tmID" not in test_data.columns:
        raise ValueError("test_data must contain tmID")

    base, ranks = _component_probabilities(train_data, test_data, kwargs)
    if base.empty:
        raise ValueError("No NHL teams to predict")

    n_teams = len(base)
    risk = _risk_score(ranks, n_teams)
    decision = _decision_score(ranks, n_teams)

    # A monotone logistic transform keeps ROC-AUC / PR-AUC identical to the
    # underlying risk ranking while producing values in (0, 1).
    probability = 1.0 / (1.0 + np.exp(-PROBABILITY_SLOPE * (risk - 0.5)))

    result = base.copy()
    result["change_probability"] = probability
    result["predicted_change"] = _flag_top(decision)
    result.attrs["cutoff_info"] = {
        "strategy": "stacked_rank_blend",
        "components": [name for name, _ in COMPONENTS if name in ranks],
        "flag_fraction": FLAG_FRACTION,
    }
    return result[["year", "tmID", "change_probability", "predicted_change"]]
