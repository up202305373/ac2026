"""Coach-change prediction model 13 (stacked rank ensemble with dynamic cutoff
and missing-season lookback).

Key improvements over v1
------------------------
* Missing-season lookback (e.g. 2004-05 NHL lockout before test_year=2005):
  Detects when ``test_year - 1`` has no played games, fetches the latest valid
  prior season ``y_prev``, shrinks extreme numeric stats toward the league
  mean to account for staleness, and bridges both ``train_data`` / auxiliary
  tables and ``test_data`` features so downstream models do not drop out.
* Dynamic operating-point estimation:
  Replaces the static 68% cutoff with a bounded estimate (36%–58%) derived
  from the raw probability mass of ``model_2`` + ``model_12`` and a
  mean-reverting adjustment from the previous season's observed turnover.
* Preserves the two-layer architecture (calibrated risk score for ROC/PR-AUC
  and agreement-boosted rank decision score for binary classification).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models import baseline_2, baseline_3, model_2, model_3, model_12

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

# Dynamic cutoff bounds (centered near the ~41% historical NHL turnover rate)
BASE_FLAG_FRACTION = 0.58
MIN_FLAG_FRACTION = 0.36
MAX_FLAG_FRACTION = 0.58
MIN_FLAGS = 3

# Staleness shrinkage applied when bridging across a canceled season (gap > 1)
STALENESS_SHRINKAGE = 0.80
PROBABILITY_SLOPE = 4.0

_SWEEP_CACHE: dict[int, dict] = {}

_ID_COLS = {"year", "tmID", "lgID", "lgID_x", "lgID_y", "franchID", "confID", "divID", "coachID", "playerID"}


def _filter_nhl(df: pd.DataFrame) -> pd.DataFrame:
    league_col = next((c for c in ("lgID", "lgID_y", "lgID_x") if c in df.columns), None)
    if league_col is not None:
        return df[df[league_col] == "NHL"].copy()
    return df.copy()


def _valid_seasons(df: pd.DataFrame, before_year: int) -> list[int]:
    """Return sorted NHL seasons strictly before ``before_year`` with actual games."""
    if df.empty or "year" not in df.columns:
        return []
    sub = _filter_nhl(df)
    sub = sub[sub["year"] < before_year]
    if sub.empty:
        return []
    if "G" in sub.columns:
        games_by_year = sub.groupby("year")["G"].sum(min_count=1)
        return sorted(int(y) for y, g in games_by_year.items() if pd.notna(g) and g > 0)
    return sorted(int(y) for y in sub["year"].dropna().unique())


def _shrink_season_stats(season_df: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Pull numeric performance columns toward the within-season mean."""
    out = season_df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        if col in _ID_COLS:
            continue
        col_mean = out[col].mean()
        if pd.notna(col_mean):
            out[col] = factor * out[col] + (1.0 - factor) * col_mean
    return out


def _bridge_frame_year(
    df: pd.DataFrame,
    source_year: int,
    target_year: int,
    shrinkage: float,
) -> pd.DataFrame:
    """Append a shrunk copy of ``source_year`` labeled as ``target_year`` if absent."""
    if df.empty or "year" not in df.columns:
        return df
    existing_target = df[df["year"] == target_year]
    if not existing_target.empty:
        games_col = next((c for c in ("G", "g") if c in existing_target.columns), None)
        if games_col is None or existing_target[games_col].fillna(0).sum() > 0:
            return df

    src = df[df["year"] == source_year].copy()
    if src.empty:
        return df

    bridged = _shrink_season_stats(src, shrinkage)
    bridged["year"] = target_year
    cleaned = df[df["year"] != target_year]
    return pd.concat([cleaned, bridged], ignore_index=True)


def _bridge_missing_prior_season(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    kwargs: dict,
    test_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """If ``test_year - 1`` is missing (e.g. 2004 lockout), bridge from ``y_prev``."""
    expected_prev = test_year - 1
    seasons = _valid_seasons(train_data, test_year)
    if not seasons or expected_prev in seasons:
        return train_data, test_data, kwargs, {"bridged": False, "source_year": expected_prev}

    y_prev = max(seasons)
    gap = test_year - y_prev
    shrinkage = STALENESS_SHRINKAGE ** (gap - 1)

    bridged_train = _bridge_frame_year(train_data, y_prev, expected_prev, shrinkage)
    bridged_kwargs = {
        k: _bridge_frame_year(v, y_prev, expected_prev, shrinkage)
        if isinstance(v, pd.DataFrame)
        else v
        for k, v in kwargs.items()
    }

    # Backfill NaN feature columns directly on test_data if pre-joined upstream.
    bridged_test = test_data.copy()
    if "tmID" in bridged_test.columns and "tmID" in bridged_train.columns:
        prev_rows = (
            _filter_nhl(bridged_train[bridged_train["year"] == expected_prev])
            .drop_duplicates("tmID")
            .set_index("tmID")
        )
        shared_cols = [
            c
            for c in bridged_test.columns
            if c in prev_rows.columns and c not in _ID_COLS
        ]
        for col in shared_cols:
            missing_mask = bridged_test[col].isna()
            if not missing_mask.any():
                continue

            mapped = bridged_test.loc[missing_mask, "tmID"].map(prev_rows[col])

            # All-NaN string columns (e.g. playoff) default to float64 in pandas;
            # cast to object before assigning non-numeric values.
            if not pd.api.types.is_numeric_dtype(prev_rows[col]):
                bridged_test[col] = bridged_test[col].astype("object")
                bridged_test.loc[missing_mask, col] = mapped.astype("object")
            else:
                bridged_test[col] = pd.to_numeric(bridged_test[col], errors="coerce")
                bridged_test.loc[missing_mask, col] = pd.to_numeric(mapped, errors="coerce")
                if bridged_test[col].isna().any():
                    bridged_test[col] = bridged_test[col].fillna(prev_rows[col].median())

    meta = {"bridged": True, "source_year": y_prev, "target_year": expected_prev, "gap": gap}
    return bridged_train, bridged_test, bridged_kwargs, meta


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Within-split percentile rank; missing values become neutral 0.5."""
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rank(method="average", pct=True).fillna(0.5).to_numpy(dtype=float)


def _component_probabilities(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    kwargs: dict,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run available component models and return both raw probabilities and ranks."""
    base = _filter_nhl(test_data)
    base = base[["year", "tmID"]].drop_duplicates(["year", "tmID"]).reset_index(drop=True)

    raw_probs: dict[str, np.ndarray] = {}
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
                continue
            probs = aligned.to_numpy(dtype=float)
            raw_probs[name] = probs
            ranks[name] = _percentile_rank(probs)
        except Exception:  # noqa: BLE001
            continue

    return base, raw_probs, ranks


def _prior_turnover_rate(train_data: pd.DataFrame, kwargs: dict, test_year: int) -> float | None:
    """Estimate the most recent season's coaching change rate for mean-reversion."""
    for candidate in (train_data, kwargs.get("coaches"), kwargs.get("teams")):
        if not isinstance(candidate, pd.DataFrame) or candidate.empty:
            continue
        if not {"year", "tmID", "coachID"}.issubset(candidate.columns):
            continue
        sub = _filter_nhl(candidate)
        years = sorted(int(y) for y in sub["year"].dropna().unique() if y < test_year)
        if len(years) < 2:
            continue
        y_last, y_prev = years[-1], years[-2]
        # Primary (last-stint) coach per team in each season
        c_prev = sub[sub["year"] == y_prev].sort_values("stint" if "stint" in sub.columns else "year").groupby("tmID")["coachID"].last()
        c_last = sub[sub["year"] == y_last].sort_values("stint" if "stint" in sub.columns else "year").groupby("tmID")["coachID"].last()
        shared = c_prev.index.intersection(c_last.index)
        if len(shared) >= 10:
            return float((c_prev.loc[shared] != c_last.loc[shared]).mean())
    return None


def _estimate_flag_fraction(
    raw_probs: dict[str, np.ndarray],
    prev_rate: float | None,
    bridged: bool,
    base_frac: float = BASE_FLAG_FRACTION,
    min_frac: float = MIN_FLAG_FRACTION,
    max_frac: float = MAX_FLAG_FRACTION,
) -> float:
    """Estimate the season's positive-flagging rate without overfitting."""
    prob_components = [
        (RISK_WEIGHTS[name], float(np.mean(raw_probs[name])))
        for name in ("model_2", "model_12")
        if name in raw_probs
    ]
    if prob_components:
        w_sum = sum(w for w, _ in prob_components)
        model_rate = sum(w * rate for w, rate in prob_components) / w_sum
        estimate = 0.5 * base_frac + 0.5 * model_rate
    else:
        estimate = base_frac

    if prev_rate is not None:
        estimate -= 0.25 * (prev_rate - base_frac)

    if bridged:
        estimate = min(estimate, base_frac)

    return float(np.clip(estimate, min_frac, max_frac))


def _flag_top(
    scores: np.ndarray,
    flag_fraction: float,
    max_frac: float = MAX_FLAG_FRACTION,
) -> np.ndarray:
    n_teams = len(scores)
    if n_teams == 0:
        return np.zeros(0, dtype=int)
    k = max(MIN_FLAGS, int(round(flag_fraction * n_teams)))
    k = min(k, n_teams, int(np.ceil(max_frac * n_teams)))
    predicted = np.zeros(n_teams, dtype=int)
    if k > 0:
        predicted[np.argsort(-scores, kind="stable")[:k]] = 1
    return predicted

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

    return (
        base
        + AGREEMENT_PRODUCT_WEIGHT * r2 * r12
        + AGREEMENT_MIN_WEIGHT * np.minimum(r2, r12)
    )

def train_and_predict(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    *args,
    **kwargs,
) -> pd.DataFrame:
    if test_data.empty or "tmID" not in test_data.columns:
        raise ValueError("test_data must contain tmID")

    test_year = int(test_data["year"].dropna().iloc[0]) if "year" in test_data.columns else -1
    if test_year > 0:
        train_data, test_data, kwargs, bridge_meta = _bridge_missing_prior_season(
            train_data, test_data, kwargs, test_year
        )
    else:
        bridge_meta = {"bridged": False}

    base, raw_probs, ranks = _component_probabilities(train_data, test_data, kwargs)
    if base.empty:
        raise ValueError("No NHL teams to predict")

    n_teams = len(base)
    risk = _risk_score(ranks, n_teams)
    decision = _decision_score(ranks, n_teams)
    prev_rate = _prior_turnover_rate(train_data, kwargs, test_year) if test_year > 0 else None
    bridged = bool(bridge_meta.get("bridged", False))

    flag_fraction = _estimate_flag_fraction(
        raw_probs=raw_probs,
        prev_rate=prev_rate,
        bridged=bridged,
    )

    probability = 1.0 / (1.0 + np.exp(-PROBABILITY_SLOPE * (risk - 0.5)))

    result = base.copy()
    result["change_probability"] = probability
    result["predicted_change"] = _flag_top(decision, flag_fraction)
    result.attrs["cutoff_info"] = {
        "strategy": "stacked_rank_blend_dynamic",
        "components": [name for name, _ in COMPONENTS if name in ranks],
        "flag_fraction": round(flag_fraction, 4),
        "bridge": bridge_meta,
    }

    if test_year > 0:
        _SWEEP_CACHE[test_year] = {
            "teams": base[["year", "tmID"]].copy(),
            "decision": decision.copy(),
            "raw_probs": {k: v.copy() for k, v in raw_probs.items()},
            "prev_rate": prev_rate,
            "bridged": bridged,
        }

    return result[["year", "tmID", "change_probability", "predicted_change"]]


def predict_for_base_fraction(test_year: int, base_frac: float) -> pd.DataFrame:
    """Recompute binary flags for a cached test year using a candidate base_frac."""
    if test_year not in _SWEEP_CACHE:
        raise KeyError(f"No cached scores for test_year={test_year}")

    entry = _SWEEP_CACHE[test_year]
    sweep_min = max(0.15, base_frac - 0.10)
    sweep_max = min(0.85, base_frac + 0.12)

    frac = _estimate_flag_fraction(
        raw_probs=entry["raw_probs"],
        prev_rate=entry["prev_rate"],
        bridged=entry["bridged"],
        base_frac=base_frac,
        min_frac=sweep_min,
        max_frac=sweep_max,
    )
    out = entry["teams"].copy()
    out["predicted_change"] = _flag_top(entry["decision"], frac, max_frac=sweep_max)
    return out