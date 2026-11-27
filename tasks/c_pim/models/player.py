from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

POSITIONS = ("C", "D", "G", "L", "R")

FEATURE_GROUPS = {
    "base": (
        "prior_pim",
        "prior_gp",
        "prior_sog",
        "prior_pim_per_gp",
        "has_prior",
    ),
    "pos": tuple(f"pos_{p}" for p in POSITIONS),
}
DEFAULT_FEATURES = ("base",)
DEFAULT_ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)

def features_for(groups):
    """Flat column list for the given feature groups."""
    return [col for group in groups for col in FEATURE_GROUPS[group]]

def _require_columns(data: pd.DataFrame, name: str) -> None:
    required = {"year", "playerID", "stint", "pos", "GP", "PIM", "SOG"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")

def _prior_year(years, target_year: int) -> int | None:
    prior = sorted({int(y) for y in years if int(y) < int(target_year)})
    return prior[-1] if prior else None

def _feature_frame(target_year: int, player_ids, scoring: pd.DataFrame,
                   features) -> pd.DataFrame:
    """Build prior-season features for target-season player IDs."""
    ids = pd.DataFrame({"playerID": pd.Index(player_ids).unique()})
    prior_year = _prior_year(scoring["year"].unique(), target_year)
    columns = ["playerID", "pos", "GP", "PIM", "SOG"]
    prior_columns = ["playerID", "pos", "prior_gp", "prior_pim", "prior_sog"]

    if prior_year is None:
        prior = pd.DataFrame(columns=prior_columns)
    else:
        prior = scoring[scoring["year"] == prior_year][columns].copy()
        for column in ("GP", "PIM", "SOG"):
            prior[column] = pd.to_numeric(prior[column], errors="coerce")
        # Last season's stints are observed and aggregated across all of them.
        prior = prior.groupby("playerID", as_index=False).agg(
            pos=("pos", "first"),
            prior_gp=("GP", "sum"),
            prior_pim=("PIM", "sum"),
            prior_sog=("SOG", "sum"),
        )

    out = ids.merge(prior, on="playerID", how="left", validate="one_to_one",
                    indicator=True)
    out["has_prior"] = out["_merge"].eq("both").astype(int)
    out = out.drop(columns="_merge")
    for column in ("prior_gp", "prior_pim", "prior_sog"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    # +1 avoids division by zero
    out["prior_pim_per_gp"] = out["prior_pim"] / (out["prior_gp"] + 1.0)

    pos = out["pos"].fillna("UNK")
    for p in POSITIONS:
        out[f"pos_{p}"] = pos.eq(p).astype(int)

    return out[["playerID", *features]]

def _training_examples(train_data: pd.DataFrame, features) -> pd.DataFrame:
    # Labels stay no-trade (stint 1), but prior-season features use all stints.
    target_scoring = train_data[train_data["stint"] == 1].copy()
    rows = []

    for target_year in sorted(target_scoring["year"].unique()):
        target_year = int(target_year)
        if _prior_year(train_data["year"].unique(), target_year) is None:
            continue

        target = (
            target_scoring[target_scoring["year"] == target_year][["playerID", "PIM"]]
            .drop_duplicates("playerID")
            .dropna(subset=["PIM"])
        )
        if target.empty:
            continue

        frame = _feature_frame(target_year, target["playerID"], train_data, features)
        labeled = frame.merge(target, on="playerID", how="inner",
                              validate="one_to_one")
        labeled["year"] = target_year
        rows.append(labeled)

    if not rows:
        raise ValueError("No usable training examples with prior-season features")
    return pd.concat(rows, ignore_index=True)

def _fit_predict(alpha: float, train: pd.DataFrame, predict: pd.DataFrame,
                 features) -> np.ndarray:
    scaler = StandardScaler().fit(train[features])
    model = Ridge(alpha=alpha).fit(
        scaler.transform(train[features]), train["PIM"].to_numpy())
    return np.clip(model.predict(scaler.transform(predict[features])), 0.0, None)

def train_and_predict(train_data: pd.DataFrame, test_data: pd.DataFrame, features=DEFAULT_FEATURES, alphas=DEFAULT_ALPHAS, *_, **__) -> pd.DataFrame:
    """Ridge regression using only prior-season player information."""
    _require_columns(train_data, "train_data")
    _require_columns(test_data, "test_data")

    active_features = features_for(features)
    alphas = list(alphas) if alphas else list(DEFAULT_ALPHAS)
    examples = _training_examples(train_data, active_features)

    holdout_year = int(examples["year"].max())
    fit_rows = examples[examples["year"] < holdout_year]
    val_rows = examples[examples["year"] == holdout_year]
    if fit_rows.empty or val_rows.empty:
        best_alpha = alphas[0]
    else:
        errors = {}
        for alpha in alphas:
            pred = _fit_predict(alpha, fit_rows, val_rows, active_features)
            errors[alpha] = float(np.mean(np.abs(pred - val_rows["PIM"].to_numpy())))
        best_alpha = min(errors, key=errors.get)

    test_year = int(test_data["year"].iloc[0])
    test_features = _feature_frame(
        test_year, test_data["playerID"], train_data, active_features)
    predicted = _fit_predict(best_alpha, examples, test_features, active_features)

    predictions = test_data[["year", "playerID"]].copy()
    predictions["predicted_pim"] = predicted
    predictions.attrs["selected_alpha"] = float(best_alpha)
    return predictions
