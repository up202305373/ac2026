from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_FORWARDS = 12
MIN_SKATER_GP = 20
MIN_GOALIE_GP = 10
DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# Used when alpha is not fixed: selected on a held-out training season.
VALIDATION_ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)

DEFAULT_KNOBS = {
    "n_forwards": N_FORWARDS,
    "min_skater_gp": MIN_SKATER_GP,
    "min_goalie_gp": MIN_GOALIE_GP,
}

FEATURE_GROUPS = {
    "base":      ["team_gd", "fwd_ppg", "goalie_svpct", "new_share"],
    "shots":     ["shot_diff", "shoot_pct", "team_sv_pct"],
    "special":   ["pp_pct", "pk_pct"],
    "prior_pts": ["prior_pts"],
}

def features_for(groups):
    """Flat column list for the given feature groups."""
    return [col for group in groups for col in FEATURE_GROUPS[group]]

FEATURES = features_for(("base", "shots", "special"))

def pos_group(pos) -> str:
    """Defense -> D, Goalies -> G, Centers, Left Wings, Right Wings, Forwards -> F."""
    """Uses the first listed position if multiple are given (e.g. "C/G")."""
    """Missing positions default to F."""
    if pd.isna(pos):
        return "F"
    first = str(pos).split("/")[0]
    if first in ("G", "D"):
        return first
    return "F"

def prior_skaters(scoring: pd.DataFrame, year: int, knobs=None) -> pd.DataFrame:
    """Calculates each qualifying forward's points per game for one season."""
    knobs = knobs or DEFAULT_KNOBS
    s = scoring[scoring["year"] == year].copy()
    s["group"] = s["pos"].map(pos_group)
    s = s[s["group"] == "F"]
    tot = s.groupby("playerID").agg(GP=("GP", "sum"), Pts=("Pts", "sum"), group=("group", "first"))
    tot = tot[tot["GP"] >= knobs["min_skater_gp"]].copy()
    tot["ppg"] = tot["Pts"] / tot["GP"]
    return tot[["ppg", "group"]]

def prior_goalies(goalies: pd.DataFrame, year: int, knobs=None) -> pd.DataFrame:
    """Calculates each qualifying goalie's save percentage for one season."""
    knobs = knobs or DEFAULT_KNOBS
    g = goalies[goalies["year"] == year]
    tot = g.groupby("playerID").agg(prior_GP=("GP", "sum"), GA=("GA", "sum"), SA=("SA", "sum"))
    tot = tot[(tot["prior_GP"] >= knobs["min_goalie_gp"]) & (tot["SA"] > 0)].copy()
    tot["svpct"] = 1 - tot["GA"] / tot["SA"]
    return tot[["svpct", "prior_GP"]]

def prior_shot_features(scoring, goalies, teams, year: int) -> pd.DataFrame:
    """Prior-season shot differential, shooting % and save % by franchise."""
    t = teams[(teams["year"] == year) & (teams["lgID"] == "NHL")]
    sog = scoring[scoring["year"] == year].groupby("tmID")["SOG"].sum()
    sa = goalies[goalies["year"] == year].groupby("tmID")["SA"].sum()
    shots = pd.concat([sog.rename("SOG"), sa.rename("SA")], axis=1).fillna(0)
    shots["franchID"] = shots.index.map(dict(zip(t["tmID"], t["franchID"])))
    agg = shots.groupby("franchID")[["SOG", "SA"]].sum()
    games = t.groupby("franchID")["G"].mean()
    gf = t.groupby("franchID")["GF"].mean()
    ga = t.groupby("franchID")["GA"].mean()
    return pd.DataFrame({
        "shot_diff":   (agg["SOG"] - agg["SA"]) / games,
        "shoot_pct":   gf / agg["SOG"],
        "team_sv_pct": 1 - ga / agg["SA"],
    })

def prior_special_features(teams, year: int) -> pd.DataFrame:
    """Prior-season power-play and penalty-kill rates by franchise."""
    t = teams[(teams["year"] == year) & (teams["lgID"] == "NHL")]
    rates = pd.DataFrame({
        "pp_pct": t["PPG"].astype(float) / t["PPC"].astype(float),
        "pk_pct": 1 - t["SHA"].astype(float) / t["PKC"].astype(float),
        "franchID": t["franchID"].values,
    })
    return rates.groupby("franchID")[["pp_pct", "pk_pct"]].mean()

def team_features(skaters, goalies, prior_sk, prior_g, league, prior_player_ids=None,
                  knobs=None) -> dict:
    """Builds 3 team features from current-roster players and prior-season data."""
    knobs = knobs or DEFAULT_KNOBS
    sk = skaters.copy()
    sk["group"] = sk["pos"].map(pos_group)
    sk = sk.merge(prior_sk[["ppg"]], left_on="playerID", right_index=True, how="left")
    sk["ppg"] = sk["ppg"].fillna(league["ppg"])

    forward_ppg = sk.loc[sk["group"] == "F", "ppg"]
    n = knobs["n_forwards"]
    fwd = (forward_ppg.mean() if n == "all" else forward_ppg.nlargest(int(n)).mean())
    gk = goalies.merge(prior_g, left_on="playerID", right_index=True, how="left")
    known = gk.dropna(subset=["prior_GP"])
    svpct = (known.sort_values("prior_GP", ascending=False)["svpct"].iloc[0]
             if not known.empty else league["svpct"])

    if prior_player_ids is None:
        prior_player_ids = set(prior_sk.index) | set(prior_g.index)
    roster = pd.concat([sk[["playerID"]], gk[["playerID"]]], ignore_index=True)

    """1) Mean PPG of the top N_FORWARDS forwards, or league average if no forwards qualify."""
    """2) Save percentage of the goalie with the most prior-season GP, or league average if no goalies qualify."""
    """3) Proportion of roster players not seen in the prior season, or NaN if the roster is empty"""

    return {
        "fwd_ppg": fwd if pd.notna(fwd) else league["ppg"],
        "goalie_svpct": svpct,
        "new_share": (~roster["playerID"].isin(prior_player_ids)).mean() if len(roster) else np.nan,
    }

def previous_season(years, year):
    """Returns the most recent available season before the target year."""
    earlier = [y for y in set(years) if y < year]
    return max(earlier) if earlier else None

def build_rows(teams, roster_scoring, roster_goalies, prior_scoring, prior_goalies_all, year,
               prior_teams, knobs=None) -> pd.DataFrame:
    """Builds one feature row per team using its roster and prior-season context."""
    knobs = knobs or DEFAULT_KNOBS
    prev = previous_season(prior_scoring["year"], year)
    prior_sk = prior_skaters(prior_scoring, prev, knobs)
    prior_g = prior_goalies(prior_goalies_all, prev, knobs)
    prior_player_ids = set(
        prior_scoring.loc[prior_scoring["year"] == prev, "playerID"]
    ) | set(prior_goalies_all.loc[prior_goalies_all["year"] == prev, "playerID"])
    pt = prior_teams[prior_teams["year"] == prev]
    team_gd = ((pt["GF"] - pt["GA"]) / pt["G"]).groupby(pt["franchID"].values).mean()
    prior_pts = pt.groupby("franchID")["Pts"].mean()
    league = {"ppg": prior_sk["ppg"].mean(), "svpct": prior_g["svpct"].mean()}
    shot_feats = prior_shot_features(prior_scoring, prior_goalies_all, prior_teams, prev)
    special_feats = prior_special_features(prior_teams, prev)

    roster_sk = roster_scoring[(roster_scoring["year"] == year) & (roster_scoring["stint"] == 1)]
    roster_g = roster_goalies[(roster_goalies["year"] == year) & (roster_goalies["stint"] == 1)]
    rows = []
    team_list = teams.loc[teams["year"] == year, ["tmID", "franchID"]].drop_duplicates()
    for tmID, franchID in team_list.itertuples(index=False):
        feats = team_features(
            roster_sk[roster_sk["tmID"] == tmID],
            roster_g[roster_g["tmID"] == tmID],
            prior_sk, prior_g, league, prior_player_ids, knobs,
        )
        feats["team_gd"] = team_gd.get(franchID, team_gd.mean())
        feats["prior_pts"] = prior_pts.get(franchID, prior_pts.mean())
        for column, lookup in (("shot_diff", shot_feats), ("shoot_pct", shot_feats),
                               ("team_sv_pct", shot_feats), ("pp_pct", special_feats),
                               ("pk_pct", special_feats)):
            feats[column] = lookup[column].get(franchID, lookup[column].mean())
        rows.append({"year": year, "franchID": franchID, **feats})
    return pd.DataFrame(rows)

def build_features(teams, roster_scoring, roster_goalies, prior_scoring, prior_goalies_all, target_year, prior_teams, previous_years=1, knobs=None) -> pd.DataFrame:
    """Build target-year features from the last N available prior seasons.

    Each prior season produces one full feature table via build_rows(); the
    tables are then averaged per franchise. With previous_years=1 this returns
    exactly what build_rows() would have returned on its own.
    """
    previous_years = int(previous_years)
    if previous_years < 1:
        raise ValueError("previous_years must be >= 1")

    seasons = sorted({int(y) for y in prior_scoring["year"] if y < target_year})[-previous_years:]
    if not seasons:
        raise ValueError(f"No prior seasons available before {target_year}")

    frames = []
    for season in seasons:
        frames.append(build_rows(
            teams, roster_scoring, roster_goalies,
            prior_scoring[prior_scoring["year"] <= season],
            prior_goalies_all[prior_goalies_all["year"] <= season],
            target_year,
            prior_teams[prior_teams["year"] <= season],
            knobs,
        ))

    if len(frames) == 1:
        return frames[0]

    all_features = pd.concat(frames, ignore_index=True)
    feature_cols = [c for c in all_features.columns if c not in ("year", "franchID")]
    averaged = all_features.groupby("franchID")[feature_cols].mean()
    out = all_features[["year", "franchID"]].drop_duplicates().merge(
        averaged, left_on="franchID", right_index=True, how="left")
    out["year"] = target_year
    return out[["year", "franchID"] + feature_cols].reset_index(drop=True)

def load_players(split: str) -> dict:
    return {t: pd.read_csv(DATA_DIR / split / f"{t}.csv") for t in ("Scoring", "Goalies")}

def make_model(alpha):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha))


def training_rows(train_data, train_players, cols, previous_years, knobs=None) -> pd.DataFrame:
    """One row per team-season, built only from that season's prior context."""
    frames = []
    for year in sorted(train_data["year"].unique()):
        earlier = train_data[train_data["year"] < year]
        if earlier.empty:
            continue
        feats = build_features(
            train_data, train_players["Scoring"], train_players["Goalies"],
            train_players["Scoring"][train_players["Scoring"]["year"] < year],
            train_players["Goalies"][train_players["Goalies"]["year"] < year],
            year, earlier, previous_years, knobs,
        )
        target = train_data.loc[train_data["year"] == year, ["franchID", "Pts"]]
        frames.append(feats.merge(target, on="franchID"))
    return pd.concat(frames).dropna(subset=cols + ["Pts"])


def select_alpha(train_data, train_players, cols, previous_years, alphas=VALIDATION_ALPHAS, knobs=None):
    """Pick alpha on the latest training season, before fitting on all seasons.

    The held-out season is the last one available before the target year, so
    this stays a rolling-origin validation and never touches test-year labels.
    """
    seasons = sorted(train_data["year"].unique())
    if len(seasons) < 2:
        return alphas[len(alphas) // 2]

    val_year = seasons[-1]
    val_train = train_data[train_data["year"] < val_year]
    rows = training_rows(val_train, train_players, cols, previous_years, knobs)
    val_feats = build_features(
        train_data, train_players["Scoring"], train_players["Goalies"],
        train_players["Scoring"][train_players["Scoring"]["year"] < val_year],
        train_players["Goalies"][train_players["Goalies"]["year"] < val_year],
        val_year, val_train, previous_years, knobs,
    ).dropna(subset=cols)
    actual = train_data.loc[train_data["year"] == val_year, ["franchID", "Pts"]]
    val = val_feats.merge(actual, on="franchID").dropna(subset=cols + ["Pts"])

    if rows.empty or val.empty:
        return alphas[len(alphas) // 2]

    scores = []
    for alpha in alphas:
        model = make_model(alpha)
        model.fit(rows[cols], rows["Pts"].astype(float))
        error = model.predict(val[cols]) - val["Pts"].astype(float)
        scores.append((float(error.abs().mean()), alpha))
    return min(scores)[1]


def train_and_predict(train_data, test_data, train_players=None, test_players=None,
                      alpha=None, features=None, previous_years=1,
                      validation_alphas=VALIDATION_ALPHAS,
                      n_forwards=N_FORWARDS, min_skater_gp=MIN_SKATER_GP,
                      min_goalie_gp=MIN_GOALIE_GP) -> pd.DataFrame:
    """Trains a ridge model on prior seasons and predicts the test-year rankings.

    alpha=None selects the value on a held-out training season; the chosen
    value is exposed in the returned frame's attrs for logging.
    """
    cols = features_for(features) if features is not None else FEATURES
    train_players = train_players or load_players("train")
    test_players = test_players or load_players("test")
    knobs = {
        "n_forwards": n_forwards,
        "min_skater_gp": min_skater_gp,
        "min_goalie_gp": min_goalie_gp,
    }

    selected_alpha = None
    if alpha is None:
        alpha = select_alpha(train_data, train_players, cols, previous_years,
                             validation_alphas, knobs)
        selected_alpha = alpha

    train_rows = training_rows(train_data, train_players, cols, previous_years, knobs)
    model = make_model(alpha)
    model.fit(train_rows[cols], train_rows["Pts"].astype(float))

    test_year = int(test_data["year"].iloc[0])
    test_feats = build_features(
        test_data, test_players["Scoring"], test_players["Goalies"],
        train_players["Scoring"], train_players["Goalies"], test_year, train_data, previous_years, knobs,
    ).dropna(subset=cols)
    predictions = pd.DataFrame({
        "year": test_feats["year"].values,
        "franchID": test_feats["franchID"].values,
        "predicted_pts": model.predict(test_feats[cols]),
    })
    if selected_alpha is not None:
        predictions.attrs["selected_alpha"] = selected_alpha
    return predictions