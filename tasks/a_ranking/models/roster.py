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

FEATURE_GROUPS = {
    "base":    ["team_gd", "fwd_ppg", "goalie_svpct", "new_share"],
    "shots":   ["shot_diff", "shoot_pct", "team_sv_pct"],
    "special": ["pp_pct", "pk_pct"],
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

def prior_skaters(scoring: pd.DataFrame, year: int) -> pd.DataFrame:
    """Calculates each qualifying forward's points per game for one season."""
    s = scoring[scoring["year"] == year].copy()
    s["group"] = s["pos"].map(pos_group)
    s = s[s["group"] == "F"]
    tot = s.groupby("playerID").agg(GP=("GP", "sum"), Pts=("Pts", "sum"), group=("group", "first"))
    tot = tot[tot["GP"] >= MIN_SKATER_GP].copy()
    tot["ppg"] = tot["Pts"] / tot["GP"]
    return tot[["ppg", "group"]]

def prior_goalies(goalies: pd.DataFrame, year: int) -> pd.DataFrame:
    """Calculates each qualifying goalie's save percentage for one season."""
    g = goalies[goalies["year"] == year]
    tot = g.groupby("playerID").agg(prior_GP=("GP", "sum"), GA=("GA", "sum"), SA=("SA", "sum"))
    tot = tot[(tot["prior_GP"] >= MIN_GOALIE_GP) & (tot["SA"] > 0)].copy()
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

def team_features(skaters, goalies, prior_sk, prior_g, league, prior_player_ids=None) -> dict:
    """Builds 3 team features from current-roster players and prior-season data."""
    sk = skaters.copy()
    sk["group"] = sk["pos"].map(pos_group)
    sk = sk.merge(prior_sk[["ppg"]], left_on="playerID", right_index=True, how="left")
    sk["ppg"] = sk["ppg"].fillna(league["ppg"])

    fwd = sk.loc[sk["group"] == "F", "ppg"].nlargest(N_FORWARDS).mean()
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

def build_rows(teams, roster_scoring, roster_goalies, prior_scoring, prior_goalies_all, year, prior_teams) -> pd.DataFrame:
    """Builds one feature row per team using its roster and prior-season context."""
    prev = previous_season(prior_scoring["year"], year)
    prior_sk = prior_skaters(prior_scoring, prev)
    prior_g = prior_goalies(prior_goalies_all, prev)
    prior_player_ids = set(
        prior_scoring.loc[prior_scoring["year"] == prev, "playerID"]
    ) | set(prior_goalies_all.loc[prior_goalies_all["year"] == prev, "playerID"])
    pt = prior_teams[prior_teams["year"] == prev]
    team_gd = ((pt["GF"] - pt["GA"]) / pt["G"]).groupby(pt["franchID"].values).mean()
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
            prior_sk, prior_g, league, prior_player_ids,
        )
        feats["team_gd"] = team_gd.get(franchID, team_gd.mean())
        for column, lookup in (("shot_diff", shot_feats), ("shoot_pct", shot_feats),
                               ("team_sv_pct", shot_feats), ("pp_pct", special_feats),
                               ("pk_pct", special_feats)):
            feats[column] = lookup[column].get(franchID, lookup[column].mean())
        rows.append({"year": year, "franchID": franchID, **feats})
    return pd.DataFrame(rows)

def load_players(split: str) -> dict:
    return {t: pd.read_csv(DATA_DIR / split / f"{t}.csv") for t in ("Scoring", "Goalies")}

def train_and_predict(train_data, test_data, train_players=None, test_players=None, alpha=30.0, features=None) -> pd.DataFrame:
    """Trains a ridge model on prior seasons and predicts the test-year rankings."""
    cols = features_for(features) if features is not None else FEATURES
    train_players = train_players or load_players("train")
    test_players = test_players or load_players("test")
    train_years = set(train_data["year"].unique())
    frames = []
    for year in sorted(train_years):
        earlier = train_data[train_data["year"] < year]
        if earlier.empty:
            continue
        feats = build_rows(
            train_data, train_players["Scoring"], train_players["Goalies"],
            train_players["Scoring"][train_players["Scoring"]["year"] < year],
            train_players["Goalies"][train_players["Goalies"]["year"] < year], year, earlier,
        )
        target = train_data.loc[train_data["year"] == year, ["franchID", "Pts"]]
        frames.append(feats.merge(target, on="franchID"))
    train_rows = pd.concat(frames).dropna(subset=cols + ["Pts"])

    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha))
    model.fit(train_rows[cols], train_rows["Pts"].astype(float))

    test_year = int(test_data["year"].iloc[0])
    test_feats = build_rows(
        test_data, test_players["Scoring"], test_players["Goalies"],
        train_players["Scoring"], train_players["Goalies"], test_year, train_data,
    ).dropna(subset=cols)
    return pd.DataFrame({
        "year": test_feats["year"].values,
        "franchID": test_feats["franchID"].values,
        "predicted_pts": model.predict(test_feats[cols]),
    })