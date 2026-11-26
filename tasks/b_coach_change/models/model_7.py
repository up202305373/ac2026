"""Coach-change prediction model 6 (pruned + player/goalie deltas).

Key improvements
----------------
* Prunes high-variance / heavily-shrunk noise features (Coach_Age, Has_Award,
  Coach_Change_Rate, Team_Change_Rate) that StandardScaler over-weights.
* Replaces collinear raw GD_pg_Z_P1 with Pts_Luck_P1 (Pts_Pct_Z_P1 - GD_pg_Z_P1)
  and splits linear Coach_Tenure into Is_New_Coach + capped tenure.
* Adds leakage-free player & goalie performance deltas:
  - Goalie_Perf_Z_P1 & Goalie_Perf_Z_Delta (SV% / -GAA season z-score & YoY delta)
  - Skater_Core_Delta_P1 (YoY Pts/GP z-score change for returning skaters)
  - Roster_Talent_Delta (offseason incoming vs. outgoing Y-1 skater talent when
    test-season roster rows are provided)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from coach_thresholding import select_coach_changes

CHANGE_FRACTION = 0.34
MIN_FLAGS = 3
TRAINING_WINDOW_SEASONS = 25
MAX_PLAYOFF_DROUGHT = 6
MAX_TENURE_CAP = 6

FEATURES = [
    "Pts_Pct_Z_P1",
    "Pts_Pct_Z_Delta",
    "Pts_Luck_P1",
    "Made_Playoffs_P1",
    "Playoff_Drought_P1",
    "Midseason_Change_P1",
    "Is_New_Coach",
    "Coach_Tenure_Capped",
    "Goalie_Perf_Z_P1",
    "Goalie_Perf_Z_Delta",
    "Skater_Core_Delta_P1",
    "Roster_Talent_Delta",
]


def _make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.35, max_iter=2000),
    )


def _league_col(frame: pd.DataFrame) -> str:
    for col in ("lgID", "lgID_y", "lgID_x"):
        if col in frame.columns:
            return col
    return ""


def _filter_nhl(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    lcol = _league_col(df)
    if lcol:
        df = df[df[lcol] == "NHL"].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    return df


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.replace(0, np.nan)
    return numerator / den


def _nhl_season_map(years: pd.Series) -> dict[int, int]:
    unique_years = sorted(pd.to_numeric(years, errors="coerce").dropna().astype(int).unique())
    return {year: idx for idx, year in enumerate(unique_years)}


def _season_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("season_idx")[column]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return (frame[column] - mean) / std


def _first_last_coaches(coaches: pd.DataFrame, season_col: str = "year"):
    ordered = coaches.sort_values(["tmID", season_col, "stint"], na_position="last")
    first = ordered.drop_duplicates(["tmID", season_col], keep="first")
    last = ordered.drop_duplicates(["tmID", season_col], keep="last")
    return first, last


def _extract_tables(args: tuple, kwargs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Locate Scoring and Goalies DataFrames from kwargs or positional args."""
    scoring_frames: list[pd.DataFrame] = []
    goalie_frames: list[pd.DataFrame] = []

    for key in ("train_scoring", "test_scoring", "scoring", "train_players", "test_players"):
        val = kwargs.get(key)
        if isinstance(val, pd.DataFrame) and not val.empty:
            scoring_frames.append(val)

    for key in ("train_goalies", "test_goalies", "goalies"):
        val = kwargs.get(key)
        if isinstance(val, pd.DataFrame) and not val.empty:
            goalie_frames.append(val)

    for arg in args:
        if not isinstance(arg, pd.DataFrame) or arg.empty:
            continue
        cols = set(arg.columns)
        if {"playerID", "tmID", "year"}.issubset(cols):
            if {"A", "Pts"}.issubset(cols) or "SOG" in cols or "PPG" in cols:
                scoring_frames.append(arg)
            elif "GA" in cols and ("Min" in cols or "SA" in cols or "SHO" in cols):
                goalie_frames.append(arg)

    scoring = pd.concat(scoring_frames, ignore_index=True) if scoring_frames else pd.DataFrame()
    goalies = pd.concat(goalie_frames, ignore_index=True) if goalie_frames else pd.DataFrame()
    return scoring, goalies


def build_history_labels(train_coaches: pd.DataFrame, train_teams: pd.DataFrame) -> pd.DataFrame:
    """One labelled row per NHL (tmID, year), using the next NHL season's finishing coach."""
    nhl = _filter_nhl(train_teams)[["year", "tmID"]].drop_duplicates().copy()
    year_to_idx = _nhl_season_map(nhl["year"])
    nhl["season_idx"] = nhl["year"].map(year_to_idx)

    coaches = _filter_nhl(train_coaches)
    coaches["stint"] = _numeric(coaches, "stint")

    _, last = _first_last_coaches(coaches, season_col="year")
    labelled = nhl.merge(last[["tmID", "year", "coachID"]], on=["tmID", "year"], how="inner")

    following = labelled[["tmID", "season_idx", "coachID"]].copy()
    following["season_idx"] = following["season_idx"] - 1
    following = following.rename(columns={"coachID": "Next_Coach"})

    labelled = labelled.merge(following, on=["tmID", "season_idx"], how="left")
    labelled["Coach_Changed"] = np.where(
        labelled["Next_Coach"].isna(),
        np.nan,
        (labelled["coachID"] != labelled["Next_Coach"]).astype(float),
    )
    return labelled[["tmID", "year", "coachID", "Next_Coach", "Coach_Changed"]]


def _compute_playoff_features(
    coaches: pd.DataFrame,
    team_seasons: pd.DataFrame,
    test_season_idx: int,
) -> pd.DataFrame:
    """Compute prior-season playoff appearance and consecutive missed-playoff drought."""
    completed_coaches = coaches[coaches["season_idx"] < test_season_idx]
    postg = (
        completed_coaches.groupby(["tmID", "season_idx"])["postg"]
        .sum(min_count=1)
        .rename("postg_sum")
        .reset_index()
    )

    completed = (
        team_seasons[team_seasons["season_idx"] < test_season_idx][["tmID", "season_idx", "G"]]
        .merge(postg, on=["tmID", "season_idx"], how="left")
        .sort_values(["tmID", "season_idx"])
        .reset_index(drop=True)
    )

    season_has_playoffs = completed.groupby("season_idx")["postg_sum"].transform(
        lambda s: (s.fillna(0) > 0).any()
    )
    valid = season_has_playoffs & (completed["G"] > 0)
    completed["Made_Playoffs"] = np.where(
        valid,
        (completed["postg_sum"].fillna(0) > 0).astype(float),
        np.nan,
    )

    droughts = np.full(len(completed), np.nan)
    for _, grp in completed.groupby("tmID", sort=False):
        streak = 0
        prev_s = None
        for idx, s, made in zip(grp.index, grp["season_idx"], grp["Made_Playoffs"]):
            if np.isnan(made):
                streak = 0
                prev_s = None
                continue
            if prev_s is None or s != prev_s + 1:
                streak = 0
            streak = 0 if made == 1.0 else min(streak + 1, MAX_PLAYOFF_DROUGHT)
            droughts[idx] = float(streak)
            prev_s = s

    completed["Playoff_Drought"] = droughts
    shifted = completed[["tmID", "season_idx", "Made_Playoffs", "Playoff_Drought"]].copy()
    shifted["season_idx"] = shifted["season_idx"] + 1
    return shifted.rename(
        columns={
            "Made_Playoffs": "Made_Playoffs_P1",
            "Playoff_Drought": "Playoff_Drought_P1",
        }
    )


def _compute_goalie_features(
    goalies: pd.DataFrame,
    year_to_idx: dict[int, int],
    test_season_idx: int,
) -> pd.DataFrame:
    """Compute prior-season team goalie z-score (SV% with -GAA fallback) and YoY delta."""
    cols = ["tmID", "season_idx", "Goalie_Perf_Z_P1", "Goalie_Perf_Z_Delta"]
    if goalies.empty or not {"tmID", "year", "GA"}.issubset(goalies.columns):
        return pd.DataFrame(columns=cols)

    df = _filter_nhl(goalies)
    df["season_idx"] = df["year"].map(year_to_idx)
    df = df.dropna(subset=["tmID", "season_idx"]).copy()
    df["season_idx"] = df["season_idx"].astype(int)
    df = df[df["season_idx"] < test_season_idx]
    if df.empty:
        return pd.DataFrame(columns=cols)

    for col in ("GP", "Min", "GA", "SA"):
        df[col] = _numeric(df, col)

    team_g = (
        df.groupby(["tmID", "season_idx"], as_index=False)[["GP", "Min", "GA", "SA"]]
        .sum(min_count=1)
    )
    team_g["sv_pct"] = _safe_divide(team_g["SA"] - team_g["GA"], team_g["SA"])
    team_g["neg_gaa"] = -60.0 * _safe_divide(team_g["GA"], team_g["Min"])
    team_g["neg_ga_pg"] = -_safe_divide(team_g["GA"], team_g["GP"])

    sv_z = _season_zscore(team_g, "sv_pct")
    gaa_z = _season_zscore(team_g, "neg_gaa")
    gapg_z = _season_zscore(team_g, "neg_ga_pg")
    team_g["Goalie_Perf_Z"] = sv_z.fillna(gaa_z).fillna(gapg_z)

    p1 = team_g[["tmID", "season_idx", "Goalie_Perf_Z"]].copy()
    p1["season_idx"] = p1["season_idx"] + 1
    p1 = p1.rename(columns={"Goalie_Perf_Z": "Goalie_Perf_Z_P1"})

    p2 = team_g[["tmID", "season_idx", "Goalie_Perf_Z"]].copy()
    p2["season_idx"] = p2["season_idx"] + 2
    p2 = p2.rename(columns={"Goalie_Perf_Z": "Goalie_Perf_Z_P2"})

    out = p1.merge(p2, on=["tmID", "season_idx"], how="left")
    out["Goalie_Perf_Z_Delta"] = out["Goalie_Perf_Z_P1"] - out["Goalie_Perf_Z_P2"]
    return out[cols]


def _compute_skater_features(
    scoring: pd.DataFrame,
    year_to_idx: dict[int, int],
    test_season_idx: int,
) -> pd.DataFrame:
    """Compute returning-skater YoY delta (Y-2 -> Y-1) and incoming roster talent delta."""
    cols = ["tmID", "season_idx", "Skater_Core_Delta_P1", "Roster_Talent_Delta"]
    if scoring.empty or not {"playerID", "tmID", "year"}.issubset(scoring.columns):
        return pd.DataFrame(columns=cols)

    df = _filter_nhl(scoring)
    df["season_idx"] = df["year"].map(year_to_idx)
    df = df.dropna(subset=["playerID", "tmID", "season_idx"]).copy()
    df["season_idx"] = df["season_idx"].astype(int)
    if df.empty:
        return pd.DataFrame(columns=cols)

    for col in ("GP", "G", "A", "Pts"):
        df[col] = _numeric(df, col)
    if "Pts" not in scoring.columns or df["Pts"].isna().all():
        df["Pts"] = df["G"].fillna(0) + df["A"].fillna(0)

    # Historical completed player-team stints (strictly before test_season_idx)
    past = df[df["season_idx"] < test_season_idx].copy()
    past = past[past["GP"].fillna(0) > 0]
    if past.empty:
        return pd.DataFrame(columns=cols)

    # Aggregate across multiple stints if a player had duplicate rows on same team-season
    stints = (
        past.groupby(["playerID", "tmID", "season_idx"], as_index=False)[["GP", "Pts"]]
        .sum(min_count=1)
    )
    stints["pts_pg"] = _safe_divide(stints["Pts"], stints["GP"])
    # Normalize player scoring rate within each NHL season (minimum 10 GP for stability)
    qualified = stints[stints["GP"] >= 10].copy()
    season_stats = (
        qualified.groupby("season_idx")["pts_pg"]
        .agg(s_mean="mean", s_std="std")
        .reset_index()
    )
    stints = stints.merge(season_stats, on="season_idx", how="left")
    stints["pts_pg_z"] = (stints["pts_pg"] - stints["s_mean"]) / stints["s_std"].replace(0, np.nan)

    # 1. Returning skater core delta: players on the same tmID in both (s-2) and (s-1)
    curr = stints[["playerID", "tmID", "season_idx", "GP", "pts_pg_z"]].copy()
    prev = stints[["playerID", "tmID", "season_idx", "GP", "pts_pg_z"]].copy()
    prev["season_idx"] = prev["season_idx"] + 1
    prev = prev.rename(columns={"GP": "GP_prev", "pts_pg_z": "pts_pg_z_prev"})

    returning = curr.merge(prev, on=["playerID", "tmID", "season_idx"], how="inner")
    returning = returning[(returning["GP"] >= 10) & (returning["GP_prev"] >= 10)].copy()
    returning["w"] = np.minimum(returning["GP"], returning["GP_prev"])
    returning["w_delta"] = returning["w"] * (returning["pts_pg_z"] - returning["pts_pg_z_prev"])

    core_agg = (
        returning.groupby(["tmID", "season_idx"], as_index=False)[["w_delta", "w"]]
        .sum()
    )
    core_agg["Skater_Core_Delta_P1"] = _safe_divide(core_agg["w_delta"], core_agg["w"])
    core_agg["season_idx"] = core_agg["season_idx"] + 1  # aligns (Y-2 -> Y-1) delta to target Y

    # 2. Offseason roster talent delta: incoming roster(Y)'s Y-1 talent vs. team's Y-1 talent
    player_y1 = (
        stints.groupby(["playerID", "season_idx"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "prev_gp": g["GP"].sum(),
                    "prev_pts_z": np.average(g["pts_pg_z"].fillna(0.0), weights=g["GP"].clip(lower=1)),
                }
            ),
            include_groups=False,
        )
    )
    player_y1["season_idx"] = player_y1["season_idx"] + 1  # Y-1 player rating mapped to Y

    # Team's actual Y-1 GP-weighted skater quality
    stints["w_z"] = stints["GP"] * stints["pts_pg_z"].fillna(0.0)
    team_y1 = stints.groupby(["tmID", "season_idx"], as_index=False)[["w_z", "GP"]].sum()
    team_y1["team_y1_z"] = _safe_divide(team_y1["w_z"], team_y1["GP"])
    team_y1["season_idx"] = team_y1["season_idx"] + 1

    # Roster membership at season Y (uses only playerID, tmID, season_idx — no Y stats)
    rosters = df[["playerID", "tmID", "season_idx"]].drop_duplicates()
    roster_rated = rosters.merge(player_y1, on=["playerID", "season_idx"], how="inner")
    roster_rated["w_in"] = roster_rated["prev_gp"].clip(lower=1) * roster_rated["prev_pts_z"]
    incoming = roster_rated.groupby(["tmID", "season_idx"], as_index=False)[["w_in", "prev_gp"]].sum()
    incoming["incoming_z"] = _safe_divide(incoming["w_in"], incoming["prev_gp"].clip(lower=1))

    talent = incoming.merge(team_y1[["tmID", "season_idx", "team_y1_z"]], on=["tmID", "season_idx"], how="inner")
    talent["Roster_Talent_Delta"] = talent["incoming_z"] - talent["team_y1_z"]

    out = core_agg[["tmID", "season_idx", "Skater_Core_Delta_P1"]].merge(
        talent[["tmID", "season_idx", "Roster_Talent_Delta"]],
        on=["tmID", "season_idx"],
        how="outer",
    )
    return out[cols]


def build_features(
    combined: pd.DataFrame,
    scoring: pd.DataFrame,
    goalies: pd.DataFrame,
    labels: pd.DataFrame,
    test_year: int,
) -> pd.DataFrame:
    """Assemble one feature row per NHL team-season using only pre-season information."""
    nhl_combined = _filter_nhl(combined)
    year_to_idx = _nhl_season_map(nhl_combined["year"])
    nhl_combined["season_idx"] = nhl_combined["year"].map(year_to_idx)
    test_season_idx = year_to_idx[int(test_year)]

    # ---- Team-season box score features (normalized within each NHL season) ----
    team_cols = ["year", "season_idx", "tmID", "G", "W", "L", "T", "OTL", "GF", "GA", "Pts"]
    team_seasons = nhl_combined.drop_duplicates(["tmID", "year"])[
        [c for c in team_cols if c in nhl_combined.columns]
    ].copy()
    for col in ("G", "W", "L", "T", "OTL", "GF", "GA", "Pts"):
        team_seasons[col] = _numeric(team_seasons, col)

    team_seasons["Pts_Pct"] = _safe_divide(team_seasons["Pts"], 2 * team_seasons["G"])
    team_seasons["GD_pg"] = _safe_divide(team_seasons["GF"] - team_seasons["GA"], team_seasons["G"])
    team_seasons["Pts_Pct_Z"] = _season_zscore(team_seasons, "Pts_Pct")
    team_seasons["GD_pg_Z"] = _season_zscore(team_seasons, "GD_pg")

    base = team_seasons[["year", "season_idx", "tmID"]].copy()

    p1_perf = team_seasons[["tmID", "season_idx", "Pts_Pct_Z", "GD_pg_Z"]].copy()
    p1_perf["season_idx"] = p1_perf["season_idx"] + 1
    p1_perf = p1_perf.rename(columns={"Pts_Pct_Z": "Pts_Pct_Z_P1", "GD_pg_Z": "GD_pg_Z_P1"})
    base = base.merge(p1_perf, on=["tmID", "season_idx"], how="left")

    p2_perf = team_seasons[["tmID", "season_idx", "Pts_Pct_Z"]].copy()
    p2_perf["season_idx"] = p2_perf["season_idx"] + 2
    p2_perf = p2_perf.rename(columns={"Pts_Pct_Z": "Pts_Pct_Z_P2"})
    base = base.merge(p2_perf, on=["tmID", "season_idx"], how="left")

    base["Pts_Pct_Z_Delta"] = base["Pts_Pct_Z_P1"] - base["Pts_Pct_Z_P2"]
    base["Pts_Luck_P1"] = base["Pts_Pct_Z_P1"] - base["GD_pg_Z_P1"]

    # ---- Coach stints & playoff context ----
    coach_cols = ["year", "season_idx", "tmID", "coachID", "stint", "postg"]
    coaches = nhl_combined[[c for c in coach_cols if c in nhl_combined.columns]].copy()
    coaches["stint"] = _numeric(coaches, "stint")
    coaches["postg"] = _numeric(coaches, "postg")
    coaches = coaches.drop_duplicates(["tmID", "season_idx", "coachID", "stint"])

    first, _ = _first_last_coaches(coaches, season_col="season_idx")

    playoff_p1 = _compute_playoff_features(coaches, team_seasons, test_season_idx)
    base = base.merge(playoff_p1, on=["tmID", "season_idx"], how="left")

    past_coaches = coaches[coaches["season_idx"] < test_season_idx]
    midseason = (
        past_coaches.groupby(["tmID", "season_idx"])["coachID"]
        .nunique()
        .gt(1)
        .astype(float)
        .rename("Midseason_Change_P1")
        .reset_index()
    )
    midseason["season_idx"] = midseason["season_idx"] + 1
    base = base.merge(midseason, on=["tmID", "season_idx"], how="left")

    start = first[["tmID", "season_idx", "coachID"]].rename(columns={"coachID": "Start_Coach"})
    base = base.merge(start, on=["tmID", "season_idx"], how="left")

    # Consecutive prior seasons Start_Coach(Y) has coached tmID
    spells = (
        past_coaches[["coachID", "tmID", "season_idx"]]
        .dropna(subset=["coachID"])
        .drop_duplicates()
        .sort_values(["coachID", "tmID", "season_idx"])
    )
    gap = spells.groupby(["coachID", "tmID"])["season_idx"].diff()
    new_run = gap.isna() | (gap > 1)
    spells["run"] = new_run.groupby([spells["coachID"], spells["tmID"]]).cumsum()
    spells["Coach_Tenure"] = spells.groupby(["coachID", "tmID", "run"]).cumcount() + 1

    tenure_p1 = spells[["coachID", "tmID", "season_idx", "Coach_Tenure"]].copy()
    tenure_p1["season_idx"] = tenure_p1["season_idx"] + 1
    tenure_p1 = tenure_p1.rename(columns={"coachID": "Start_Coach"})
    base = base.merge(tenure_p1, on=["tmID", "season_idx", "Start_Coach"], how="left")

    raw_tenure = np.where(
        base["Start_Coach"].isna(),
        np.nan,
        base["Coach_Tenure"].fillna(0.0),
    )
    base["Is_New_Coach"] = np.where(np.isnan(raw_tenure), np.nan, (raw_tenure == 0.0).astype(float))
    base["Coach_Tenure_Capped"] = np.minimum(raw_tenure, float(MAX_TENURE_CAP))

    # ---- Player & goalie performance deltas ----
    goalie_feats = _compute_goalie_features(goalies, year_to_idx, test_season_idx)
    base = base.merge(goalie_feats, on=["tmID", "season_idx"], how="left")

    skater_feats = _compute_skater_features(scoring, year_to_idx, test_season_idx)
    base = base.merge(skater_feats, on=["tmID", "season_idx"], how="left")

    base = base.merge(labels[["year", "tmID", "Coach_Changed"]], on=["year", "tmID"], how="left")
    return base


def train_and_predict(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    *args,
    **kwargs,
) -> pd.DataFrame:
    if train_data.empty or test_data.empty:
        raise ValueError("train_data and test_data must not be empty")

    test_year = int(test_data["year"].max())
    combined = pd.concat([train_data, test_data], ignore_index=True)
    scoring, goalies = _extract_tables(args, kwargs)

    labels = build_history_labels(train_data, train_data)
    data = build_features(combined, scoring, goalies, labels, test_year)

    test_rows = data[data["year"] == test_year].copy()
    if test_rows.empty:
        raise ValueError(f"No usable test rows for test_year={test_year}")

    test_season_idx = int(test_rows["season_idx"].iloc[0])
    train_rows = data[
        (data["season_idx"] < test_season_idx)
        & (data["season_idx"] >= test_season_idx - TRAINING_WINDOW_SEASONS)
    ].dropna(subset=["Coach_Changed"])

    if train_rows.empty:
        raise ValueError("No usable training rows for coach-change prediction")

    # Automatically drop optional player/goalie columns if those tables weren't supplied
    active_features = [f for f in FEATURES if train_rows[f].notna().any()]

    strategy = kwargs.get("strategy", "implied_count")
    model = _make_model()
    model.fit(train_rows[active_features].astype(float), train_rows["Coach_Changed"].astype(int))
    probability = model.predict_proba(test_rows[active_features].astype(float))[:, 1]

    predicted, cutoff_info = select_coach_changes(
        probabilities=probability,
        train_rows=train_rows,
        strategy=strategy,
        features=active_features,
        model_factory=_make_model,
        fixed_fraction=CHANGE_FRACTION,
        min_flags=MIN_FLAGS,
    )

    result = test_rows[["year", "tmID"]].copy()
    result["change_probability"] = probability
    result["predicted_change"] = predicted
    result.attrs["cutoff_info"] = cutoff_info
    return result[["year", "tmID", "change_probability", "predicted_change"]]