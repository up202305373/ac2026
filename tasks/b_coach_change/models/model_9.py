"""Coach-change prediction model 9 (pruned core + career-baseline player/goalie deltas).

Key improvements
----------------
* Prunes noisy / heavily-shrunk features (Coach_Age, Has_Award, Coach_Change_Rate,
  Team_Change_Rate, redundant GD_pg_Z_P1) down to 8 orthogonal features suited for
  short (diff=8) training windows, paired with stronger L2 regularization (C=0.15).
* Replaces collinear New_Start_Coach + linear Coach_Tenure with log1p(Coach_Tenure).
* Replaces high-variance P1-P2 player/goalie deltas with Y-1 performance minus each
  player/goalie's own expanding prior career baseline (< Y-1), eliminating NaNs in
  the earliest seasons of an 8-year window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from coach_thresholding import select_coach_changes

CHANGE_FRACTION = 0.30
MIN_FLAGS = 3

TRAINING_WINDOW_SEASONS = 25
MAX_PLAYOFF_DROUGHT = 6
GOALIE_CAREER_PRIOR_GP = 30.0
GOALIE_SEASON_PRIOR_GP = 15.0
SKATER_RATE_PRIOR_GP = 25.0
SKATER_CAREER_PRIOR_GP = 40.0
SKATER_SEASON_PRIOR_GP = 20.0

FEATURES = [
    "Pts_Pct_Z_P1",
    "Pts_Pct_Z_Delta",
    "Made_Playoffs_P1",
    "Playoff_Drought_P1",
    "Midseason_Change_P1",
    "Coach_Tenure_Log",
    "Goalie_Delta_P1",
    "Skater_Delta_P1",
]


def _make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.15, max_iter=2000),
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


def _compute_goalie_baseline_delta(
    goalies: pd.DataFrame,
    year_to_idx: dict[int, int],
    test_season_idx: int,
) -> pd.DataFrame:
    """Y-1 goalie SV%/-GAA z-score minus each goalie's prior career baseline (< Y-1)."""
    cols = ["tmID", "season_idx", "Goalie_Delta_P1"]
    if goalies.empty or not {"tmID", "year", "GA"}.issubset(goalies.columns):
        return pd.DataFrame(columns=cols)

    df = _filter_nhl(goalies)
    df["season_idx"] = df["year"].map(year_to_idx)
    df = df.dropna(subset=["tmID", "season_idx"]).copy()
    df["season_idx"] = df["season_idx"].astype(int)
    df = df[df["season_idx"] < test_season_idx]
    if df.empty:
        return pd.DataFrame(columns=cols)

    has_player = "playerID" in df.columns and df["playerID"].notna().any()
    if not has_player:
        df["playerID"] = df["tmID"]

    dedup_cols = [c for c in ("playerID", "tmID", "season_idx", "stint") if c in df.columns]
    df = df.drop_duplicates(dedup_cols)

    for col in ("GP", "Min", "GA", "SA"):
        df[col] = _numeric(df, col)

    df["GP_eff"] = df["GP"].fillna(_safe_divide(df["Min"], pd.Series(60.0, index=df.index))).fillna(1.0)
    df = df[df["GP_eff"] > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    stints = (
        df.groupby(["playerID", "tmID", "season_idx"], as_index=False)[["GP_eff", "Min", "GA", "SA"]]
        .sum(min_count=1)
    )
    stints["sv_pct"] = _safe_divide(stints["SA"] - stints["GA"], stints["SA"])
    stints["neg_gaa"] = -60.0 * _safe_divide(stints["GA"], stints["Min"])
    stints["neg_ga_pg"] = -_safe_divide(stints["GA"], stints["GP_eff"])

    # Normalize across qualified goalies within each NHL season
    stints["perf_z"] = (
        _season_zscore(stints, "sv_pct")
        .fillna(_season_zscore(stints, "neg_gaa"))
        .fillna(_season_zscore(stints, "neg_ga_pg"))
        .fillna(0.0)
    )

    # Expanding prior career baseline per goalie strictly before each season_idx
    p_season = (
        stints.assign(w_z=stints["GP_eff"] * stints["perf_z"])
        .groupby(["playerID", "season_idx"], as_index=False)[["w_z", "GP_eff"]]
        .sum()
        .sort_values(["playerID", "season_idx"])
    )
    p_season["cum_wz"] = p_season.groupby("playerID")["w_z"].cumsum() - p_season["w_z"]
    p_season["cum_gp"] = p_season.groupby("playerID")["GP_eff"].cumsum() - p_season["GP_eff"]
    p_season["baseline_z"] = p_season["cum_wz"] / (p_season["cum_gp"] + GOALIE_CAREER_PRIOR_GP)

    stints = stints.merge(
        p_season[["playerID", "season_idx", "baseline_z"]],
        on=["playerID", "season_idx"],
        how="left",
    )
    shrink = stints["GP_eff"] / (stints["GP_eff"] + GOALIE_SEASON_PRIOR_GP)
    stints["goalie_delta"] = shrink * (stints["perf_z"] - stints["baseline_z"].fillna(0.0))
    stints["w_delta"] = stints["GP_eff"] * stints["goalie_delta"]

    team_g = stints.groupby(["tmID", "season_idx"], as_index=False)[["w_delta", "GP_eff"]].sum()
    team_g["Goalie_Delta_P1"] = _safe_divide(team_g["w_delta"], team_g["GP_eff"])
    team_g["season_idx"] = team_g["season_idx"] + 1  # aligns Y-1 delta to target Y
    return team_g[cols]


def _compute_skater_baseline_delta(
    scoring: pd.DataFrame,
    goalies: pd.DataFrame,
    year_to_idx: dict[int, int],
    test_season_idx: int,
) -> pd.DataFrame:
    """Y-1 skater Pts/GP z-score minus each skater's prior career baseline (< Y-1)."""
    cols = ["tmID", "season_idx", "Skater_Delta_P1"]
    if scoring.empty or not {"playerID", "tmID", "year"}.issubset(scoring.columns):
        return pd.DataFrame(columns=cols)

    df = _filter_nhl(scoring)
    df["season_idx"] = df["year"].map(year_to_idx)
    df = df.dropna(subset=["playerID", "tmID", "season_idx"]).copy()
    df["season_idx"] = df["season_idx"].astype(int)
    df = df[df["season_idx"] < test_season_idx]
    if df.empty:
        return pd.DataFrame(columns=cols)

    if "pos" in df.columns:
        df = df[df["pos"].fillna("") != "G"]
    if not goalies.empty and "playerID" in goalies.columns:
        goalie_ids = set(goalies["playerID"].dropna().unique())
        df = df[~df["playerID"].isin(goalie_ids)]

    dedup_cols = [c for c in ("playerID", "tmID", "season_idx", "stint") if c in df.columns]
    df = df.drop_duplicates(dedup_cols)

    for col in ("GP", "G", "A", "Pts"):
        df[col] = _numeric(df, col)
    if "Pts" not in scoring.columns or df["Pts"].isna().all():
        df["Pts"] = df["G"].fillna(0) + df["A"].fillna(0)

    df = df[df["GP"].fillna(0) > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    stints = (
        df.groupby(["playerID", "tmID", "season_idx"], as_index=False)[["GP", "Pts"]]
        .sum(min_count=1)
    )
    season_totals = (
        stints.groupby("season_idx", as_index=False)[["Pts", "GP"]]
        .sum()
        .rename(columns={"Pts": "lg_pts", "GP": "lg_gp"})
    )
    season_totals["lg_rate"] = _safe_divide(season_totals["lg_pts"], season_totals["lg_gp"])
    stints = stints.merge(season_totals[["season_idx", "lg_rate"]], on="season_idx", how="left")
    stints["shrunk_rate"] = (
        stints["Pts"] + SKATER_RATE_PRIOR_GP * stints["lg_rate"]
    ) / (stints["GP"] + SKATER_RATE_PRIOR_GP)
    stints["rate_z"] = _season_zscore(stints, "shrunk_rate").fillna(0.0)

    # Expanding prior career baseline per skater strictly before each season_idx
    p_season = (
        stints.assign(w_z=stints["GP"] * stints["rate_z"])
        .groupby(["playerID", "season_idx"], as_index=False)[["w_z", "GP"]]
        .sum()
        .sort_values(["playerID", "season_idx"])
    )
    p_season["cum_wz"] = p_season.groupby("playerID")["w_z"].cumsum() - p_season["w_z"]
    p_season["cum_gp"] = p_season.groupby("playerID")["GP"].cumsum() - p_season["GP"]
    p_season["baseline_z"] = p_season["cum_wz"] / (p_season["cum_gp"] + SKATER_CAREER_PRIOR_GP)

    stints = stints.merge(
        p_season[["playerID", "season_idx", "baseline_z"]],
        on=["playerID", "season_idx"],
        how="left",
    )
    shrink = stints["GP"] / (stints["GP"] + SKATER_SEASON_PRIOR_GP)
    stints["skater_delta"] = shrink * (stints["rate_z"] - stints["baseline_z"].fillna(0.0))
    stints["w_delta"] = stints["GP"] * stints["skater_delta"]

    team_s = stints.groupby(["tmID", "season_idx"], as_index=False)[["w_delta", "GP"]].sum()
    team_s["Skater_Delta_P1"] = _safe_divide(team_s["w_delta"], team_s["GP"])
    team_s["season_idx"] = team_s["season_idx"] + 1  # aligns Y-1 delta to target Y
    return team_s[cols]


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
    team_cols = ["year", "season_idx", "tmID", "G", "Pts"]
    team_seasons = nhl_combined.drop_duplicates(["tmID", "year"])[
        [c for c in team_cols if c in nhl_combined.columns]
    ].copy()
    team_seasons["G"] = _numeric(team_seasons, "G")
    team_seasons["Pts"] = _numeric(team_seasons, "Pts")
    team_seasons["Pts_Pct"] = _safe_divide(team_seasons["Pts"], 2 * team_seasons["G"])
    team_seasons["Pts_Pct_Z"] = _season_zscore(team_seasons, "Pts_Pct")

    base = team_seasons[["year", "season_idx", "tmID"]].copy()

    p1_perf = team_seasons[["tmID", "season_idx", "Pts_Pct_Z"]].copy()
    p1_perf["season_idx"] = p1_perf["season_idx"] + 1
    p1_perf = p1_perf.rename(columns={"Pts_Pct_Z": "Pts_Pct_Z_P1"})
    base = base.merge(p1_perf, on=["tmID", "season_idx"], how="left")

    p2_perf = team_seasons[["tmID", "season_idx", "Pts_Pct_Z"]].copy()
    p2_perf["season_idx"] = p2_perf["season_idx"] + 2
    p2_perf = p2_perf.rename(columns={"Pts_Pct_Z": "Pts_Pct_Z_P2"})
    base = base.merge(p2_perf, on=["tmID", "season_idx"], how="left")

    # Neutral 0.0 delta when P1 exists but P2 falls outside short training windows
    raw_delta = base["Pts_Pct_Z_P1"] - base["Pts_Pct_Z_P2"]
    base["Pts_Pct_Z_Delta"] = np.where(base["Pts_Pct_Z_P1"].notna(), raw_delta.fillna(0.0), np.nan)

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
    base["Coach_Tenure_Log"] = np.log1p(raw_tenure)

    # ---- Career-baseline player & goalie performance deltas ----
    goalie_delta = _compute_goalie_baseline_delta(goalies, year_to_idx, test_season_idx)
    base = base.merge(goalie_delta, on=["tmID", "season_idx"], how="left")

    skater_delta = _compute_skater_baseline_delta(scoring, goalies, year_to_idx, test_season_idx)
    base = base.merge(skater_delta, on=["tmID", "season_idx"], how="left")

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

    scoring, goalies = _extract_tables(args, kwargs)
    test_year = int(test_data["year"].max())
    combined = pd.concat([train_data, test_data], ignore_index=True)

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

    active_features = [
        f for f in FEATURES if train_rows[f].notna().any() and test_rows[f].notna().any()
    ]

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