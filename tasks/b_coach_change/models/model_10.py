"""Coach-change prediction model 10 (warning-free, diff=1 support, core player/goalie deltas).

Key improvements
----------------
* Eliminates all SimpleImputer warnings across diff=2, 3, 5, 8 by filtering
  active_features to columns with >1 unique non-null value in train_rows and
  at least one observed value in test_rows.
* Handles diff=1 gracefully via a pre-season risk score computed from test_year-1
  standings/playoff/coaching signals instead of raising on empty train labels.
* Retains model_6's proven 12-feature coaching/standings foundation while adding:
  1. Coach_Blame_Goalie_Gap_P1 (Goalie_Perf_Z_P1 - Pts_Pct_Z_P1): flags teams
     whose standings underperformed relative to their goaltending (available even
     at diff=3 since it only requires Y-1).
  2. Top_Skater_Core_Delta_P1: GP-weighted YoY Pts/GP z-score delta (Y-2 -> Y-1)
     restricted to each team's top-12 core skaters with >=35 GP in both seasons.
* Calibrates CHANGE_FRACTION = 0.30 with C=0.35 L2 regularization to maintain >50%
  precision on flagged coach changes.
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
TEAM_RATE_WINDOW_SEASONS = 15
COACH_PRIOR_STRENGTH = 5.0
TEAM_PRIOR_STRENGTH = 6.0
MAX_PLAYOFF_DROUGHT = 6
CORE_SKATER_MIN_GP = 35
CORE_SKATER_TOP_N = 12

FEATURES = [
    "Pts_Pct_Z_P1",
    "GD_pg_Z_P1",
    "Pts_Pct_Z_Delta",
    "Made_Playoffs_P1",
    "Playoff_Drought_P1",
    "Midseason_Change_P1",
    "New_Start_Coach",
    "Coach_Tenure",
    "Coach_Age",
    "Has_Award",
    "Coach_Change_Rate",
    "Team_Change_Rate",
    "Coach_Blame_Goalie_Gap_P1",
    "Top_Skater_Core_Delta_P1",
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


def _compute_goalie_perf_p1(
    goalies: pd.DataFrame,
    year_to_idx: dict[int, int],
    test_season_idx: int,
) -> pd.DataFrame:
    """Prior-season team goalie SV% (with -GAA fallback) z-scored within each NHL season."""
    cols = ["tmID", "season_idx", "Goalie_Perf_Z_P1"]
    if goalies.empty or not {"tmID", "year", "GA"}.issubset(goalies.columns):
        return pd.DataFrame(columns=cols)

    df = _filter_nhl(goalies)
    df["season_idx"] = df["year"].map(year_to_idx)
    df = df.dropna(subset=["tmID", "season_idx"]).copy()
    df["season_idx"] = df["season_idx"].astype(int)
    df = df[df["season_idx"] < test_season_idx]
    if df.empty:
        return pd.DataFrame(columns=cols)

    dedup_cols = [c for c in ("playerID", "tmID", "season_idx", "stint") if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(dedup_cols)

    for col in ("GP", "Min", "GA", "SA"):
        df[col] = _numeric(df, col)

    team_g = (
        df.groupby(["tmID", "season_idx"], as_index=False)[["GP", "Min", "GA", "SA"]]
        .sum(min_count=1)
    )
    team_g["sv_pct"] = _safe_divide(team_g["SA"] - team_g["GA"], team_g["SA"])
    team_g["neg_gaa"] = -60.0 * _safe_divide(team_g["GA"], team_g["Min"])
    team_g["neg_ga_pg"] = -_safe_divide(team_g["GA"], team_g["GP"])

    team_g["Goalie_Perf_Z_P1"] = (
        _season_zscore(team_g, "sv_pct")
        .fillna(_season_zscore(team_g, "neg_gaa"))
        .fillna(_season_zscore(team_g, "neg_ga_pg"))
    )
    team_g["season_idx"] = team_g["season_idx"] + 1
    return team_g[cols]


def _compute_top_skater_core_delta(
    scoring: pd.DataFrame,
    goalies: pd.DataFrame,
    year_to_idx: dict[int, int],
    test_season_idx: int,
) -> pd.DataFrame:
    """GP-weighted YoY Pts/GP z-score delta (Y-2 -> Y-1) for top-12 core skaters (>=35 GP)."""
    cols = ["tmID", "season_idx", "Top_Skater_Core_Delta_P1"]
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
    if dedup_cols:
        df = df.drop_duplicates(dedup_cols)

    for col in ("GP", "G", "A", "Pts"):
        df[col] = _numeric(df, col)
    if "Pts" not in scoring.columns or df["Pts"].isna().all():
        df["Pts"] = df["G"].fillna(0) + df["A"].fillna(0)

    stints = (
        df.groupby(["playerID", "tmID", "season_idx"], as_index=False)[["GP", "Pts"]]
        .sum(min_count=1)
    )
    stints = stints[stints["GP"].fillna(0) >= CORE_SKATER_MIN_GP].copy()
    if stints.empty:
        return pd.DataFrame(columns=cols)

    # Keep top-12 scorers per team-season to focus on core lineup drivers
    stints = stints.sort_values(["tmID", "season_idx", "Pts", "GP"], ascending=[True, True, False, False])
    stints = stints.groupby(["tmID", "season_idx"], as_index=False).head(CORE_SKATER_TOP_N).copy()

    stints["pts_pg"] = _safe_divide(stints["Pts"], stints["GP"])
    stints["rate_z"] = _season_zscore(stints, "pts_pg")

    curr = stints[["playerID", "tmID", "season_idx", "GP", "rate_z"]].copy()
    prev = stints[["playerID", "tmID", "season_idx", "GP", "rate_z"]].copy()
    prev["season_idx"] = prev["season_idx"] + 1
    prev = prev.rename(columns={"GP": "GP_prev", "rate_z": "rate_z_prev"})

    returning = curr.merge(prev, on=["playerID", "tmID", "season_idx"], how="inner")
    if returning.empty:
        return pd.DataFrame(columns=cols)

    returning["w"] = np.minimum(returning["GP"], returning["GP_prev"])
    returning["w_delta"] = returning["w"] * (returning["rate_z"] - returning["rate_z_prev"])

    agg = returning.groupby(["tmID", "season_idx"], as_index=False)[["w_delta", "w"]].sum()
    agg["Top_Skater_Core_Delta_P1"] = _safe_divide(agg["w_delta"], agg["w"])
    agg["season_idx"] = agg["season_idx"] + 1  # aligns (Y-2 -> Y-1) delta to target Y
    return agg[cols]


def build_features(
    combined: pd.DataFrame,
    awards: pd.DataFrame,
    master: pd.DataFrame,
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

    # ---- Coach stints (deduplicated per stint) ----
    coach_cols = ["year", "season_idx", "tmID", "coachID", "stint", "postg"]
    coaches = nhl_combined[[c for c in coach_cols if c in nhl_combined.columns]].copy()
    coaches["stint"] = _numeric(coaches, "stint")
    coaches["postg"] = _numeric(coaches, "postg")
    coaches = coaches.drop_duplicates(["tmID", "season_idx", "coachID", "stint"])

    first, last = _first_last_coaches(coaches, season_col="season_idx")

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

    prior_end = last[last["season_idx"] < test_season_idx][["tmID", "season_idx", "coachID"]].copy()
    prior_end["season_idx"] = prior_end["season_idx"] + 1
    prior_end = prior_end.rename(columns={"coachID": "End_Coach_P1"})
    base = base.merge(prior_end, on=["tmID", "season_idx"], how="left")

    known_pair = base["Start_Coach"].notna() & base["End_Coach_P1"].notna()
    base["New_Start_Coach"] = np.where(
        known_pair,
        (base["Start_Coach"] != base["End_Coach_P1"]).astype(float),
        np.nan,
    )

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
    base["Coach_Tenure"] = np.where(
        base["Start_Coach"].isna(),
        np.nan,
        base["Coach_Tenure"].fillna(0.0),
    )

    if master is not None and not master.empty and {"coachID", "birthYear"}.issubset(master.columns):
        ages = (
            master[["coachID", "birthYear"]]
            .dropna(subset=["coachID"])
            .drop_duplicates("coachID")
            .copy()
        )
        ages["birthYear"] = pd.to_numeric(ages["birthYear"], errors="coerce")
        ages = ages.rename(columns={"coachID": "Start_Coach"})
        base = base.merge(ages, on="Start_Coach", how="left")
        base["Coach_Age"] = base["year"] - base["birthYear"]
    else:
        base["Coach_Age"] = np.nan

    if awards is not None and not awards.empty and {"coachID", "year"}.issubset(awards.columns):
        won = awards[["coachID", "year"]].copy()
        won["year"] = _numeric(won, "year")
        won["season_idx"] = won["year"].map(year_to_idx)
        won = won.dropna(subset=["coachID", "season_idx"]).copy()
        won["season_idx"] = won["season_idx"].astype(int)
        recent = pd.concat(
            [won.assign(season_idx=won["season_idx"] + offset) for offset in (1, 2, 3)],
            ignore_index=True,
        )
        recent = recent[["coachID", "season_idx"]].drop_duplicates()
        recent["Has_Award"] = 1.0
        recent = recent.rename(columns={"coachID": "Start_Coach"})
        base = base.merge(recent, on=["Start_Coach", "season_idx"], how="left")
        base["Has_Award"] = base["Has_Award"].fillna(0.0)
    else:
        base["Has_Award"] = 0.0

    # ---- Leakage-free, Bayesian-smoothed coach & franchise change rates ----
    history = labels[["year", "tmID", "Coach_Changed"]].copy()
    history["season_idx"] = pd.to_numeric(history["year"], errors="coerce").map(year_to_idx)
    history = history.dropna(subset=["season_idx", "Coach_Changed"]).copy()
    history["season_idx"] = history["season_idx"].astype(int)
    history = history.merge(start, on=["tmID", "season_idx"], how="left")

    base["Coach_Change_Rate"] = np.nan
    base["Team_Change_Rate"] = np.nan
    for s in sorted(base["season_idx"].unique()):
        past = history[history["season_idx"] <= s - 2]
        prior = float(past["Coach_Changed"].mean()) if not past.empty else CHANGE_FRACTION

        coach_past = past.dropna(subset=["Start_Coach"])
        if not coach_past.empty:
            c_agg = coach_past.groupby("Start_Coach")["Coach_Changed"].agg(["sum", "count"])
            c_rates = (
                (c_agg["sum"] + COACH_PRIOR_STRENGTH * prior)
                / (c_agg["count"] + COACH_PRIOR_STRENGTH)
            ).to_dict()
        else:
            c_rates = {}

        team_past = past[past["season_idx"] >= s - TEAM_RATE_WINDOW_SEASONS]
        if not team_past.empty:
            t_agg = team_past.groupby("tmID")["Coach_Changed"].agg(["sum", "count"])
            t_rates = (
                (t_agg["sum"] + TEAM_PRIOR_STRENGTH * prior)
                / (t_agg["count"] + TEAM_PRIOR_STRENGTH)
            ).to_dict()
        else:
            t_rates = {}

        mask = base["season_idx"] == s
        base.loc[mask, "Coach_Change_Rate"] = [
            np.nan if pd.isna(c) else c_rates.get(c, prior)
            for c in base.loc[mask, "Start_Coach"]
        ]
        base.loc[mask, "Team_Change_Rate"] = [
            t_rates.get(t, prior) for t in base.loc[mask, "tmID"]
        ]

    # ---- Focused player & goalie performance deltas ----
    goalie_p1 = _compute_goalie_perf_p1(goalies, year_to_idx, test_season_idx)
    base = base.merge(goalie_p1, on=["tmID", "season_idx"], how="left")
    base["Coach_Blame_Goalie_Gap_P1"] = base["Goalie_Perf_Z_P1"] - base["Pts_Pct_Z_P1"]

    skater_delta = _compute_top_skater_core_delta(scoring, goalies, year_to_idx, test_season_idx)
    base = base.merge(skater_delta, on=["tmID", "season_idx"], how="left")

    base = base.merge(labels[["year", "tmID", "Coach_Changed"]], on=["year", "tmID"], how="left")
    return base


def _diff1_prior_predictions(test_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fallback risk score when diff=1 leaves zero labeled season-to-season transitions."""
    pts = test_rows["Pts_Pct_Z_P1"].fillna(0.0)
    missed_po = 1.0 - test_rows["Made_Playoffs_P1"].fillna(0.5)
    midseason = test_rows["Midseason_Change_P1"].fillna(0.0)
    new_coach = test_rows["New_Start_Coach"].fillna(0.0)

    raw_score = -0.9 * pts + 0.5 * missed_po + 0.6 * midseason - 0.8 * new_coach
    z = (raw_score - raw_score.mean()) / (raw_score.std() or 1.0)
    prob = 1.0 / (1.0 + np.exp(-(np.log(CHANGE_FRACTION / (1.0 - CHANGE_FRACTION)) + 0.6 * z)))

    n_flags = max(MIN_FLAGS, int(round(CHANGE_FRACTION * len(test_rows))))
    cutoff = np.partition(prob, -n_flags)[-n_flags]
    pred = (prob >= cutoff).astype(int)
    return prob.to_numpy(dtype=float), pred.to_numpy(dtype=int)


def train_and_predict(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    *args,
    **kwargs,
) -> pd.DataFrame:
    if train_data.empty or test_data.empty:
        raise ValueError("train_data and test_data must not be empty")

    train_awards = kwargs.get("train_awards")
    if train_awards is None and len(args) >= 2:
        train_awards = args[1]
    train_master = kwargs.get("train_master")
    if train_master is None and len(args) >= 8:
        train_master = args[7]
    test_master = kwargs.get("test_master")
    if test_master is None and len(args) >= 9:
        test_master = args[8]

    scoring, goalies = _extract_tables(args, kwargs)

    test_year = int(test_data["year"].max())
    combined = pd.concat([train_data, test_data], ignore_index=True)
    awards = train_awards if train_awards is not None else pd.DataFrame()
    master = pd.concat(
        [frame for frame in (train_master, test_master) if frame is not None and not frame.empty],
        ignore_index=True,
    )

    labels = build_history_labels(train_data, train_data)
    data = build_features(combined, awards, master, scoring, goalies, labels, test_year)

    test_rows = data[data["year"] == test_year].copy()
    if test_rows.empty:
        raise ValueError(f"No usable test rows for test_year={test_year}")

    test_season_idx = int(test_rows["season_idx"].iloc[0])
    train_rows = data[
        (data["season_idx"] < test_season_idx)
        & (data["season_idx"] >= test_season_idx - TRAINING_WINDOW_SEASONS)
    ].dropna(subset=["Coach_Changed"])

    # Handle diff=1 (single prior season -> no observable s -> s+1 transition in train_data)
    if train_rows.empty or train_rows["Coach_Changed"].nunique() < 2:
        probability, predicted = _diff1_prior_predictions(test_rows)
        result = test_rows[["year", "tmID"]].copy()
        result["change_probability"] = probability
        result["predicted_change"] = predicted
        result.attrs["cutoff_info"] = {"strategy": "diff1_prior_fallback"}
        return result[["year", "tmID", "change_probability", "predicted_change"]]

    # Strictly require >1 distinct observed value in train_rows and >=1 in test_rows
    # so SimpleImputer never skips an empty column or emits a UserWarning.
    active_features = [
        f
        for f in FEATURES
        if train_rows[f].dropna().nunique() > 1 and test_rows[f].notna().any()
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