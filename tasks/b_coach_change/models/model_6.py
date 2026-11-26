"""Coach-change prediction model 4 (refactored).

Key fixes & improvements
------------------------
* Contiguous NHL ``season_idx`` indexing so the 2004-05 lockout does not break
  prior-season lookups, tenure streaks, award windows, or 2003 training labels.
* All coach-level features (``Coach_Tenure``, ``Coach_Change_Rate``,
  ``Coach_Age``, ``Has_Award``) are aligned to ``Start_Coach(Y)`` rather than
  the outgoing coach from ``Y-1``.
* Historical change rates (``Coach_Change_Rate`` and rolling franchise
  ``Team_Change_Rate``) strictly filter to ``season_idx <= s - 2`` (zero target
  leakage or train/test skew) and apply Bayesian smoothing toward the prior.
* Prior-season performance (``Pts_Pct``, ``GD_pg``) is z-scored within each NHL
  season to normalize across rule eras (OTL, post-2005 shootout), alongside
  2-year momentum, playoff status/drought, and prior mid-season firing signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from coach_thresholding import select_coach_changes

# Roughly one third of NHL franchises change coach in a typical season.
CHANGE_FRACTION = 0.34
MIN_FLAGS = 3

# Modeling and smoothing hyperparameters.
TRAINING_WINDOW_SEASONS = 25
TEAM_RATE_WINDOW_SEASONS = 15
COACH_PRIOR_STRENGTH = 5.0
TEAM_PRIOR_STRENGTH = 6.0
MAX_PLAYOFF_DROUGHT = 6

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
]


def _make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=2000),
    )


def _league_col(frame: pd.DataFrame) -> str:
    if "lgID" in frame.columns:
        return "lgID"
    if "lgID_y" in frame.columns:
        return "lgID_y"
    return "lgID_x"


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


def build_history_labels(train_coaches: pd.DataFrame, train_teams: pd.DataFrame) -> pd.DataFrame:
    """One labelled row per NHL (tmID, year), using the next NHL season's finishing coach.

    Uses contiguous NHL season indices so 2003 properly pairs with 2005 across
    the 2004-05 lockout. Rows without a following NHL season get NaN labels.
    """
    league_col = _league_col(train_teams)
    nhl = train_teams.loc[train_teams[league_col] == "NHL", ["year", "tmID"]].drop_duplicates().copy()
    nhl["year"] = pd.to_numeric(nhl["year"], errors="coerce").astype("Int64")
    nhl = nhl.dropna(subset=["year"])
    nhl["year"] = nhl["year"].astype(int)

    year_to_idx = _nhl_season_map(nhl["year"])
    nhl["season_idx"] = nhl["year"].map(year_to_idx)

    coaches = train_coaches.copy()
    coaches["stint"] = _numeric(coaches, "stint")
    coaches["year"] = pd.to_numeric(coaches["year"], errors="coerce")
    coaches = coaches.dropna(subset=["year"]).copy()
    coaches["year"] = coaches["year"].astype(int)

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


def build_features(
    combined: pd.DataFrame,
    awards: pd.DataFrame,
    master: pd.DataFrame,
    labels: pd.DataFrame,
    test_year: int,
) -> pd.DataFrame:
    """Assemble one feature row per NHL team-season using only pre-season information."""
    league_col = _league_col(combined)
    nhl_combined = combined[combined[league_col] == "NHL"].copy()
    nhl_combined["year"] = pd.to_numeric(nhl_combined["year"], errors="coerce")
    nhl_combined = nhl_combined.dropna(subset=["year"]).copy()
    nhl_combined["year"] = nhl_combined["year"].astype(int)

    year_to_idx = _nhl_season_map(nhl_combined["year"])
    nhl_combined["season_idx"] = nhl_combined["year"].map(year_to_idx)
    test_season_idx = year_to_idx[int(test_year)]

    # ---- Team-season box score features (normalized within each NHL season) ----
    team_cols = ["year", "season_idx", "tmID", "G", "W", "L", "T", "OTL", "GF", "GA", "Pts"]
    team_seasons = nhl_combined.drop_duplicates(["tmID", "year"])[
        [c for c in team_cols if c in nhl_combined.columns]
    ].copy()
    for col in ["G", "W", "L", "T", "OTL", "GF", "GA", "Pts"]:
        team_seasons[col] = _numeric(team_seasons, col)

    team_seasons["Pts_Pct"] = _safe_divide(team_seasons["Pts"], 2 * team_seasons["G"])
    team_seasons["GD_pg"] = _safe_divide(team_seasons["GF"] - team_seasons["GA"], team_seasons["G"])
    team_seasons["Pts_Pct_Z"] = _season_zscore(team_seasons, "Pts_Pct")
    team_seasons["GD_pg_Z"] = _season_zscore(team_seasons, "GD_pg")

    base = team_seasons[["year", "season_idx", "tmID"]].copy()

    # Prior season (P1) and two seasons ago (P2) form
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

    # Playoff appearance and drought from completed prior seasons
    playoff_p1 = _compute_playoff_features(coaches, team_seasons, test_season_idx)
    base = base.merge(playoff_p1, on=["tmID", "season_idx"], how="left")

    # Mid-season coaching change in prior completed season
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

    # Starting coach of season Y and finishing coach of season Y-1
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

    # ---- Consecutive prior seasons Start_Coach(Y) has coached tmID (0 for new hires) ----
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

    # ---- Age of Start_Coach(Y) ----
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

    # ---- Recent coaching award for Start_Coach(Y) (prior 3 NHL seasons) ----
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
    # Label at season_idx = s depends on End_Coach(s+1), so at the start of
    # target season t only outcomes with season_idx <= t - 2 are finalized.
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

    train_awards = kwargs.get("train_awards")
    if train_awards is None and len(args) >= 2:
        train_awards = args[1]
    train_master = kwargs.get("train_master")
    if train_master is None and len(args) >= 8:
        train_master = args[7]
    test_master = kwargs.get("test_master")
    if test_master is None and len(args) >= 9:
        test_master = args[8]

    test_year = int(test_data["year"].max())
    combined = pd.concat([train_data, test_data], ignore_index=True)
    awards = train_awards if train_awards is not None else pd.DataFrame()
    master = pd.concat(
        [frame for frame in (train_master, test_master) if frame is not None and not frame.empty],
        ignore_index=True,
    )

    labels = build_history_labels(train_data, train_data)
    data = build_features(combined, awards, master, labels, test_year)

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

    strategy = kwargs.get("strategy", "implied_count")

    model = _make_model()
    model.fit(train_rows[FEATURES].astype(float), train_rows["Coach_Changed"].astype(int))
    probability = model.predict_proba(test_rows[FEATURES].astype(float))[:, 1]

    predicted, cutoff_info = select_coach_changes(
        probabilities=probability,
        train_rows=train_rows,
        strategy=strategy,
        features=FEATURES,
        model_factory=_make_model,
        fixed_fraction=CHANGE_FRACTION,
        min_flags=MIN_FLAGS,
    )

    result = test_rows[["year", "tmID"]].copy()
    result["change_probability"] = probability
    result["predicted_change"] = predicted
    result.attrs["cutoff_info"] = cutoff_info
    return result[["year", "tmID", "change_probability", "predicted_change"]]