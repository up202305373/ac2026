"""Coach-change prediction model 4.

Key differences from models 1-3
-------------------------------
* The sampler blanks every in-season performance column in the test split
  (``W``, ``L``, ``Pts``, ``GF``, ``GA``, ... in Teams and the coaching stint
  statistics).  Training on *same-season* performance and then predicting the
  test season therefore feeds the model a feature vector it has never seen:
  all of those values are missing.  Here every performance feature is taken
  from the **previous** completed season, which is available for both the
  training targets and the test year.
* The target is built exactly like ``evaluate.py`` does it: the coach who
  *finished* season Y must differ from the coach who *finished* season Y+1.
  Rows whose following season is unavailable are dropped instead of being
  silently mislabelled as a change.
* Coaching changes are rare and the evaluation set is small, so a fixed
  probability cutoff is unstable.  Teams are ranked by predicted risk and the
  top ~1/3 of them (about ten teams in a 30-team league) are flagged.  This
  keeps both recall on the few real changes and precision on the many stable
  teams.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from coach_thresholding import select_coach_changes

def _make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000),
    )

# Roughly one third of NHL franchises change coach in a typical season.
CHANGE_FRACTION = 0.34
MIN_FLAGS = 3


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.replace(0, np.nan)
    return numerator / den


def _first_last_coaches(coaches: pd.DataFrame):
    ordered = coaches.sort_values(["tmID", "year", "stint"])
    first = ordered.drop_duplicates(["tmID", "year"], keep="first")
    last = ordered.drop_duplicates(["tmID", "year"], keep="last")
    return first, last


def build_history_labels(train_coaches: pd.DataFrame, train_teams: pd.DataFrame) -> pd.DataFrame:
    """One labelled row per (tmID, year), using next season's finishing coach.

    ``Coach_Changed`` is NaN when the following season is missing; those rows
    are not usable for supervised training (they are not genuine "0" labels).
    """
    _, last = _first_last_coaches(train_coaches)
    labelled = last[["tmID", "year", "coachID"]].copy()
    following = labelled.copy()
    following["year"] = following["year"] - 1
    following = following.rename(columns={"coachID": "Next_Coach"})
    labelled = labelled.merge(following, on=["tmID", "year"], how="left")
    labelled["Coach_Changed"] = np.where(
        labelled["Next_Coach"].isna(),
        np.nan,
        (labelled["coachID"] != labelled["Next_Coach"]).astype(float),
    )
    league_col = "lgID" if "lgID" in train_teams.columns else (
        "lgID_y" if "lgID_y" in train_teams.columns else "lgID_x"
    )
    nhl = train_teams.loc[train_teams[league_col] == "NHL", ["year", "tmID"]].drop_duplicates()
    return labelled.merge(nhl, on=["year", "tmID"], how="inner")


def build_features(
    combined: pd.DataFrame,
    awards: pd.DataFrame,
    master: pd.DataFrame,
    labels: pd.DataFrame,
    test_year: int,
) -> pd.DataFrame:
    """Assemble one feature row per team-season from information known at the time."""
    league_col = "lgID" if "lgID" in combined.columns else (
        "lgID_y" if "lgID_y" in combined.columns else "lgID_x"
    )
    team_cols = ["year", "tmID", league_col, "G", "W", "L", "T", "OTL", "GF", "GA", "Pts"]
    team_seasons = combined.drop_duplicates(["tmID", "year"])[
        [c for c in team_cols if c in combined.columns]
    ].copy()
    team_seasons = team_seasons.rename(columns={league_col: "lgID"})
    for col in ["G", "W", "L", "T", "OTL", "GF", "GA", "Pts"]:
        team_seasons[col] = _numeric(team_seasons, col)

    team_seasons["Pts_Pct"] = _safe_divide(team_seasons["Pts"], 2 * team_seasons["G"])
    team_seasons["GD_pg"] = _safe_divide(team_seasons["GF"] - team_seasons["GA"], team_seasons["G"])

    base = team_seasons[team_seasons["lgID"] == "NHL"][["year", "tmID"]].copy()

    # ---- Prior season form (the only form available in the test split) ----
    previous = team_seasons[["tmID", "year", "Pts_Pct", "GD_pg"]].copy()
    previous["year"] = previous["year"] + 1
    previous = previous.rename(columns={"Pts_Pct": "Pts_Pct_P1", "GD_pg": "GD_pg_P1"})
    base = base.merge(previous, on=["tmID", "year"], how="left")

    coach_cols = ["year", "tmID", "coachID", "stint", "g", "w", "l", "t", "postg"]
    coaches = combined[[c for c in coach_cols if c in combined.columns]].copy()
    coaches["stint"] = _numeric(coaches, "stint")
    first, last = _first_last_coaches(coaches)

    start = first[["tmID", "year", "coachID"]].rename(columns={"coachID": "Start_Coach"})
    base = base.merge(start, on=["tmID", "year"], how="left")

    prior_end = last[["tmID", "year", "coachID"]].rename(columns={"coachID": "End_Coach_P1"})
    prior_end["next_year"] = prior_end["year"] + 1
    base = base.merge(prior_end, left_on=["tmID", "year"], right_on=["tmID", "next_year"], how="left")
    base = base.drop(columns=["next_year"])
    if "year_x" in base.columns:
        base = base.rename(columns={"year_x": "year"}).drop(columns=["year_y"])

    base["New_Start_Coach"] = (base["Start_Coach"] != base["End_Coach_P1"]).astype(int)

    # ---- Tenure of the coach who finished the previous season ----
    spells = coaches[["coachID", "tmID", "year"]].drop_duplicates().sort_values(
        ["coachID", "tmID", "year"]
    )
    gap = spells.groupby(["coachID", "tmID"])["year"].diff()
    new_run = gap.isna() | (gap > 2)  # a gap > 2 tolerates the 2004-05 lockout
    spells["run"] = new_run.groupby([spells["coachID"], spells["tmID"]]).cumsum()
    spells["Tenure"] = spells.groupby(["coachID", "tmID", "run"]).cumcount() + 1
    tenure = spells[spells["year"] < test_year][["coachID", "tmID", "year", "Tenure"]].copy()
    tenure["year"] = tenure["year"] + 1
    tenure = tenure.rename(columns={"coachID": "End_Coach_P1", "Tenure": "Coach_Tenure_P1"})
    base = base.merge(tenure, on=["tmID", "year", "End_Coach_P1"], how="left")

    # ---- Age of the coach starting the season ----
    if master is not None and not master.empty and "birthYear" in master.columns:
        ages = master[["coachID", "birthYear"]].drop_duplicates("coachID").copy()
        ages["birthYear"] = pd.to_numeric(ages["birthYear"], errors="coerce")
        ages = ages.rename(columns={"coachID": "Start_Coach"})
        base = base.merge(ages, on="Start_Coach", how="left")
        base["Coach_Age"] = base["year"] - base["birthYear"]
    else:
        base["Coach_Age"] = np.nan

    # ---- Recent coaching award (a strong retention signal) ----
    if awards is not None and not awards.empty:
        won = awards[["coachID", "year"]].copy()
        won["year"] = _numeric(won, "year")
        recent = pd.concat(
            [won.assign(year=won["year"] + offset) for offset in (1, 2, 3)], ignore_index=True
        )
        recent = recent[["coachID", "year"]].drop_duplicates()
        recent["Has_Award"] = 1
        recent = recent.rename(columns={"coachID": "Start_Coach"})
        base = base.merge(recent, on=["Start_Coach", "year"], how="left")
        base["Has_Award"] = base["Has_Award"].fillna(0)
    else:
        base["Has_Award"] = 0

    # ---- Historical firing rate of the outgoing coach, using past data only ----
    end_coach_map = prior_end.rename(columns={"End_Coach_P1": "History_Coach"})[
        ["tmID", "year", "History_Coach"]
    ].copy()
    # prior_end is indexed by the season the coach finished; the label for that
    # season belongs to the following year, so shift the key to line the
    # historical coach up with the row whose change we are predicting.
    end_coach_map["year"] = end_coach_map["year"] + 1
    history = labels.merge(end_coach_map, on=["tmID", "year"], how="left")
    history = history.dropna(subset=["History_Coach", "Coach_Changed"])
    rates = {}
    for year in sorted(base["year"].unique()):
        past = history[history["year"] < year]
        rates[year] = past.groupby("History_Coach")["Coach_Changed"].mean()
    base["Coach_Change_Rate"] = [
        rates[year].get(coach, np.nan)
        for year, coach in zip(base["year"], base["End_Coach_P1"])
    ]

    base = base.merge(labels[["year", "tmID", "Coach_Changed"]], on=["year", "tmID"], how="left")
    return base


FEATURES = [
    "Pts_Pct_P1",
    "GD_pg_P1",
    "New_Start_Coach",
    "Coach_Tenure_P1",
    "Coach_Age",
    "Has_Award",
    "Coach_Change_Rate",
]


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

    train_rows = data[(data["year"] < test_year) & (data["year"] >= max(1960, test_year - 45))]
    train_rows = train_rows.dropna(subset=["Coach_Changed"])
    test_rows = data[data["year"] == test_year].copy()

    if train_rows.empty or test_rows.empty:
        raise ValueError("No usable training or test rows for coach-change prediction")

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
    result.attrs["cutoff_info"] = cutoff_info  # inspect chosen threshold/fraction later
    return result[["year", "tmID", "change_probability", "predicted_change"]]
