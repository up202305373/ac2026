"""Coach-change prediction model 11 (TeamsHalf, TeamSplits & TeamsPost integration).

Key improvements over model_10
------------------------------
* Adds temporal intra-season trajectory features from TeamsHalf and TeamSplits:
  1. Late_Season_Collapse_P1: second-half / Jan-Apr points% z-score minus
     first-half / Oct-Dec points% z-score (flags late-season collapses).
  2. Road_Perf_Z_P1: prior-season road points% z-score from TeamSplits.
* Adds playoff expectation gap from TeamsPost (with Coaches postw fallback):
  3. Playoff_Overachieve_P1: normalized playoff wins minus regular-season
     Pts_Pct_Z_P1 for playoff teams (penalizes early exits by top seeds).
* Scales L2 regularization C smoothly with training sample size to prevent
  overfitting on short diff windows while remaining warning-free across all diffs.
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
    "Late_Season_Collapse_P1",
    "Road_Perf_Z_P1",
    "Playoff_Overachieve_P1",
]


def _make_model(c_value: float = 0.30):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=c_value, max_iter=2000),
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


def _numeric_ci(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    """Case-insensitive numeric column lookup across candidate column names."""
    lower_map = {c.lower(): c for c in frame.columns}
    for cand in candidates:
        actual = lower_map.get(cand.lower())
        if actual is not None:
            return pd.to_numeric(frame[actual], errors="coerce")
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


def _extract_tables(args: tuple, kwargs: dict) -> dict[str, pd.DataFrame]:
    """Locate auxiliary tables (Scoring, Goalies, TeamsHalf, TeamSplits, TeamsPost)."""
    buckets: dict[str, list[pd.DataFrame]] = {
        "scoring": [],
        "goalies": [],
        "teams_half": [],
        "team_splits": [],
        "teams_post": [],
    }

    key_map = {
        "scoring": ("train_scoring", "test_scoring", "scoring", "train_players", "test_players"),
        "goalies": ("train_goalies", "test_goalies", "goalies"),
        "teams_half": ("train_teams_half", "test_teams_half", "teams_half", "teamshalf"),
        "team_splits": ("train_team_splits", "test_team_splits", "team_splits", "teamsplits"),
        "teams_post": ("train_teams_post", "test_teams_post", "teams_post", "teamspost"),
    }
    for bucket, keys in key_map.items():
        for k in keys:
            val = kwargs.get(k)
            if isinstance(val, pd.DataFrame) and not val.empty:
                buckets[bucket].append(val)

    # Also scan any remaining kwargs or positional args by column signature
    all_frames = list(args) + [
        v for k, v in kwargs.items() if isinstance(v, pd.DataFrame) and k not in ("train_data", "test_data")
    ]
    for df in all_frames:
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        cols = set(df.columns)
        cols_lower = {c.lower() for c in cols}
        if {"playerID", "tmID", "year"}.issubset(cols):
            if {"A", "Pts"}.issubset(cols) or "SOG" in cols or "PPG" in cols:
                buckets["scoring"].append(df)
            elif "GA" in cols and ("Min" in cols or "SA" in cols or "SHO" in cols):
                buckets["goalies"].append(df)
        elif {"tmid", "year", "half"}.issubset(cols_lower):
            buckets["teams_half"].append(df)
        elif {"tmid", "year", "hw", "rw"}.issubset(cols_lower):
            buckets["team_splits"].append(df)
        elif (
            {"tmid", "year", "g", "w", "l", "gf", "ga"}.issubset(cols_lower)
            and not {"pts", "rank", "playoff", "coachid", "playerid", "half"}.intersection(cols_lower)
        ):
            buckets["teams_post"].append(df)

    return {
        k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame())
        for k, v in buckets.items()
    }


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
    teams_post: pd.DataFrame,
    team_seasons: pd.DataFrame,
    year_to_idx: dict[int, int],
    test_season_idx: int,
) -> pd.DataFrame:
    """Compute playoff appearance, missed-playoff drought, and playoff overachievement."""
    completed_coaches = coaches[coaches["season_idx"] < test_season_idx]
    coach_post = (
        completed_coaches.groupby(["tmID", "season_idx"], as_index=False)
        .agg(
            postg_sum=("postg", lambda s: s.sum(min_count=1)),
            postw_sum=("postw", lambda s: s.sum(min_count=1)) if "postw" in completed_coaches.columns else ("postg", lambda s: np.nan),
        )
    )

    completed = (
        team_seasons[team_seasons["season_idx"] < test_season_idx][["tmID", "season_idx", "G", "Pts_Pct_Z"]]
        .merge(coach_post, on=["tmID", "season_idx"], how="left")
        .sort_values(["tmID", "season_idx"])
        .reset_index(drop=True)
    )

    # Prefer TeamsPost table when available for exact postseason G and W
    if not teams_post.empty and {"tmID", "year"}.issubset(teams_post.columns):
        tp = _filter_nhl(teams_post)
        tp["season_idx"] = tp["year"].map(year_to_idx)
        tp = tp.dropna(subset=["tmID", "season_idx"]).copy()
        tp["season_idx"] = tp["season_idx"].astype(int)
        tp = tp[tp["season_idx"] < test_season_idx]
        if not tp.empty:
            tp["tp_g"] = _numeric_ci(tp, ("G",))
            tp["tp_w"] = _numeric_ci(tp, ("W",))
            tp_agg = tp.groupby(["tmID", "season_idx"], as_index=False)[["tp_g", "tp_w"]].sum(min_count=1)
            completed = completed.merge(tp_agg, on=["tmID", "season_idx"], how="left")
            completed["postg_eff"] = completed["tp_g"].fillna(completed["postg_sum"])
            completed["postw_eff"] = completed["tp_w"].fillna(completed["postw_sum"])
        else:
            completed["postg_eff"] = completed["postg_sum"]
            completed["postw_eff"] = completed["postw_sum"]
    else:
        completed["postg_eff"] = completed["postg_sum"]
        completed["postw_eff"] = completed["postw_sum"]

    season_has_playoffs = completed.groupby("season_idx")["postg_eff"].transform(
        lambda s: (s.fillna(0) > 0).any()
    )
    valid = season_has_playoffs & (completed["G"] > 0)
    completed["Made_Playoffs"] = np.where(
        valid,
        (completed["postg_eff"].fillna(0) > 0).astype(float),
        np.nan,
    )

    # Playoff overachievement: normalized playoff wins minus regular-season Pts_Pct_Z (0 if missed)
    po_teams = completed["Made_Playoffs"] == 1.0
    completed["postw_z"] = np.nan
    if po_teams.any():
        po_sub = completed.loc[po_teams, ["season_idx", "postw_eff"]].copy()
        completed.loc[po_teams, "postw_z"] = _season_zscore(po_sub, "postw_eff")

    completed["Playoff_Overachieve"] = np.where(
        completed["Made_Playoffs"].isna(),
        np.nan,
        np.where(
            po_teams,
            completed["postw_z"].fillna(0.0) - completed["Pts_Pct_Z"].fillna(0.0),
            0.0,
        ),
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
    shifted = completed[
        ["tmID", "season_idx", "Made_Playoffs", "Playoff_Drought", "Playoff_Overachieve"]
    ].copy()
    shifted["season_idx"] = shifted["season_idx"] + 1
    return shifted.rename(
        columns={
            "Made_Playoffs": "Made_Playoffs_P1",
            "Playoff_Drought": "Playoff_Drought_P1",
            "Playoff_Overachieve": "Playoff_Overachieve_P1",
        }
    )


def _compute_split_features(
    teams_half: pd.DataFrame,
    team_splits: pd.DataFrame,
    year_to_idx: dict[int, int],
    test_season_idx: int,
) -> pd.DataFrame:
    """Compute Late_Season_Collapse_P1 and Road_Perf_Z_P1 from TeamsHalf & TeamSplits."""
    cols = ["tmID", "season_idx", "Late_Season_Collapse_P1", "Road_Perf_Z_P1"]
    half_collapse = pd.DataFrame(columns=["tmID", "season_idx", "half_collapse"])
    split_feats = pd.DataFrame(columns=["tmID", "season_idx", "split_collapse", "Road_Perf_Z"])

    # 1. TeamsHalf (half 2 vs half 1 points% z-score)
    if not teams_half.empty and {"tmID", "year"}.issubset(teams_half.columns):
        th = _filter_nhl(teams_half)
        th["season_idx"] = th["year"].map(year_to_idx)
        th["half_num"] = _numeric_ci(th, ("half",))
        th = th.dropna(subset=["tmID", "season_idx", "half_num"]).copy()
        th["season_idx"] = th["season_idx"].astype(int)
        th = th[(th["season_idx"] < test_season_idx) & (th["half_num"].isin([1, 2]))]
        if not th.empty:
            th = th.drop_duplicates(["tmID", "season_idx", "half_num"])
            g = _numeric_ci(th, ("G",))
            w = _numeric_ci(th, ("W",)).fillna(0.0)
            t = _numeric_ci(th, ("T",)).fillna(0.0)
            otl = _numeric_ci(th, ("OTL", "OL")).fillna(0.0)
            pts = _numeric_ci(th, ("Pts",)).fillna(2.0 * w + t + otl)
            th["pts_pct"] = _safe_divide(pts, 2.0 * g)

            h1 = th[th["half_num"] == 1][["tmID", "season_idx", "pts_pct"]].copy()
            h1["h1_z"] = _season_zscore(h1, "pts_pct")
            h2 = th[th["half_num"] == 2][["tmID", "season_idx", "pts_pct"]].copy()
            h2["h2_z"] = _season_zscore(h2, "pts_pct")

            merged_h = h1[["tmID", "season_idx", "h1_z"]].merge(
                h2[["tmID", "season_idx", "h2_z"]],
                on=["tmID", "season_idx"],
                how="inner",
            )
            merged_h["half_collapse"] = merged_h["h2_z"] - merged_h["h1_z"]
            half_collapse = merged_h[["tmID", "season_idx", "half_collapse"]]

    # 2. TeamSplits (Jan-Apr vs Oct-Dec points% z-score, plus Road points% z-score)
    if not team_splits.empty and {"tmID", "year"}.issubset(team_splits.columns):
        ts = _filter_nhl(team_splits)
        ts["season_idx"] = ts["year"].map(year_to_idx)
        ts = ts.dropna(subset=["tmID", "season_idx"]).copy()
        ts["season_idx"] = ts["season_idx"].astype(int)
        ts = ts[ts["season_idx"] < test_season_idx]
        if not ts.empty:
            ts = ts.drop_duplicates(["tmID", "season_idx"])

            # Road record
            rw = _numeric_ci(ts, ("rW",)).fillna(0.0)
            rl = _numeric_ci(ts, ("rL",)).fillna(0.0)
            rt = _numeric_ci(ts, ("rT",)).fillna(0.0)
            rotl = _numeric_ci(ts, ("rOTL", "rOL")).fillna(0.0)
            rg = rw + rl + rt + rotl
            ts["road_pts_pct"] = _safe_divide(2.0 * rw + rt + rotl, 2.0 * rg)
            ts["Road_Perf_Z"] = _season_zscore(ts, "road_pts_pct")

            # Early (Sep-Dec) vs Late (Jan-Apr) monthly records
            def _month_pts_games(months: tuple[str, ...]) -> tuple[pd.Series, pd.Series]:
                pts_sum = pd.Series(0.0, index=ts.index)
                g_sum = pd.Series(0.0, index=ts.index)
                for m in months:
                    mw = _numeric_ci(ts, (f"{m}W",)).fillna(0.0)
                    ml = _numeric_ci(ts, (f"{m}L",)).fillna(0.0)
                    mt = _numeric_ci(ts, (f"{m}T",)).fillna(0.0)
                    mol = _numeric_ci(ts, (f"{m}OL", f"{m}OTL")).fillna(0.0)
                    pts_sum = pts_sum + (2.0 * mw + mt + mol)
                    g_sum = g_sum + (mw + ml + mt + mol)
                return pts_sum, g_sum

            early_pts, early_g = _month_pts_games(("Sep", "Oct", "Nov", "Dec"))
            late_pts, late_g = _month_pts_games(("Jan", "Feb", "Mar", "Apr"))

            ts["early_pct"] = _safe_divide(early_pts, 2.0 * early_g)
            ts["late_pct"] = _safe_divide(late_pts, 2.0 * late_g)
            ts["early_z"] = _season_zscore(ts, "early_pct")
            ts["late_z"] = _season_zscore(ts, "late_pct")
            ts["split_collapse"] = ts["late_z"] - ts["early_z"]
            split_feats = ts[["tmID", "season_idx", "split_collapse", "Road_Perf_Z"]]

    if half_collapse.empty and split_feats.empty:
        return pd.DataFrame(columns=cols)

    combined_splits = split_feats.merge(half_collapse, on=["tmID", "season_idx"], how="outer")
    combined_splits["Late_Season_Collapse_P1"] = combined_splits["half_collapse"].fillna(
        combined_splits["split_collapse"]
    )
    combined_splits["Road_Perf_Z_P1"] = combined_splits["Road_Perf_Z"]
    combined_splits["season_idx"] = combined_splits["season_idx"] + 1
    return combined_splits[cols]


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
    agg["season_idx"] = agg["season_idx"] + 1
    return agg[cols]


def build_features(
    combined: pd.DataFrame,
    awards: pd.DataFrame,
    master: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
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
    coach_cols = ["year", "season_idx", "tmID", "coachID", "stint", "postg", "postw"]
    coaches = nhl_combined[[c for c in coach_cols if c in nhl_combined.columns]].copy()
    coaches["stint"] = _numeric(coaches, "stint")
    coaches["postg"] = _numeric(coaches, "postg")
    if "postw" in coaches.columns:
        coaches["postw"] = _numeric(coaches, "postw")
    coaches = coaches.drop_duplicates(["tmID", "season_idx", "coachID", "stint"])

    first, last = _first_last_coaches(coaches, season_col="season_idx")

    playoff_p1 = _compute_playoff_features(
        coaches,
        tables["teams_post"],
        team_seasons,
        year_to_idx,
        test_season_idx,
    )
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

    # ---- Player, goalie & intra-season split features ----
    goalie_p1 = _compute_goalie_perf_p1(tables["goalies"], year_to_idx, test_season_idx)
    base = base.merge(goalie_p1, on=["tmID", "season_idx"], how="left")
    base["Coach_Blame_Goalie_Gap_P1"] = base["Goalie_Perf_Z_P1"] - base["Pts_Pct_Z_P1"]

    skater_delta = _compute_top_skater_core_delta(
        tables["scoring"],
        tables["goalies"],
        year_to_idx,
        test_season_idx,
    )
    base = base.merge(skater_delta, on=["tmID", "season_idx"], how="left")

    split_p1 = _compute_split_features(
        tables["teams_half"],
        tables["team_splits"],
        year_to_idx,
        test_season_idx,
    )
    base = base.merge(split_p1, on=["tmID", "season_idx"], how="left")

    base = base.merge(labels[["year", "tmID", "Coach_Changed"]], on=["year", "tmID"], how="left")
    return base


def _diff1_prior_predictions(test_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fallback risk score when diff=1 (or lockout gap) leaves zero labeled transitions."""
    def _col(name: str, default: float) -> np.ndarray:
        if name not in test_rows.columns:
            return np.full(len(test_rows), default, dtype=float)
        return pd.to_numeric(test_rows[name], errors="coerce").fillna(default).to_numpy(dtype=float)

    pts = _col("Pts_Pct_Z_P1", 0.0)
    missed_po = 1.0 - _col("Made_Playoffs_P1", 0.5)
    midseason = _col("Midseason_Change_P1", 0.0)
    new_coach = _col("New_Start_Coach", 0.0)
    collapse = _col("Late_Season_Collapse_P1", 0.0)
    po_over = _col("Playoff_Overachieve_P1", 0.0)

    raw_score = (
        -0.85 * pts
        + 0.45 * missed_po
        + 0.55 * midseason
        - 0.80 * new_coach
        - 0.30 * collapse
        - 0.30 * po_over
    )
    std = float(np.std(raw_score))
    z = (raw_score - float(np.mean(raw_score))) / std if std > 1e-12 else np.zeros_like(raw_score)
    logit = np.log(CHANGE_FRACTION / (1.0 - CHANGE_FRACTION)) + 0.6 * z
    prob = 1.0 / (1.0 + np.exp(-logit))

    n_flags = min(len(test_rows), max(MIN_FLAGS, int(round(CHANGE_FRACTION * len(test_rows)))))
    cutoff = float(np.partition(prob, -n_flags)[-n_flags])
    pred = (prob >= cutoff).astype(int)
    return prob, pred


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

    tables = _extract_tables(args, kwargs)

    test_year = int(test_data["year"].max())
    combined = pd.concat([train_data, test_data], ignore_index=True)
    awards = train_awards if train_awards is not None else pd.DataFrame()
    master = pd.concat(
        [frame for frame in (train_master, test_master) if frame is not None and not frame.empty],
        ignore_index=True,
    )

    labels = build_history_labels(train_data, train_data)
    data = build_features(combined, awards, master, tables, labels, test_year)

    test_rows = data[data["year"] == test_year].copy()
    if test_rows.empty:
        raise ValueError(f"No usable test rows for test_year={test_year}")

    test_season_idx = int(test_rows["season_idx"].iloc[0])
    train_rows = data[
        (data["season_idx"] < test_season_idx)
        & (data["season_idx"] >= test_season_idx - TRAINING_WINDOW_SEASONS)
    ].dropna(subset=["Coach_Changed"])

    if train_rows.empty or train_rows["Coach_Changed"].nunique() < 2:
        probability, predicted = _diff1_prior_predictions(test_rows)
        result = test_rows[["year", "tmID"]].copy()
        result["change_probability"] = probability
        result["predicted_change"] = predicted
        result.attrs["cutoff_info"] = {"strategy": "diff1_prior_fallback"}
        return result[["year", "tmID", "change_probability", "predicted_change"]]

    active_features = [
        f
        for f in FEATURES
        if train_rows[f].dropna().nunique() > 1 and test_rows[f].notna().any()
    ]

    # Scale L2 regularization smoothly with training sample size
    c_val = float(np.clip(len(train_rows) / 600.0, 0.12, 0.30))
    model_factory = lambda: _make_model(c_value=c_val)

    strategy = kwargs.get("strategy", "implied_count")
    model = model_factory()
    model.fit(train_rows[active_features].astype(float), train_rows["Coach_Changed"].astype(int))
    probability = model.predict_proba(test_rows[active_features].astype(float))[:, 1]

    predicted, cutoff_info = select_coach_changes(
        probabilities=probability,
        train_rows=train_rows,
        strategy=strategy,
        features=active_features,
        model_factory=model_factory,
        fixed_fraction=CHANGE_FRACTION,
        min_flags=MIN_FLAGS,
    )

    result = test_rows[["year", "tmID"]].copy()
    result["change_probability"] = probability
    result["predicted_change"] = predicted
    result.attrs["cutoff_info"] = cutoff_info
    result.attrs["active_features"] = active_features
    return result[["year", "tmID", "change_probability", "predicted_change"]]