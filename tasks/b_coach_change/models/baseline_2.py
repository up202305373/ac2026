from __future__ import annotations

import numpy as np
import pandas as pd


def train_and_predict(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    *args,
    X_seasons: int = 3,
    **kwargs,
) -> pd.DataFrame:
    if train_data.empty or test_data.empty:
        raise ValueError("train_data and test_data must not be empty")

    required_columns = {"year", "tmID", "coachID", "stint"}
    for name, data in (("train_data", train_data), ("test_data", test_data)):
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    train = train_data.copy()
    test = test_data.copy()
    for df in (train, test):
        if "lgID" in df.columns:
            df.drop(df[df["lgID"] != "NHL"].index, inplace=True)
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
        df["stint"] = pd.to_numeric(df["stint"], errors="coerce")

    # Contiguous NHL season index so 2005 looks back to 2003 across the 2004-05 lockout
    all_years = sorted(set(train["year"]).union(set(test["year"])))
    year_to_idx = {y: i for i, y in enumerate(all_years)}
    train["season_idx"] = train["year"].map(year_to_idx)
    test["season_idx"] = test["year"].map(year_to_idx)

    # Deduplicate coaching stints
    train_stints = (
        train[["year", "season_idx", "tmID", "coachID", "stint"]]
        .drop_duplicates(["season_idx", "tmID", "coachID", "stint"])
        .sort_values(["tmID", "season_idx", "stint"], na_position="last")
        .copy()
    )
    train_stints["stint_order"] = train_stints.groupby(["season_idx", "tmID"]).cumcount() + 1
    train_stints["total_stints"] = train_stints.groupby(["season_idx", "tmID"])["coachID"].transform("count")

    # 1. Team Stability: Did the team have >1 coaching stint in the prior NHL season?
    team_yearly = (
        train_stints[["season_idx", "tmID", "total_stints"]]
        .drop_duplicates(["season_idx", "tmID"])
        .copy()
    )
    team_yearly["team_changed_midseason"] = (team_yearly["total_stints"] > 1).astype(int)

    previous_team_change = team_yearly[["season_idx", "tmID", "team_changed_midseason"]].rename(
        columns={"season_idx": "prev_season_idx", "team_changed_midseason": "team_unstable"}
    )

    # 2. Coach Stability: Was the coach replaced before the team's final stint that season?
    train_stints["fired_midseason"] = (
        train_stints["stint_order"] < train_stints["total_stints"]
    ).astype(int)

    # 3. Starting coach for each team in test_data (handles stint=0, stint=1, or NaN)
    test_stints = (
        test[["year", "season_idx", "tmID", "coachID", "stint"]]
        .drop_duplicates(["season_idx", "tmID", "coachID", "stint"])
        .sort_values(["tmID", "season_idx", "stint"], na_position="last")
    )
    predictions = test_stints.drop_duplicates(["season_idx", "tmID"], keep="first")[
        ["year", "season_idx", "tmID", "coachID"]
    ].copy()
    predictions["prev_season_idx"] = predictions["season_idx"] - 1

    predictions = predictions.merge(
        previous_team_change,
        on=["prev_season_idx", "tmID"],
        how="left",
        validate="many_to_one",
    )
    predictions["team_unstable"] = predictions["team_unstable"].fillna(0).astype(int)

    # 4. Count mid-season firings for Start_Coach in the prior X_seasons NHL seasons
    def count_recent_firings(row: pd.Series) -> int:
        if pd.isna(row["coachID"]):
            return 0
        recent = train_stints[
            (train_stints["coachID"] == row["coachID"])
            & (train_stints["season_idx"] >= row["season_idx"] - X_seasons)
            & (train_stints["season_idx"] < row["season_idx"])
        ]
        return int(recent["fired_midseason"].sum())

    predictions["recent_firings"] = predictions.apply(count_recent_firings, axis=1)
    predictions["predicted_change"] = (
        (predictions["team_unstable"] == 1) | (predictions["recent_firings"] >= 2)
    ).astype(int)

    # Smooth heuristic score in [0, 1] for compatibility with probability-expecting utilities
    raw_score = 0.45 * predictions["team_unstable"] + 0.25 * predictions["recent_firings"].clip(upper=3)
    predictions["change_probability"] = np.clip(0.15 + raw_score, 0.05, 0.95)

    return predictions[["year", "tmID", "change_probability", "predicted_change"]]