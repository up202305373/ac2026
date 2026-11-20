from __future__ import annotations

import pandas as pd

def train_and_predict(train_data: pd.DataFrame, test_data: pd.DataFrame, X_seasons: int = 3) -> pd.DataFrame:
    if train_data.empty or test_data.empty:
        raise ValueError("train_data and test_data must not be empty")

    required_columns = {"year", "tmID", "coachID", "stint"}
    for name, data in (("train_data", train_data), ("test_data", test_data)):
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    # 1. Define Mid-Season Changes for Teams (Did the team have >1 stint this year?)
    team_yearly = train_data.groupby(["year", "tmID"])["stint"].max().reset_index()
    team_yearly["team_changed_midseason"] = (team_yearly["stint"] > 1).astype(int)
    
    previous_team_change = team_yearly[["year", "tmID", "team_changed_midseason"]].rename(
        columns={"year": "previous_year", "team_changed_midseason": "predicted_change"}
    )

    # 2. Define Mid-Season Firings for Coaches 
    # (A coach was fired if their stint was not the team's final stint that year)
    train_coaches = train_data.merge(
        team_yearly.rename(columns={"stint": "max_stint"}), 
        on=["year", "tmID"]
    )
    train_coaches["fired_midseason"] = (train_coaches["stint"] < train_coaches["max_stint"]).astype(int)

    # 3. Setup Test Predictions based on Starting Coaches (stint == 1)
    predictions = test_data[test_data["stint"] == 1][["year", "tmID", "coachID"]].copy()
    predictions["previous_year"] = predictions["year"] - 1

    # Apply Team Stability: Did the team fire a coach mid-season last year?
    predictions = predictions.merge(
        previous_team_change,
        on=["previous_year", "tmID"],
        how="left",
        validate="many_to_one",
    )
    predictions["predicted_change"] = predictions["predicted_change"].fillna(0).astype(int)

    # 4. Apply Coach Stability Override
    def count_recent_firings(row):
        recent = train_coaches[
            (train_coaches["coachID"] == row["coachID"]) &
            (train_coaches["year"] >= row["year"] - X_seasons) &
            (train_coaches["year"] < row["year"])
        ]
        return recent["fired_midseason"].sum()

    predictions["recent_firings"] = predictions.apply(count_recent_firings, axis=1)

    # If the coach was fired 2 or more times in the last X seasons, predict another firing
    predictions.loc[predictions["recent_firings"] >= 2, "predicted_change"] = 1

    return predictions[["year", "tmID", "predicted_change"]]