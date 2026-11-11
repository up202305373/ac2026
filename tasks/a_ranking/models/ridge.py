from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

FEATURES = ["lag_pts", "lag_gf", "lag_ga", "lag_w"]

def build_lagged(teams: pd.DataFrame) -> pd.DataFrame:
    df = teams[teams["lgID"] == "NHL"].copy()
    for col in ["Pts", "GF", "GA", "W", "G"]:
        df[col] = df[col].astype(float)
    df = df.sort_values(["franchID", "year"])
    lagged = df.groupby("franchID")[["Pts", "GF", "GA", "W", "G"]].shift(1)
    df["lag_pts"] = lagged["Pts"]
    df["lag_gf"] = lagged["GF"] / lagged["G"]
    df["lag_ga"] = lagged["GA"] / lagged["G"]
    df["lag_w"] = lagged["W"] / lagged["G"]
    return df

def train_and_predict(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    test_year = int(test_data["year"].iloc[0])
    lagged = build_lagged(train_data)
    train_rows = lagged.dropna(subset=FEATURES + ["Pts"])

    model = Ridge(alpha=1.0).fit(train_rows[FEATURES], train_rows["Pts"].astype(float))

    last = build_lagged(pd.concat([train_data, test_data.assign(Pts=np.nan)]))
    test_rows = last[(last["year"] == test_year)].dropna(subset=FEATURES)

    return pd.DataFrame({
        "year": test_rows["year"].values,
        "franchID": test_rows["franchID"].values,
        "predicted_pts": model.predict(test_rows[FEATURES]),
    })