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
    """Simple domain baseline: flag the bottom 30% of teams by prior-season points %."""
    test_year = int(test_data["year"].max())
    nhl = pd.concat([train_data, test_data], ignore_index=True)

    lg_col = next((c for c in ("lgID", "lgID_y", "lgID_x") if c in nhl.columns), None)
    if lg_col is not None:
        nhl = nhl[nhl[lg_col] == "NHL"]

    nhl = nhl.drop_duplicates(["year", "tmID"]).copy()

    nhl["pts_pct"] = pd.to_numeric(nhl["Pts"], errors="coerce") / (
        2.0 * pd.to_numeric(nhl["G"], errors="coerce").replace(0, np.nan)
    )
    prev = nhl[nhl["year"] == test_year - 1][["tmID", "pts_pct"]].rename(
        columns={"pts_pct": "prev_pts_pct"}
    )

    out = nhl[nhl["year"] == test_year][["year", "tmID"]].merge(prev, on="tmID", how="left")
    # Lower prior-season points% -> higher firing probability
    rank_risk = 1.0 - out["prev_pts_pct"].rank(pct=True).fillna(0.5)
    n_flags = max(3, int(round(0.30 * len(out))))
    cutoff = np.partition(rank_risk.to_numpy(), -n_flags)[-n_flags]

    out["change_probability"] = rank_risk
    out["predicted_change"] = (rank_risk >= cutoff).astype(int)
    return out[["year", "tmID", "change_probability", "predicted_change"]]