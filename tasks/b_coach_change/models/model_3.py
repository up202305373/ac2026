import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- Knobs ------------------------------------------------------------------
# "last"  : label = end-of-season coach in Y differs from END-of-season coach in Y+1
#           (matches your current main.py; noisy, because a coach fired mid-Y+1
#           still counts as a "change" for year Y).
# "first" : label = end-of-season coach in Y differs from the coach who STARTED Y+1.
#           Cleaner target. If you switch, change load_actual_coach_changes in
#           main.py the same way so train and evaluation labels agree.
NEXT_COACH = "last"
HALF_LIFE_YEARS = 12      # older seasons count less (1950s hockey != 2000s hockey)
THRESHOLD = 0.5   # or a float such as 0.5

FEATURES = [
    # Tenure / career
    "Tenure", "Career_Years", "Coach_Age",
    # Results, league-adjusted
    "Pts_Pct", "Pts_Rank_Pct", "GD_pg", "GF_rel", "GA_rel", "PP_rel", "PK_rel",
    # Playoffs
    "Made_Playoffs", "Missed_Streak", "Post_Games", "Post_Win_Pct",
    # Trajectory
    "Pts_Pct_Last_Year", "Pts_Pct_Delta", "Pts_Pct_2yr_Avg", "Has_Prior",
    # Coach-specific context
    "Mid_Season_Hire", "N_Coaches_Year", "Coach_Games_Share", "Coach_Win_Pct",
    "Has_Award",
    # Franchise instability
    "Team_Changes_Prev5",
]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _missed_streak(made) -> list:
    out, run = [], 0
    for m in made:
        run = 0 if m else run + 1
        out.append(run)
    return out


def prep_features(df: pd.DataFrame,
                  awards_df: pd.DataFrame = None,
                  master_df: pd.DataFrame = None) -> pd.DataFrame:
    """One row per (tmID, year) describing the coach who finished that season."""
    df = df.copy()
    for c in ["g", "w", "l", "t", "postg", "postw", "postl", "postt", "stint",
              "G", "W", "L", "T", "OTL", "GF", "GA", "PPG", "PPC", "PKG", "PKC"]:
        df[c] = _num(df, c)
    df = df.sort_values(["tmID", "year", "stint"]).reset_index(drop=True)

    # --- Tenure: consecutive seasons with the same team (stints don't double count)
    cty = (df[["coachID", "tmID", "year"]].drop_duplicates()
           .sort_values(["coachID", "tmID", "year"]).reset_index(drop=True))
    gap = cty.groupby(["coachID", "tmID"])["year"].diff()
    new_run = (gap.isna() | (gap > 2))  # >2 tolerates the 2004-05 lockout
    cty["run"] = new_run.groupby([cty["coachID"], cty["tmID"]]).cumsum()
    cty["Tenure"] = cty.groupby(["coachID", "tmID", "run"]).cumcount() + 1

    # --- Career years: distinct seasons coached, ordered by TIME (old code ordered by team)
    cy = (df[["coachID", "year"]].drop_duplicates()
          .sort_values(["coachID", "year"]).reset_index(drop=True))
    cy["Career_Years"] = cy.groupby("coachID").cumcount() + 1

    # --- Team-season aggregates over all coaches that season
    team = df.groupby(["tmID", "year"]).agg(
        N_Coaches_Year=("coachID", "nunique"),
        Post_Games=("postg", "sum"),
        Post_W=("postw", "sum"),
    ).reset_index()
    first = (df.drop_duplicates(["tmID", "year"], keep="first")
             [["tmID", "year", "coachID"]].rename(columns={"coachID": "First_Coach"}))

    last = df.drop_duplicates(["tmID", "year"], keep="last").copy()
    last = (last.merge(team, on=["tmID", "year"], how="left")
                .merge(first, on=["tmID", "year"], how="left")
                .merge(cty[["coachID", "tmID", "year", "Tenure"]], on=["coachID", "tmID", "year"], how="left")
                .merge(cy, on=["coachID", "year"], how="left"))

    # --- Results, league-adjusted
    games = last["G"].fillna(last[["W", "L", "T", "OTL"]].sum(axis=1)).replace(0, np.nan)
    pts = 2 * last["W"].fillna(0) + last["T"].fillna(0) + last["OTL"].fillna(0)
    last["Pts_Pct"] = pts / (2 * games)
    last["Pts_Rank_Pct"] = last.groupby("year")["Pts_Pct"].rank(pct=True)
    last["GD_pg"] = (last["GF"] - last["GA"]) / games
    gf_pg, ga_pg = last["GF"] / games, last["GA"] / games
    last["GF_rel"] = gf_pg - gf_pg.groupby(last["year"]).transform("mean")
    last["GA_rel"] = ga_pg - ga_pg.groupby(last["year"]).transform("mean")
    pp = (last["PPG"] / last["PPC"]).where(last["PPC"] > 0)
    pk = (last["PKG"] / last["PKC"]).where(last["PKC"] > 0)  # goals-against rate: higher = worse
    last["PP_rel"] = pp - pp.groupby(last["year"]).transform("mean")
    last["PK_rel"] = pk - pk.groupby(last["year"]).transform("mean")

    # --- Playoffs (coach postseason games are a more reliable signal than the label text)
    label_made = pd.Series(False, index=last.index)
    if "playoff" in last.columns:
        p = last["playoff"].astype(str).str.upper().str.strip()
        label_made = last["playoff"].notna() & ~p.isin(["N", "NAN", "DNQ", ""])
    last["Made_Playoffs"] = ((last["Post_Games"] > 0) | label_made).astype(int)
    last["Post_Win_Pct"] = np.where(last["Post_Games"] > 0, last["Post_W"] / last["Post_Games"].replace(0, np.nan), 0)

    # --- Coach-specific
    last["Mid_Season_Hire"] = (last["stint"] > 1).astype(int)
    last["Coach_Games_Share"] = (last["g"] / games).clip(0, 1)
    last["Coach_Win_Pct"] = ((last["w"] + 0.5 * last["t"].fillna(0)) / last["g"].replace(0, np.nan)).fillna(last["Pts_Pct"])

    # --- Trajectory + franchise instability (per team, in time order)
    last = last.sort_values(["tmID", "year"]).reset_index(drop=True)
    by = last.groupby("tmID")
    last["Pts_Pct_Last_Year"] = by["Pts_Pct"].shift(1)
    last["Has_Prior"] = last["Pts_Pct_Last_Year"].notna().astype(int)
    last["Pts_Pct_Delta"] = last["Pts_Pct"] - last["Pts_Pct_Last_Year"]
    last["Pts_Pct_2yr_Avg"] = by["Pts_Pct"].transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
    last["Missed_Streak"] = by["Made_Playoffs"].transform(lambda s: pd.Series(_missed_streak(s.tolist()), index=s.index))
    prev_coach = by["coachID"].shift(1)
    last["_changed_in"] = ((last["coachID"] != prev_coach) & prev_coach.notna()).astype(int)
    last["Team_Changes_Prev5"] = last.groupby("tmID")["_changed_in"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).sum())

    # --- Age (left as NaN when unknown; imputed later with the training median)
    if master_df is not None and not master_df.empty:
        ages = master_df[["coachID", "birthYear"]].dropna(subset=["coachID"]).drop_duplicates("coachID")
        last = last.merge(ages, on="coachID", how="left")
        last["Coach_Age"] = last["year"] - pd.to_numeric(last["birthYear"], errors="coerce")
    else:
        last["Coach_Age"] = np.nan

    # --- Coaching award in this or previous 2 seasons
    if awards_df is not None and not awards_df.empty:
        aw = awards_df[["coachID", "year"]].copy()
        shifted = pd.concat([aw.assign(year=aw["year"] + i) for i in range(3)]).drop_duplicates()
        shifted["Has_Award"] = 1
        last = last.merge(shifted, on=["coachID", "year"], how="left")
        last["Has_Award"] = last["Has_Award"].fillna(0)
    else:
        last["Has_Award"] = 0

    return last


def train_and_predict(train_data: pd.DataFrame, test_data: pd.DataFrame,
                      train_splits=None, train_awards=None, test_splits=None, test_awards=None,
                      train_scoring=None, test_scoring=None,
                      train_awards_players=None, test_awards_players=None,
                      train_goalies=None, test_goalies=None,
                      train_master=None, test_master=None,
                      train_teams_half=None, test_teams_half=None) -> pd.DataFrame:
    # Extra datasets are accepted so main.py keeps working. Splits / goalies / half-season /
    # star-player features were dropped: with ~1000 noisy rows they added variance, not signal.
    # Re-add one at a time and keep it only if pooled AUC improves.
    combined = pd.concat([train_data, test_data], ignore_index=True)
    awards = pd.concat([d for d in (train_awards, test_awards) if d is not None], ignore_index=True)
    master = pd.concat([d for d in (train_master, test_master) if d is not None], ignore_index=True)

    full = prep_features(combined, awards, master)
    test_year = int(test_data["year"].max())

    # Label for season Y comes from season Y+1, which may live in the test year's data.
    # That is legitimate: it is already known when we predict after test_year ends.
    nxt = full[["tmID", "year", "coachID", "First_Coach"]].copy()
    nxt["year"] = nxt["year"] - 1
    nxt["Next_Coach"] = nxt["First_Coach"] if NEXT_COACH == "first" else nxt["coachID"]
    nxt = nxt[["tmID", "year", "Next_Coach"]]

    train_df = full[full["year"] < test_year].merge(nxt, on=["tmID", "year"], how="left")
    train_df = train_df.dropna(subset=["Next_Coach"]).copy()
    train_df["Coach_Changed"] = (train_df["coachID"] != train_df["Next_Coach"]).astype(int)
    test_df = full[full["year"] == test_year].copy()

    X_train, y_train = train_df[FEATURES], train_df["Coach_Changed"]
    X_test = test_df[FEATURES]
    w = 0.5 ** ((test_year - 1 - train_df["year"].to_numpy()) / HALF_LIFE_YEARS)

    # Loosen the constraints so the models can actually learn a distinct rank order
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(C=1.0, max_iter=2000))
    
    rf = make_pipeline(SimpleImputer(strategy="median"),
                       RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=4,
                                              max_features="sqrt", random_state=42, n_jobs=-1))
                                              
    lr.fit(X_train, y_train, logisticregression__sample_weight=w)
    rf.fit(X_train, y_train, randomforestclassifier__sample_weight=w)

    prob = 0.5 * lr.predict_proba(X_test)[:, 1] + 0.5 * rf.predict_proba(X_test)[:, 1]
    
    # DYNAMIC RANK-BASED THRESHOLDING
    # Calculate the historical firing rate from the training set (usually ~15-20%)
    base_rate = y_train.mean()
    
    # Find the probability threshold that marks that exact top percentile in the current test year.
    # This guarantees the model predicts a proportional, realistic number of firings.
    dynamic_thr = np.quantile(prob, 1 - base_rate)

    test_df["change_probability"] = prob
    test_df["predicted_change"] = (prob > dynamic_thr).astype(int)
    
    return test_df[["year", "tmID", "change_probability", "predicted_change"]]