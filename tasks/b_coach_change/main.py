import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from evaluate import evaluate_coach_change, plot_evaluation_graphs
from models import MODELS
from src.sampler import sampler

OUTPUT_DIR = ROOT / "tasks" / "b_coach_change" / "logs"
PLOTS_DIR = OUTPUT_DIR / "plots"
LOG_FILE = OUTPUT_DIR / "output.log"
RESULTS_CSV = OUTPUT_DIR / "results.csv"
# TEST_YEARS = list(range(1980, 2009, 5))
TEST_YEARS = list(range(1981, 2009, 2))
# TEST_YEARS = [2000, 2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009]
# TEST_YEARS = [2000, 2003, 2007, 2009]


def setup_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("b_coach_change")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(LOG_FILE, mode="a")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    logger.info(f"\n##### Run started {datetime.now():%Y-%m-%d %H:%M:%S} #####")
    return logger


def load_actual_coach_changes(year: int) -> pd.DataFrame:
    with sqlite3.connect(ROOT / "database" / "hockey.db") as connection:
        df = pd.read_sql_query(
            "SELECT year, tmID, coachID, stint FROM Coaches WHERE lgID = 'NHL'",
            connection,
        )

    # Contiguous NHL season index so 2003 pairs with 2005 across the 2004-05 lockout
    nhl_years = sorted(df["year"].dropna().astype(int).unique())
    year_to_idx = {y: i for i, y in enumerate(nhl_years)}
    df["season_idx"] = df["year"].map(year_to_idx)
    df = df.sort_values(["tmID", "season_idx", "stint"])

    summary = (
        df.groupby(["tmID", "year", "season_idx"], as_index=False)
        .agg(
            first_coach=("coachID", "first"),
            last_coach=("coachID", "last"),
            n_stints=("coachID", "count"),
        )
        .sort_values(["tmID", "season_idx"])
    )

    next_season = summary[["tmID", "season_idx", "last_coach", "n_stints"]].copy()
    next_season["season_idx"] = next_season["season_idx"] - 1
    next_season = next_season.rename(
        columns={"last_coach": "Next_Year_Coach", "n_stints": "Next_Year_Stints"}
    )

    merged = summary.merge(next_season, on=["tmID", "season_idx"], how="left")
    target = merged[merged["year"] == year].copy()

    changed = (target["last_coach"] != target["Next_Year_Coach"]) | (
        target["Next_Year_Stints"].fillna(1) > 1
    )
    target["Coach_Changed"] = np.where(
        target["Next_Year_Coach"].isna(), np.nan, changed.astype(float)
    )
    return target.dropna(subset=["Coach_Changed"])

def load_csv_safe(path: Path) -> pd.DataFrame:
    """Helper to bypass missing files gracefully if sampler exclusions happen."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


def main() -> None:
    log = setup_logging()
    results: list[dict] = []

    try:
        for test_year in TEST_YEARS:
            # for diff in [2,3,4,5,7,8,10,12,14,16,18,20,25,30,35,40,45,50]:
            # for diff in [2,3,4,5,7,9,11,13,16,20,30]:
            # for diff in [2,3,5]:
            for diff in [2]:
                sampler(
                    train_from=test_year - diff,
                    train_until=test_year - 1,
                    test_year=test_year,
                )

                # Load Core Data
                train_teams = pd.read_csv(ROOT / "data" / "train" / "Teams.csv")
                train_coaches = pd.read_csv(ROOT / "data" / "train" / "Coaches.csv")
                train_data = pd.merge(
                    train_coaches,
                    train_teams[train_teams["lgID"] == "NHL"],
                    on=["year", "tmID"],
                )

                test_teams = pd.read_csv(ROOT / "data" / "test" / "Teams.csv")
                test_coaches = pd.read_csv(ROOT / "data" / "test" / "Coaches.csv")
                test_data = pd.merge(
                    test_coaches,
                    test_teams[test_teams["lgID"] == "NHL"],
                    on=["year", "tmID"],
                )

                # Load Existing Supplementary Feature Data
                train_splits = pd.read_csv(ROOT / "data" / "train" / "TeamSplits.csv")
                train_awards = pd.read_csv(ROOT / "data" / "train" / "AwardsCoaches.csv")
                test_splits = pd.read_csv(ROOT / "data" / "test" / "TeamSplits.csv")
                test_awards = pd.read_csv(ROOT / "data" / "test" / "AwardsCoaches.csv")

                # Load New Contextual Data for Roster & Expectations
                train_scoring = load_csv_safe(ROOT / "data" / "train" / "Scoring.csv")
                test_scoring = load_csv_safe(ROOT / "data" / "test" / "Scoring.csv")

                train_awards_players = load_csv_safe(
                    ROOT / "data" / "train" / "AwardsPlayers.csv"
                )
                test_awards_players = load_csv_safe(
                    ROOT / "data" / "test" / "AwardsPlayers.csv"
                )

                train_goalies = load_csv_safe(ROOT / "data" / "train" / "Goalies.csv")
                test_goalies = load_csv_safe(ROOT / "data" / "test" / "Goalies.csv")

                train_master = load_csv_safe(ROOT / "data" / "train" / "Master.csv")
                test_master = load_csv_safe(ROOT / "data" / "test" / "Master.csv")

                train_teams_half = load_csv_safe(ROOT / "data" / "train" / "TeamsHalf.csv")
                test_teams_half = load_csv_safe(ROOT / "data" / "test" / "TeamsHalf.csv")

                actual_test = load_actual_coach_changes(test_year)

                for name, train_and_predict in MODELS.items():
                    log.info(
                        f"\n===== Model: {name} (test_year={test_year}, diff={diff}) ====="
                    )

                    try:
                        predictions = train_and_predict(
                            train_data=train_data,
                            test_data=test_data,
                            train_splits=train_splits,
                            train_awards=train_awards,
                            test_splits=test_splits,
                            test_awards=test_awards,
                            train_scoring=train_scoring,
                            test_scoring=test_scoring,
                            train_awards_players=train_awards_players,
                            test_awards_players=test_awards_players,
                            train_goalies=train_goalies,
                            test_goalies=test_goalies,
                            train_master=train_master,
                            test_master=test_master,
                            train_teams_half=train_teams_half,
                            test_teams_half=test_teams_half,
                        )

                        log.info(f"\nTest metrics ({test_year}):")
                        metrics = evaluate_coach_change(predictions, actual_test)
                    except ValueError as exc:
                        log.warning(
                            f"Skipping {name} (test_year={test_year}, diff={diff}): {exc}"
                        )
                        continue

                    results.append(
                        {
                            "test_year": test_year,
                            "diff": diff,
                            "model": name,
                            **metrics,
                        }
                    )

                print(f"Completed test_year={test_year}, diff={diff}")
    finally:
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(RESULTS_CSV, index=False)
            plot_paths = plot_evaluation_graphs(results_df, PLOTS_DIR)

            print(f"\nLog written to {LOG_FILE}")
            print(f"Results saved to {RESULTS_CSV}")
            for path in plot_paths:
                print(f"Plot saved to {path}")
        else:
            print("No completed results to save or plot.")

if __name__ == "__main__":
    main()