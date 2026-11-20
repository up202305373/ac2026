import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from evaluate import evaluate_coach_change
from models import MODELS
from src.sampler import sampler

OUTPUT_DIR = ROOT / "tasks" / "b_coach_change" / "logs"
LOG_FILE = OUTPUT_DIR / "output.log"
TEST_YEARS = list(range(2005, 2009))

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
        query = """
        SELECT year, tmID, coachID, stint 
        FROM Coaches 
        WHERE year IN (?, ?)
        """
        df = pd.read_sql_query(query, connection, params=(year, year + 1))
        
        df = df.sort_values(by=['tmID', 'year', 'stint'])
        last_coach = df.drop_duplicates(subset=['tmID', 'year'], keep='last').copy()
        
        last_coach['Next_Year_Coach'] = last_coach.groupby('tmID')['coachID'].shift(-1)
        last_coach['Coach_Changed'] = (last_coach['coachID'] != last_coach['Next_Year_Coach']).astype(int)
        
        return last_coach[last_coach['year'] == year].copy()

def load_csv_safe(path: Path) -> pd.DataFrame:
    """Helper to bypass missing files gracefully if sampler exclusions happen."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()

def main() -> None:
    log = setup_logging()

    for test_year in TEST_YEARS:
        sampler(train_from=test_year - 50, train_until=test_year - 1, test_year=test_year)
        
        # Load Core Data
        train_teams = pd.read_csv(ROOT / "data" / "train" / "Teams.csv")
        train_coaches = pd.read_csv(ROOT / "data" / "train" / "Coaches.csv")
        train_data = pd.merge(train_coaches, train_teams[train_teams['lgID'] == 'NHL'], on=['year', 'tmID'])
        
        test_teams = pd.read_csv(ROOT / "data" / "test" / "Teams.csv")
        test_coaches = pd.read_csv(ROOT / "data" / "test" / "Coaches.csv")
        test_data = pd.merge(test_coaches, test_teams[test_teams['lgID'] == 'NHL'], on=['year', 'tmID'])

        # Load Existing Supplementary Feature Data
        train_splits = pd.read_csv(ROOT / "data" / "train" / "TeamSplits.csv")
        train_awards = pd.read_csv(ROOT / "data" / "train" / "AwardsCoaches.csv")
        test_splits = pd.read_csv(ROOT / "data" / "test" / "TeamSplits.csv")
        test_awards = pd.read_csv(ROOT / "data" / "test" / "AwardsCoaches.csv")

        # Load New Contextual Data for Roster & Expectations
        train_scoring = load_csv_safe(ROOT / "data" / "train" / "Scoring.csv")
        test_scoring = load_csv_safe(ROOT / "data" / "test" / "Scoring.csv")
        
        train_awards_players = load_csv_safe(ROOT / "data" / "train" / "AwardsPlayers.csv")
        test_awards_players = load_csv_safe(ROOT / "data" / "test" / "AwardsPlayers.csv")
        
        train_goalies = load_csv_safe(ROOT / "data" / "train" / "Goalies.csv")
        test_goalies = load_csv_safe(ROOT / "data" / "test" / "Goalies.csv")
        
        train_master = load_csv_safe(ROOT / "data" / "train" / "Master.csv")
        test_master = load_csv_safe(ROOT / "data" / "test" / "Master.csv")
        
        train_teams_half = load_csv_safe(ROOT / "data" / "train" / "TeamsHalf.csv")
        test_teams_half = load_csv_safe(ROOT / "data" / "test" / "TeamsHalf.csv")

        actual_test = load_actual_coach_changes(test_year)

        for name, train_and_predict in MODELS.items():
            log.info(f"\n===== Model: {name} =====")

            # Pass all extended datasets through to the model
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
                test_teams_half=test_teams_half
            )
            
            # log.info("Predictions:")
            # log.info(predictions.to_string(index=False))

            log.info(f"\nTest metrics ({test_year}):")
            evaluate_coach_change(predictions, actual_test)

        print(f"\nlog written to {LOG_FILE}")

if __name__ == "__main__":
    main()