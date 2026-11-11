import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from evaluate import evaluate_ranking
from models import MODELS
from src.sampler import sampler

OUTPUT_DIR = ROOT / "tasks" / "a_ranking" / "logs"
LOG_FILE = OUTPUT_DIR / "output.log"
TEST_YEARS = [2003, 2007, 2009]

def setup_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("a_ranking")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(LOG_FILE, mode="a")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    logger.info(f"\n##### Run started {datetime.now():%Y-%m-%d %H:%M:%S} #####")
    return logger

def load_actual_teams(year: int) -> pd.DataFrame:
    with sqlite3.connect(ROOT / "database" / "hockey.db") as connection:
        return pd.read_sql_query(
            'SELECT * FROM "Teams" WHERE year = ? AND lgID = \'NHL\'',
            connection,
            params=(year,),
        )

def main() -> None:
    log = setup_logging()

    for test_year in TEST_YEARS:
        sampler(train_from=test_year - 5, train_until=test_year - 1, test_year=test_year)
        train_data = pd.read_csv(ROOT / "data" / "train" / "Teams.csv")
        test_data = pd.read_csv(ROOT / "data" / "test" / "Teams.csv")

        test_year = int(test_data["year"].iloc[0])
        actual_test = load_actual_teams(test_year)

        for name, train_and_predict in MODELS.items():
            log.info(f"\n===== Model: {name} =====")

            predictions = train_and_predict(train_data, test_data)
            log.info("Predictions:")
            log.info(predictions.to_string(index=False))

            log.info(f"\nTest metrics ({test_year}):")
            evaluate_ranking(predictions, actual_test)

        print(f"\nlog written to {LOG_FILE}")

if __name__ == "__main__":
    main()