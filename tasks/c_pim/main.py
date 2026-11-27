import config
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from evaluate import evaluate_pim
from models import MODELS
from src.sampler import sampler

LOG_DIR = config.LOG_DIR
LOG_FILE = LOG_DIR / "output.log"
TEST_YEARS = config.TEST_YEARS

def indent(text, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in str(text).splitlines())

def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("c_pim")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(LOG_FILE, mode="a")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    logger.info(f"\n##### Run started {datetime.now():%Y-%m-%d %H:%M:%S} #####")
    return logger

def load_actual_players(year: int) -> pd.DataFrame:
    with sqlite3.connect(config.DATABASE) as connection:
        return pd.read_sql_query(
            'SELECT * FROM "Scoring" WHERE year = ? AND lgID = \'NHL\' AND stint = 1',
            connection,
            params=(year,),
        )

def log_summary(log, header: str, results: list) -> None:
    summary = pd.DataFrame(results)
    log.info(f"\n ===== Overall: {header} =====")
    log.info("  Mean absolute error: {:.2f} PIM (sd {:.2f})".format(
        summary["mae"].mean(), summary["mae"].std()))
    log.info("  Mean RMSE: {:.2f} PIM (sd {:.2f})".format(
        summary["rmse"].mean(), summary["rmse"].std()))
    log.info("  Mean spearman: {:.3f} (sd {:.3f})".format(
        summary["spearman"].mean(), summary["spearman"].std()))
    log.info("  Players evaluated: {}/{}".format(
        summary["evaluated"].sum(), summary["total"].sum()))

def main() -> None:
    log = setup_logging()
    results_by_model = {exp["name"]: [] for exp in config.EXPERIMENTS}

    for test_year in TEST_YEARS:
        sampler(config.TRAIN_FROM, train_until=test_year - 1, test_year=test_year)
        train_data = pd.read_csv(ROOT / "data" / "train" / "Scoring.csv")
        test_data = pd.read_csv(ROOT / "data" / "test" / "Scoring.csv")

        test_year = int(test_data["year"].iloc[0])
        actual_test = load_actual_players(test_year)
        log.info(f"\n ===== Year: {test_year} =====")

        for exp in config.EXPERIMENTS:
            name = exp["name"]
            predict = MODELS[exp["model"]]
            extra = {k: v for k, v in exp.items() if k not in ("name", "model")}
            log.info(f"\n  ===== Model: {name} =====")
            predictions = predict(train_data, test_data, **extra)
            if "selected_alpha" in predictions.attrs:
                log.info("   Selected alpha (validation): {}".format(
                    predictions.attrs["selected_alpha"]))
            log.info("   Predictions:")
            log.info(indent(predictions.to_string(index=False), 4))
            log.info("\n   Test metrics:")
            result = evaluate_pim(predictions, actual_test, name, test_year, indent="    ")
            results_by_model[name].append({"model": name, "year": test_year, **result})

    for exp in config.EXPERIMENTS:
        name = exp["name"]
        log_summary(log, name, results_by_model[name])

    print(f"\nlog written to {LOG_FILE}")

if __name__ == "__main__":
    main()
