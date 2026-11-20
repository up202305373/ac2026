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

from evaluate import evaluate_ranking
from models import MODELS
from src.sampler import sampler
from evaluate import generate_graphs

OUTPUT_DIR = config.LOG_DIR
LOG_FILE = OUTPUT_DIR / "output.log"
TEST_YEARS = config.TEST_YEARS
TRAIN_WINDOW = config.PREVIOUS_YEARS[-1]

def setup_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("a_ranking")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(LOG_FILE, mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    logger.info(f"\n##### Run started {datetime.now():%Y-%m-%d %H:%M:%S} #####")
    return logger

def load_actual_teams(year: int) -> pd.DataFrame:
    with sqlite3.connect(config.DATABASE) as connection:
        return pd.read_sql_query(
            'SELECT * FROM "Teams" WHERE year = ? AND lgID = \'NHL\'',
            connection,
            params=(year,),
        )

def log_summary(log, header: str, graph_name: str, results: list) -> None:
    summary = pd.DataFrame(results)
    log.info(f"\n===== Overall: {header} =====")
    log.info(f"Mean spearman: {summary['spearman'].mean():.3f} (sd {summary['spearman'].std():.3f})")
    log.info(f"Mean league position error: {summary['mean_pos_error'].mean():.2f} spots (sd {summary['mean_pos_error'].std():.2f})")
    log.info(f"Mean divisional position error: {summary['div_mean_pos_error'].mean():.2f} spots (sd {summary['div_mean_pos_error'].std():.2f})")
    log.info(f"League positions correct: {summary['league_correct'].sum()}/{summary['league_total'].sum()}")
    log.info(f"Divisional positions correct: {summary['div_correct'].sum()}/{summary['div_total'].sum()}")

    comparison = pd.concat([r["comparison"] for r in results], ignore_index=True)
    div = pd.concat([r["div"] for r in results], ignore_index=True)
    generate_graphs(comparison, div, graph_name)

def main() -> None:
    log = setup_logging()
    results_by_model = {exp["name"]: [] for exp in config.EXPERIMENTS}

    for test_year in TEST_YEARS:
        sampler(train_from=test_year - TRAIN_WINDOW, train_until=test_year - 1, test_year=test_year)
        train_data = pd.read_csv(ROOT / "data" / "train" / "Teams.csv")
        test_data = pd.read_csv(ROOT / "data" / "test" / "Teams.csv")

        test_year = int(test_data["year"].iloc[0])
        actual_test = load_actual_teams(test_year)

        for exp in config.EXPERIMENTS:
            name = exp["name"]
            train_and_predict = MODELS[exp["model"]]
            for alpha in exp.get("alphas", [None]):
                label = f"{name}_alpha_{alpha}" if alpha is not None else name
                header = f"{name} (alpha={alpha})" if alpha is not None else name
                log.info(f"\n===== Model: {header} =====")
                kwargs = {"alpha": alpha} if alpha is not None else {}
                predictions = train_and_predict(train_data, test_data, **kwargs)
                log.info("Predictions:")
                log.info(predictions.to_string(index=False))
                log.info(f"\nTest metrics ({test_year}):")
                result = evaluate_ranking(predictions, actual_test, name, test_year)
                results_by_model[name].append({"model": label, "year": test_year, **result})

        print(f"\nlog written to {LOG_FILE}")

    for exp in config.EXPERIMENTS:
        name = exp["name"]
        results = results_by_model[name]
        if exp.get("alphas"):
            for alpha in exp["alphas"]:
                label = f"{name}_alpha_{alpha}"
                filtered = [r for r in results if r["model"] == label]
                if filtered:
                    log_summary(log, f"{name} (alpha={alpha})", label, filtered)
        else:
            log_summary(log, name, name, results)

if __name__ == "__main__":
    main()