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

OUTPUT_DIR = ROOT / "tasks" / "a_ranking" / "logs"
LOG_FILE = OUTPUT_DIR / "output.log"
TEST_YEARS = [2000, 2001, 2002, 2003, 2006, 2007, 2008, 2009]
ALPHAS = [1.0, 3.0, 10.0, 30.0, 100.0]

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
    with sqlite3.connect(ROOT / "database" / "hockey.db") as connection:
        return pd.read_sql_query(
            'SELECT * FROM "Teams" WHERE year = ? AND lgID = \'NHL\'',
            connection,
            params=(year,),
        )

def main() -> None:
    log = setup_logging()

    baseline_results = []
    roster1_results = []
    roster2_results = []
    results_by_model = {"baseline": baseline_results, "roster1": roster1_results, "roster2": roster2_results}

    for test_year in TEST_YEARS:
        sampler(train_from=test_year - 5, train_until=test_year - 1, test_year=test_year)
        train_data = pd.read_csv(ROOT / "data" / "train" / "Teams.csv")
        test_data = pd.read_csv(ROOT / "data" / "test" / "Teams.csv")

        test_year = int(test_data["year"].iloc[0])
        actual_test = load_actual_teams(test_year)

        for name, train_and_predict in MODELS.items():
            if name == "roster1" or name == "roster2":
                for alpha in ALPHAS:
                    log.info(f"\n===== Model: {name} (alpha={alpha}) =====")
                    predictions = train_and_predict(train_data, test_data, alpha=alpha)
                    log.info("Predictions:")
                    log.info(predictions.to_string(index=False))
                    log.info(f"\nTest metrics ({test_year}):")
                    result = evaluate_ranking(predictions, actual_test, name, test_year)
                    results_by_model[name].append({"model": f"{name}_alpha_{alpha}", "year": test_year, **result})
            else:
                log.info(f"\n===== Model: {name} =====")
                predictions = train_and_predict(train_data, test_data)
                log.info("Predictions:")
                log.info(predictions.to_string(index=False))
                log.info(f"\nTest metrics ({test_year}):")
                result = evaluate_ranking(predictions, actual_test, name, test_year)
                results_by_model[name].append({"model": name, "year": test_year, **result})
                
        print(f"\nlog written to {LOG_FILE}")

        
    for name, results in results_by_model.items():
        if name == "roster1" or name == "roster2":
            for alpha in ALPHAS:
                filtered_results = [r for r in results if r["model"] == f"{name}_alpha_{alpha}"]
                if filtered_results:
                    summary = pd.DataFrame(filtered_results)
                    log.info(f"\n===== Overall: {name} (alpha={alpha}) =====")
                    log.info(f"Mean spearman: {summary['spearman'].mean():.3f} (sd {summary['spearman'].std():.3f})")
                    log.info(f"Mean league position error: {summary['mean_pos_error'].mean():.2f} spots (sd {summary['mean_pos_error'].std():.2f})")
                    log.info(f"Mean divisional position error: {summary['div_mean_pos_error'].mean():.2f} spots (sd {summary['div_mean_pos_error'].std():.2f})")
                    log.info(f"League positions correct: {summary['league_correct'].sum()}/{summary['league_total'].sum()}")
                    log.info(f"Divisional positions correct: {summary['div_correct'].sum()}/{summary['div_total'].sum()}")

                    comparison = pd.concat([r["comparison"] for r in filtered_results], ignore_index=True)
                    div = pd.concat([r["div"] for r in filtered_results], ignore_index=True)
                    generate_graphs(comparison, div, f"{name}_alpha_{alpha}")

        else:
            summary = pd.DataFrame(results)
            log.info(f"\n===== Overall: {name} =====")
            log.info(f"Mean spearman: {summary['spearman'].mean():.3f} (sd {summary['spearman'].std():.3f})")
            log.info(f"Mean league position error: {summary['mean_pos_error'].mean():.2f} spots (sd {summary['mean_pos_error'].std():.2f})")
            log.info(f"Mean divisional position error: {summary['div_mean_pos_error'].mean():.2f} spots (sd {summary['div_mean_pos_error'].std():.2f})")
            log.info(f"League positions correct: {summary['league_correct'].sum()}/{summary['league_total'].sum()}")
            log.info(f"Divisional positions correct: {summary['div_correct'].sum()}/{summary['div_total'].sum()}")

            comparison = pd.concat([r["comparison"] for r in results], ignore_index=True)
            div = pd.concat([r["div"] for r in results], ignore_index=True)
            generate_graphs(comparison, div, name)


if __name__ == "__main__":
    main()