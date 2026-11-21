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

def indent(text, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in str(text).splitlines())


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
    with sqlite3.connect(config.DATABASE) as connection:
        return pd.read_sql_query(
            'SELECT * FROM "Teams" WHERE year = ? AND lgID = \'NHL\'',
            connection,
            params=(year,),
        )

def log_summary(log, header: str, graph_name: str, results: list) -> None:
    summary = pd.DataFrame(results)
    log.info(f"\n ===== Overall: {header} =====")
    log.info("  Mean divisional position error: {:.2f} spots (sd {:.2f})".format(
        summary["div_mean_pos_error"].mean(), summary["div_mean_pos_error"].std()))
    log.info("  Divisional positions correct: {}/{}".format(
        summary["div_correct"].sum(), summary["div_total"].sum()))
    log.info("  Division winners correct: {}/{}".format(
        summary["div_winners_correct"].sum(), summary["div_winners_total"].sum()))
    log.info("  Mean spearman: {:.3f} (sd {:.3f})".format(
        summary["spearman"].mean(), summary["spearman"].std()))
    log.info("  Mean points MAE: {:.2f} (RMSE {:.2f})".format(
        summary["points_mae"].mean(), summary["points_rmse"].mean()))

    comparison = pd.concat([r["comparison"] for r in results], ignore_index=True)
    div = pd.concat([r["div"] for r in results], ignore_index=True)
    generate_graphs(comparison, div, graph_name)

def main() -> None:
    log = setup_logging()
    results_by_model = {exp["name"]: [] for exp in config.EXPERIMENTS}

    for test_year in TEST_YEARS:
        sampler(1995, train_until=test_year - 1, test_year=test_year)
        train_data = pd.read_csv(ROOT / "data" / "train" / "Teams.csv")
        test_data = pd.read_csv(ROOT / "data" / "test" / "Teams.csv")

        test_year = int(test_data["year"].iloc[0])
        actual_test = load_actual_teams(test_year)
        log.info(f"\n ===== Year: {test_year} =====")

        for exp in config.EXPERIMENTS:
            name = exp["name"]
            train_and_predict = MODELS[exp["model"]]
            extra = {k: v for k, v in exp.items() if k not in ("name", "model", "alphas")}
            for alpha in exp.get("alphas", [None]):
                label = f"{name}_alpha_{alpha}" if alpha is not None else name
                header = f"{name} (alpha={alpha})" if alpha is not None else name
                log.info(f"\n  ===== Model: {header} =====")
                kwargs = {**extra, **({"alpha": alpha} if alpha is not None else {})}
                predictions = train_and_predict(train_data, test_data, **kwargs)
                if "selected_alpha" in predictions.attrs:
                    log.info("   Selected alpha (validation): {}".format(
                        predictions.attrs["selected_alpha"]))
                log.info("   Predictions:")
                log.info(indent(predictions.to_string(index=False), 4))
                log.info("\n   Test metrics:")
                result = evaluate_ranking(predictions, actual_test, name, test_year, indent="    ")
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