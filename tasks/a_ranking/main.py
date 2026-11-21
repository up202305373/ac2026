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

from evaluate import evaluate_ranking, generate_graphs
from models import MODELS, roster
from src.sampler import sampler

OUTPUT_DIR = config.LOG_DIR
LOG_FILE = OUTPUT_DIR / "output.log"
TEST_YEARS = config.TEST_YEARS

def indent(text, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in str(text).splitlines())

def sd_text(values, decimals: int = 2) -> str:
    value = pd.Series(values).std()
    return f"{value:.{decimals}f}" if pd.notna(value) else "n/a"

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

def log_summary(log, header: str, results: list) -> None:
    if not results:
        log.info(f"\n ===== Overall: {header} =====")
        log.info("  No usable results")
        return
    summary = pd.DataFrame(results)
    log.info(f"\n ===== Overall: {header} =====")
    log.info("  Years: {}".format(", ".join(str(int(y)) for y in sorted(summary["year"]))))
    log.info("  Mean divisional position error: {:.2f} spots (sd {})".format(
        summary["div_mean_pos_error"].mean(), sd_text(summary["div_mean_pos_error"])))
    log.info("  Divisional positions correct: {}/{} (mean {:.1f}/year, sd {})".format(
        summary["div_correct"].sum(), summary["div_total"].sum(),
        summary["div_correct"].mean(), sd_text(summary["div_correct"], 1)))
    log.info("  Division winners correct: {}/{} (mean {:.1f}/year, sd {})".format(
        summary["div_winners_correct"].sum(), summary["div_winners_total"].sum(),
        summary["div_winners_correct"].mean(), sd_text(summary["div_winners_correct"], 1)))
    log.info("  Mean spearman: {:.3f} (sd {})".format(
        summary["spearman"].mean(), sd_text(summary["spearman"], 3)))
    log.info("  Mean points MAE: {:.2f} (sd {}); RMSE {:.2f} (sd {})".format(
        summary["points_mae"].mean(), sd_text(summary["points_mae"]),
        summary["points_rmse"].mean(), sd_text(summary["points_rmse"])))

    per_year = pd.DataFrame({
        "year": summary["year"].astype(int),
        "div_ok": summary["div_correct"].astype(int),
        "winners": summary["div_winners_correct"].astype(int),
        "div_mae": summary["div_mean_pos_error"].round(2),
        "spearman": summary["spearman"].round(3),
        "pts_mae": summary["points_mae"].round(2),
    }).sort_values("year")
    log.info("\n  Per year:")
    log.info(indent(per_year.to_string(index=False), 4))

def main() -> None:
    log = setup_logging()
    results_by_model = {exp["name"]: [] for exp in config.EXPERIMENTS}
    data_cache = {}

    def get_data(year: int, train_from: int):
        """Sampler output for one (test year, training window) pair."""
        key = (year, train_from)
        if key not in data_cache:
            sampler(train_from, train_until=year - 1, test_year=year)
            data_cache[key] = (
                pd.read_csv(ROOT / "data" / "train" / "Teams.csv"),
                pd.read_csv(ROOT / "data" / "test" / "Teams.csv"),
                roster.load_players("train"),
                roster.load_players("test"),
            )
        return data_cache[key]

    for test_year in TEST_YEARS:
        train_data, test_data, _, _ = get_data(test_year, 1995)

        test_year = int(test_data["year"].iloc[0])
        actual_test = load_actual_teams(test_year)
        log.info(f"\n ===== Year: {test_year} =====")

        for exp in config.EXPERIMENTS:
            name = exp["name"]
            train_and_predict = MODELS[exp["model"]]
            extra = {k: v for k, v in exp.items()
                     if k not in ("name", "model", "alphas", "train_from")}
            for alpha in exp.get("alphas", [None]):
                label = f"{name}_alpha_{alpha}" if alpha is not None else name
                header = f"{name} (alpha={alpha})" if alpha is not None else name
                log.info(f"\n  ===== Model: {header} =====")
                train_data, test_data, train_players, test_players = get_data(
                    test_year, exp.get("train_from", 1995))
                if train_data["year"].nunique() < 2:
                    log.info("   Skipped: no usable training window before this year")
                    continue
                players = ({"train_players": train_players, "test_players": test_players}
                           if exp["model"] != "baseline" else {})
                kwargs = {**extra, **players,
                          **({"alpha": alpha} if alpha is not None else {})}
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
                    log_summary(log, f"{name} (alpha={alpha})", filtered)
        else:
            log_summary(log, name, results)

    generate_graphs(results_by_model)

if __name__ == "__main__":
    main()