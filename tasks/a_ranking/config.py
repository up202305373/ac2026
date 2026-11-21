"""Single place to define what gets tested in a_ranking."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATABASE = ROOT / "database" / "hockey.db"
RESULTS_DIR = ROOT / "tasks" / "a_ranking" / "results"
LOG_DIR = ROOT / "tasks" / "a_ranking" / "logs"

TEST_YEARS = [2000, 2001, 2002, 2003, 2006, 2007, 2008, 2009]
ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
PREVIOUS_YEARS = [1]

EXPERIMENTS = [
    {"name": "baseline",       "model": "baseline"},
    {"name": "roster1",        "model": "roster1",        "validation_alphas": ALPHAS, "previous_years": PREVIOUS_YEARS[0]},
    {"name": "roster2",        "model": "roster2",        "validation_alphas": ALPHAS, "previous_years": PREVIOUS_YEARS[0]},
    {"name": "roster_shots",   "model": "roster_shots",   "validation_alphas": ALPHAS, "previous_years": PREVIOUS_YEARS[0]},
    {"name": "roster_special", "model": "roster_special", "validation_alphas": ALPHAS, "previous_years": PREVIOUS_YEARS[0]},
    {"name": "roster_pts",     "model": "roster_pts",     "validation_alphas": ALPHAS, "previous_years": PREVIOUS_YEARS[0]},
    {"name": "roster_all_pts", "model": "roster_all_pts", "validation_alphas": ALPHAS, "previous_years": PREVIOUS_YEARS[0]},
]
