"""Single place to define what gets tested in a_ranking."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATABASE = ROOT / "database" / "hockey.db"
RESULTS_DIR = ROOT / "tasks" / "a_ranking" / "results"
LOG_DIR = ROOT / "tasks" / "a_ranking" / "logs"

TEST_YEARS = [2000, 2001, 2002, 2003, 2006, 2007, 2008, 2009]
ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]

EXPERIMENTS = [
    # Compare different roster feature sets.
    {"name": "baseline",       "model": "baseline"},
    {"name": "roster_base",    "model": "roster_base",    "validation_alphas": ALPHAS, "previous_years": 1},
    {"name": "roster_all",     "model": "roster_all",     "validation_alphas": ALPHAS, "previous_years": 1},
    {"name": "roster_shots",   "model": "roster_shots",   "validation_alphas": ALPHAS, "previous_years": 1},
    {"name": "roster_special", "model": "roster_special", "validation_alphas": ALPHAS, "previous_years": 1},
    {"name": "roster_pts",     "model": "roster_pts",     "validation_alphas": ALPHAS, "previous_years": 1},
    {"name": "roster_all_pts", "model": "roster_all_pts", "validation_alphas": ALPHAS, "previous_years": 1},

    # Aggregating the last 2 / 3 available prior seasons.
    {"name": "roster_base_lookback2", "model": "roster_base", "validation_alphas": ALPHAS, "previous_years": 2},
    {"name": "roster_all_lookback2",  "model": "roster_all",  "validation_alphas": ALPHAS, "previous_years": 2},
    {"name": "roster_base_lookback3", "model": "roster_base", "validation_alphas": ALPHAS, "previous_years": 3},
    {"name": "roster_all_lookback3",  "model": "roster_all",  "validation_alphas": ALPHAS, "previous_years": 3},

    # Individual parameter experiments.
    {"name": "roster_nf3",   "model": "roster_base", "validation_alphas": ALPHAS, "previous_years": 1, "n_forwards": 3},
    {"name": "roster_nf6",   "model": "roster_base", "validation_alphas": ALPHAS, "previous_years": 1, "n_forwards": 6},
    {"name": "roster_nfall", "model": "roster_base", "validation_alphas": ALPHAS, "previous_years": 1, "n_forwards": "all"},
    {"name": "roster_gp40",  "model": "roster_base", "validation_alphas": ALPHAS, "previous_years": 1, "min_skater_gp": 40},
    {"name": "roster_gk20",  "model": "roster_base", "validation_alphas": ALPHAS, "previous_years": 1, "min_goalie_gp": 20},
]
