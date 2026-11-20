"""Single place to define what gets tested in a_ranking."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATABASE = ROOT / "database" / "hockey.db"
RESULTS_DIR = ROOT / "tasks" / "a_ranking" / "results"
LOG_DIR = ROOT / "tasks" / "a_ranking" / "logs"

TEST_YEARS = [2000, 2001, 2002, 2003, 2006, 2007, 2008, 2009]
PREVIOUS_YEARS = [1, 2, 3, 4, 5]
ALPHAS = [1.0, 3.0, 10.0, 30.0, 100.0]

EXPERIMENTS = [
    {"name": "baseline", "model": "baseline"},
    {"name": "roster1",  "model": "roster1", "alphas": ALPHAS},
    {"name": "roster2",  "model": "roster2", "alphas": ALPHAS},
]