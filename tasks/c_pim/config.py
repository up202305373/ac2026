"""Single place to define what gets tested in c_pim."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATABASE = ROOT / "database" / "hockey.db"
TRAIN_FROM = 1995
LOG_DIR = ROOT / "tasks" / "c_pim" / "logs"
PLOTS_DIR = ROOT / "tasks" / "c_pim" / "plots"

TEST_YEARS = [2000, 2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009]

EXPERIMENTS = [
    {"name": "baseline", "model": "baseline"},
]
