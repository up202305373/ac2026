from functools import partial
from models import baseline, roster

MODELS = {
    "baseline":       baseline.train_and_predict,
    "roster1":        partial(roster.train_and_predict, features=("base",)),
    "roster2":        partial(roster.train_and_predict, features=("base", "shots", "special")),
    "roster_shots":   partial(roster.train_and_predict, features=("base", "shots")),
    "roster_special": partial(roster.train_and_predict, features=("base", "special")),
    "roster_pts":     partial(roster.train_and_predict, features=("base", "prior_pts")),
    "roster_all_pts": partial(roster.train_and_predict, features=("base", "shots", "special", "prior_pts")),
}
