from functools import partial
from models import baseline, roster

MODELS = {
    "baseline":       baseline.train_and_predict,
    "roster_base":    partial(roster.train_and_predict, features=("base",)),
    "roster_all":     partial(roster.train_and_predict, features=("base", "shots", "special")),
    "roster_shots":   partial(roster.train_and_predict, features=("base", "shots")),
    "roster_special": partial(roster.train_and_predict, features=("base", "special")),
    "roster_pts":     partial(roster.train_and_predict, features=("base", "prior_pts")),
    "roster_all_pts": partial(roster.train_and_predict, features=("base", "shots", "special", "prior_pts")),
    "roster_base_age": partial(roster.train_and_predict, features=("base", "age")),
    "roster_all_age":  partial(roster.train_and_predict, features=("base", "shots", "special", "age")),
}
