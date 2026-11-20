from functools import partial
from models import baseline, roster

MODELS = {
    "baseline": baseline.train_and_predict,
    "roster1": partial(roster.train_and_predict, features=("base",)),
    "roster2": partial(roster.train_and_predict, features=("base", "shots", "special")),
}