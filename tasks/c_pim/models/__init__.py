from functools import partial
from models import baseline, player

MODELS = {
    "baseline": baseline.train_and_predict,
    "player_base": partial(player.train_and_predict, features=("base",)),
    "player_base_pos": partial(player.train_and_predict, features=("base", "pos")),
}
