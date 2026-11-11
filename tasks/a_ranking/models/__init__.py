from models import baseline
from models import ridge

MODELS = {
    "baseline": baseline.train_and_predict,
    "ridge": ridge.train_and_predict,
}