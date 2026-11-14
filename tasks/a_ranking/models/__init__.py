from models import baseline, roster

MODELS = {
    "baseline": baseline.train_and_predict,
    "roster": roster.train_and_predict,
}