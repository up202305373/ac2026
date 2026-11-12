from models import baseline, roster

MODELS = {
    "baseline": baseline.train_and_predict,
    "roster_history": roster.train_and_predict,
}