from models import baseline, roster1, roster2

MODELS = {
    "baseline": baseline.train_and_predict,
    "roster1": roster1.train_and_predict,
    "roster2": roster2.train_and_predict,
}