from models import baseline_1, baseline_2, model, model_2, model_3

MODELS = {
    # "baseline_1": baseline_1.train_and_predict, # no one changes
    # "baseline_2": baseline_2.train_and_predict, # teams that fired coaches will fire again and coaches that were fired will be fired again
    "model": model.train_and_predict,
    "model_2": model_2.train_and_predict,
    "model_3": model_3.train_and_predict
}