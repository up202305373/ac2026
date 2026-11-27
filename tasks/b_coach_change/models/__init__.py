from models import baseline_1, baseline_2, baseline_3, model, model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10, model_11, model_12

MODELS = {
    "baseline_1": baseline_1.train_and_predict, # no one changes
    # "baseline_2": baseline_2.train_and_predict, # teams that fired coaches will fire again and coaches that were fired will be fired again
    "baseline_3": baseline_3.train_and_predict, # bottom 30% changes
    # "model": model.train_and_predict,
    # "model_2": model_2.train_and_predict,
    # "model_3": model_3.train_and_predict,
    # "model_4": model_4.train_and_predict,
    # "model_5": model_5.train_and_predict,
    # "model_6": model_6.train_and_predict,
    # "model_7": model_7.train_and_predict,
    # "model_8": model_8.train_and_predict,
    # "model_9": model_9.train_and_predict,
    "model_10": model_10.train_and_predict,
    "model_11": model_11.train_and_predict,
    "model_12": model_12.train_and_predict,
}