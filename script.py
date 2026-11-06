import pandas as pd

from tasks.a_ranking.model import train_and_predict
from tasks.a_ranking.evaluate import evaluate_ranking


def main() -> None:
    train_data = pd.read_csv("data/train/Teams.csv")
    test_data = pd.read_csv("data/test/Teams.csv")

    predictions = train_and_predict(train_data, test_data)
    print("Predictions:")
    print(predictions.to_string(index=False))

    validation_year = train_data["year"].max()
    validation_train = train_data[train_data["year"] < validation_year]
    validation_actual = train_data[train_data["year"] == validation_year]
    validation_predictions = train_and_predict(
        validation_train,
        validation_actual,
    )
    print(f"\nValidation metrics ({validation_year}):")
    evaluate_ranking(validation_predictions, validation_actual)


if __name__ == "__main__":
    main()