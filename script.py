# Run from the repository root
import pandas as pd
from tasks.a_ranking.model import train_and_predict

train_data = pd.read_csv("data/train/Teams.csv")
test_data = pd.read_csv("data/test/Teams.csv")

predictions = train_and_predict(train_data, test_data)
print(predictions)