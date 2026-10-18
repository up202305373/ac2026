import sqlite3, pandas as pd
from pathlib import Path

connection = sqlite3.connect("database/hockey.db")

for file in Path("data").glob("*.csv"):
    df = pd.read_csv(file)
    df.to_sql((str(file).split(".")[0]).split("/")[-1],connection)

connection.close()