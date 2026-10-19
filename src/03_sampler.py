import sqlite3
from pathlib import Path
import pandas as pd

pd.set_option('display.max_columns', None)

def load_database():
    with sqlite3.connect("database/hockey.db") as connection:
        table_names = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type = 'table'",connection)["name"].tolist()
        tables = {table_name: pd.read_sql_query(f'SELECT * FROM "{table_name}"',connection) for table_name in table_names}
    return tables

def sampler(train_until=2008, test_from=2009, test_to=2009):
    tables = load_database()
    train = {}
    test = {}

    for table_name, table in tables.items():
        if "year" in table.columns:
            train[table_name] = table[table["year"] <= train_until].copy()
            test[table_name] = table[ (table["year"] >= test_from) & (table["year"] <= test_to)].copy()
        else:
            train[table_name] = table.copy()
            test[table_name] = table.copy()

    # TODO: check what values are information only available after the season starts, remove values from those columns in test set

    for table_name, table in test.items():
        print(f"{table_name} test columns: {list(table.columns)}")
        print(table.head(1))

    return train, test

sampler()