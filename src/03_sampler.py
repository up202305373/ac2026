import sqlite3
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

def load_database():
    with sqlite3.connect("database/hockey.db") as connection:
        table_names = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type = 'table'",connection)["name"].tolist()
        tables = {table_name: pd.read_sql_query(f'SELECT * FROM "{table_name}"',connection) for table_name in table_names}
    return tables

def hide(table_name, table, first_test_year):
    hide_columns = {}

    # Not very relevant for tasks a) or b)
    # hide_columns["AwardsCoaches"] = ["coachID", "note"]
    # hide_columns["AwardsPlayers"] = ["playerID", "note"]
    # hide_columns["AwardsMisc"] = "all"
    # hide_columns["HOF"] = "all"
    # SeriesPost 
    # TeamHalf
    #

    hide_columns["Teams"] = ["rank", "playoff", "G", "W", "L", "T", "OTL", "Pts","SoW", "SoL", "GF", "GA", "PIM", "BenchMinor","PPG", "PPC", "SHA", "PKG", "PKC", "SHF"]   

    if table_name == "Coaches":
        # Assumption: Coach the team is starting the first season of the simulation with is known
        # Assumption: Coach the team is starting the next seasons of the simulation with is unknown

        keep = ["coachID", "year", "tmID", "lgID", "stint"]

        table = table.loc[table["stint"] == 1]
        table.loc[table["year"] > first_test_year, "coachID"] = np.nan
        table.loc[:, [col for col in table.columns if col not in keep]] = np.nan
        return table

    elif table_name not in hide_columns:
        return table      

    elif hide_columns[table_name] == "all":
        table = table.iloc[:0]
        return table

    else:
        for column_name in hide_columns[table_name]:
            table[column_name] = table[column_name].iloc[:0]
        return table

def sampler(train_from= 2000, train_until=2006, test_from=2007, test_to=2009):
    tables = load_database()
    train = {}
    test = {}

    for table_name, table in tables.items():
        if "year" in table.columns:
            train[table_name] = table[(table["year"] >= train_from) & (table["year"] <= train_until)].copy()
            test[table_name] = table[(table["year"] >= test_from) & (table["year"] <= test_to)].copy()
        else:
            train[table_name] = table.copy()
            test[table_name] = table.copy()

    # Helper to check what values are information only available after the season starts
    # and remove values from those columns in test set
    for table_name, table in test.items():
        test[table_name] = hide(table_name, test[table_name], test_from)

    # Print helper to check things actually worked
    for table_name, table in test.items():
        print(f"{table_name} test columns: {list(table.columns)}")
        print(table.head(1))
        print(table.tail(1))
        if table_name == "Coaches":
            print(table.head(10))

    return train, test

sampler()