import os
import sqlite3
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

def load_database():
    with sqlite3.connect("database/hockey.db") as connection:
        table_names = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type = 'table'",connection)["name"].tolist()
        tables = {table_name: pd.read_sql_query(f'SELECT * FROM "{table_name}"',connection) for table_name in table_names}
    return tables

def save_tables(tables, directory):
    os.makedirs(directory, exist_ok=True)

    for table_name, table in tables.items():
        path = os.path.join(directory, f"{table_name}.csv")
        table.to_csv(path, index=False)

def hide(table_name, table, test_year):
    special_cases = { "Master" }
    # TODO: update this list
    remove_everything = {
        "HOF", 
        "AwardsCoaches",
        "AwardsMisc",
        "AwardsPlayers",
    }
    # TODO: update this list
    keep_columns = {
        "abbrev": ["Type", "Code", "Fullname"],
        "Coaches": ["coachID", "year", "tmID", "lgID", "stint"],
        "CombinedShutouts": [],
        "Goalies": [],
        "GoaliesSC": [],
        "GoaliesShootout": [],
        "Master": ["playerID","coachID","hofID","firstName","lastName","nameNote","nameGiven","nameNick","height","weight","shootCatch","legendsID","ihdbID","hrefID","firstNHL","lastNHL","firstWHA","lastWHA","pos","birthYear","birthMon","birthDay","birthCountry","birthState","birthCity","deathYear","deathMon","deathDay","deathCountry","deathState","deathCity"],
        "Scoring": [],
        "ScoringSC": [],
        "ScoringShootout": [],
        "ScoringSup": [],
        "SeriesPost": [],
        "Teams": ["year", "lgID", "tmID", "franchID", "confID", "divID", "name", "G"],
        "TeamsHalf": [],
        "TeamSplits": [],
        "TeamsPost": [],
        "TeamsSC": [],
        "TeamVsTeam": []
    }
    keep_rows = {
        "Coaches": {"column": "stint", "filter": 1},
        "Scoring": {"column": "stint", "filter": 1},
        "Goalies": {"column": "stint", "filter": 1},
    }

    if table_name in remove_everything:
        return table.iloc[:0]
    
    if table_name in special_cases:
        if table_name == "Master":
            death_mask = (table["deathYear"] > test_year) | ((table["deathYear"] == test_year) & (table["deathMon"] >= 10))
            table.loc[death_mask, ["deathYear","deathMon","deathDay","deathCountry","deathState","deathCity"]] = np.nan

            with sqlite3.connect("database/hockey.db") as connection:
                hof_year = pd.read_sql_query('SELECT hofID, year FROM "HOF"',connection,).set_index("hofID")["year"]
            induction_year = table["hofID"].map(hof_year)
            future_hof_mask = induction_year >= test_year
            table.loc[future_hof_mask, "hofID"] = np.nan

            # TODO: last_nhl, last_wha
            # TODO: these exact same filters should be done to the training data as well

    if table_name in keep_rows:
        column = keep_rows[table_name]["column"]  
        filter = keep_rows[table_name]["filter"]
        table = table[(table[column] == filter)]

    for column_name in table.columns:
        if column_name not in keep_columns[table_name]:
            table[column_name] = table[column_name].iloc[:0]

    return table

def sampler(train_from= 2000, train_until=2006, test_year=2007):
    tables = load_database()
    train = {}
    test = {}

    for table_name, table in tables.items():
        if "year" in table.columns:
            train[table_name] = table[(table["year"] >= train_from) & (table["year"] <= train_until)].copy()
            test[table_name] = table[(table["year"] == test_year)].copy()
        else:
            train[table_name] = table.copy()
            test[table_name] = table.copy()

    for table_name, table in test.items():
        test[table_name] = hide(table_name, test[table_name], test_year)

    # Print helper to check things actually worked
    for table_name, table in test.items():
        print(f"{table_name} test columns: {list(table.columns)}")
        print(table.head(1))
        print(table.tail(1))
        if table_name == "Coaches":
            print(table.head(10))

    save_tables(train, "data/train")
    save_tables(test, "data/test")

    return train, test

sampler()