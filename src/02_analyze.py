import sqlite3
import pandas as pd
from pathlib import Path

def load(query):
    with sqlite3.connect("database/hockey.db") as connection:
        return pd.read_sql(query, connection)

def run(script):
    connection = sqlite3.connect("database/hockey.db")
    try:
        connection.executescript(script)
    finally:
        connection.close()

def table_info():
    tables = load("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")["name"]
    for t in tables:
        print(f"\n{t} table info:")
        print(load(f"PRAGMA table_info({t})"))

table_info()
