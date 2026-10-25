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

#table_info()
#print(load("SELECT sum(G) FROM TEAMS"))
#print(load("SELECT sum(G) FROM COACHES"))
#print(load("SELECT t.year, t.tmID, t.lgID, t.G FROM Teams t WHERE NOT EXISTS ( SELECT 1 FROM Coaches c WHERE c.tmID = t.tmID AND c.year = t.year );"))
#print(load("SELECT COUNT(*) AS team_seasons, SUM(CASE WHEN t.G = c.coach_g THEN 1 ELSE 0 END) AS matching FROM Teams t JOIN (SELECT tmID, year, SUM(g) AS coach_g FROM Coaches GROUP BY tmID, year) c ON c.tmID = t.tmID AND c.year = t.year;"))
#print(load("SELECT t.year, t.tmID, t.G, c.coach_g FROM Teams t JOIN (SELECT tmID, year, SUM(g) AS coach_g FROM Coaches GROUP BY tmID, year) c   ON c.tmID = t.tmID AND c.year = t.year WHERE t.G <> c.coach_g;"))
#print(load("SELECT * FROM COACHES WHERE G IS NULL"))
#print(load("SELECT * FROM COACHES WHERE NOTES IS NOT NULL"))
#print(load("SELECT count(*) FROM COACHES WHERE POSTG IS NULL"))