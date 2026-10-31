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
#print(load("SELECT sum(GF) FROM TEAMS"))
#print(load("SELECT sum(GA) FROM TEAMS"))
#print(load("SELECT YEAR, count(*) FROM TEAMS WHERE PLAYOFF IS NOT NULL GROUP BY YEAR"))
#print(load("SELECT YEAR, rank, count(rank) FROM TEAMS GROUP BY YEAR, RANK"))
#print(load("SELECT tmID, COUNT(DISTINCT NAME) AS COUNT FROM TEAMS GROUP BY tmID"))
#print(load("SELECT TMID, YEAR, sum(G), sum(W), sum(L), sum(T), sum(GF), sum(GA), sum(half) FROM TEAMSHALF GROUP BY YEAR, TMID"))
#print(load("SELECT TMID, YEAR, G, W, L, T, GF, GA FROM TEAMS WHERE YEAR > 1915 AND YEAR < 1921 GROUP BY YEAR, TMID"))
#print(load("SELECT COUNT(*) AS total_rows, SUM(CASE WHEN s.hW + s.rW = t.W THEN 1 ELSE 0 END) AS matching_W, SUM(CASE WHEN s.hL + s.rL = t.L THEN 1 ELSE 0 END) AS matching_L, SUM(CASE WHEN s.hOTL + s.rOTL = t.OTL THEN 1 ELSE 0 END) AS matching_OTL, SUM(CASE WHEN s.hT + s.rT = t.T THEN 1 ELSE 0 END) AS matching_T FROM TeamSplits s JOIN Teams t   ON t.year = s.year  AND t.tmID = s.tmID;"))
#print(load(""" SELECT     s.year,     s.tmID,     t.lgID,     s.hW + s.rW AS split_W,     t.W AS teams_W,     s.hL + s.rL AS split_L,     t.L AS teams_L,     s.hT + s.rT AS split_T,     t.T AS teams_T,     s.hOTL + s.rOTL AS split_OTL,     t.OTL AS teams_OTL FROM TeamSplits s JOIN Teams t   ON t.year = s.year  AND t.tmID = s.tmID WHERE        (s.hW IS NOT NULL AND s.rW IS NOT NULL         AND t.W IS NOT NULL         AND s.hW + s.rW <> t.W)     OR (s.hL IS NOT NULL AND s.rL IS NOT NULL         AND t.L IS NOT NULL         AND s.hL + s.rL <> t.L)     OR (s.hT IS NOT NULL AND s.rT IS NOT NULL         AND t.T IS NOT NULL         AND s.hT + s.rT <> t.T)     OR (s.hOTL IS NOT NULL AND s.rOTL IS NOT NULL         AND t.OTL IS NOT NULL         AND s.hOTL + s.rOTL <> t.OTL) ORDER BY s.year, s.tmID; """))
#print(load("SELECT     t.lgID,     MIN(CASE WHEN s.hOTL IS NOT NULL THEN t.year END) AS first_hOTL_year,     MAX(CASE WHEN s.hOTL IS NOT NULL THEN t.year END) AS last_hOTL_year,     MIN(CASE WHEN s.rOTL IS NOT NULL THEN t.year END) AS first_rOTL_year,     MAX(CASE WHEN s.rOTL IS NOT NULL THEN t.year END) AS last_rOTL_year,     MIN(CASE WHEN s.hT IS NOT NULL THEN t.year END) AS first_hT_year,     MAX(CASE WHEN s.hT IS NOT NULL THEN t.year END) AS last_hT_year,     MIN(CASE WHEN s.rT IS NOT NULL THEN t.year END) AS first_rT_year,     MAX(CASE WHEN s.rT IS NOT NULL THEN t.year END) AS last_rT_year FROM TeamSplits s JOIN Teams t   ON t.year = s.year  AND t.tmID = s.tmID GROUP BY t.lgID;"))
#print(load(""" SELECT COUNT(*) AS matching_rows FROM TeamSplits s JOIN Teams t   ON t.year = s.year  AND t.tmID = s.tmID WHERE     COALESCE(s.SepW, 0) + COALESCE(s.OctW, 0)   + COALESCE(s.NovW, 0) + COALESCE(s.DecW, 0)   + COALESCE(s.JanW, 0) + COALESCE(s.FebW, 0)   + COALESCE(s.MarW, 0) + COALESCE(s.AprW, 0) = t.W AND     COALESCE(s.SepL, 0) + COALESCE(s.OctL, 0)   + COALESCE(s.NovL, 0) + COALESCE(s.DecL, 0)   + COALESCE(s.JanL, 0) + COALESCE(s.FebL, 0)   + COALESCE(s.MarL, 0) + COALESCE(s.AprL, 0) = t.L; """))
#print(load("SELECT * FROM SCORING WHERE G + A != PTS AND G IS NOT NULL AND A IS NOT NULL AND PTS IS NOT NULL"))
#print(load("SELECT * FROM MASTER WHERE deathYear == 2007 and deathMon >= 9"))
#print(load("SELECT count() FROM MASTER WHERE LEGENDSID IS NOT NULL"))
#print(load("SELECT t.year, t.tmID, t.G, SUM(tv.W + tv.L + tv.T + tv.OTL) AS vs_games FROM Teams t JOIN TeamVsTeam tv ON tv.year = t.year AND tv.tmID = t.tmID WHERE t.lgID = 'NHL' GROUP BY t.year, t.tmID, t.G HAVING vs_games <> t.G;"))