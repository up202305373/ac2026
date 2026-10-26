# Findings

```SQL
sqlite> SELECT franchID, MIN(year) as min_year, MAX(year) as max_year FROM Teams GROUP BY franchID ORDER BY MAX(year), MIN(year);
MOS|1909|1909
REN|1909|1910
TOO|1912|1914
-- (...)
WPG|1999|2009
CBS|2000|2009
MIN|2000|2009
```

Some franchises went defunct so they shouldn't be considered when making predictions for the 2009/2010 season.
However some franchises may have renamed. Note: look into this later.

```SQL
sqlite> SELECT * FROM Coaches c
WHERE stint > 1
AND NOT EXISTS (
  SELECT 1 FROM Coaches p
  WHERE p.tmID = c.tmID AND p.year = c.year AND p.stint = c.stint - 1
);

882|longba01c|1983|WIN|NHL|3||59.0|25.0|25.0|9.0|3.0|0.0|3.0|0.0
-- This means there's no row for stint 2 for 1983's WIN team.
```

Apparently there was an interin coach (John Fergurson (General Manager)) (which would be "stint 2") that isn't listed here.
It seems like he didn't actually coach any games, given this season had 80 games and there's 21 from Tom Barry and 59 from Barry Long.

```SQL
sqlite> SELECT year, lgID, confID, divID, rank, COUNT(*) FROM Teams WHERE year = 1975 GROUP BY 1,2,3,4,5 ORDER BY 2,3,4,5;
1975|NHL|CC|PT|1|1
1975|NHL|CC|PT|2|1
1975|NHL|CC|PT|3|1
-- (...)
1975|WHA|  |WW|3|1
1975|WHA|  |WW|4|1
1975|WHA|  |WW|5|1
```

Older seasons have multiple leagues, but the most recent one only has the NHL.

```SQL
sqlite> SELECT year, lgID, confID, divID, rank, COUNT(*) FROM Teams WHERE year = 2009 GROUP BY 1,2,3,4,5 ORDER BY 2,3,4,5; 
2009|NHL|EC|AT|1|1
2009|NHL|EC|AT|2|1
2009|NHL|EC|AT|3|1
-- (...)
2009|NHL|WC|PC|3|1
2009|NHL|WC|PC|4|1
2009|NHL|WC|PC|5|1
```

There's an East Conference and West Conference, with 15 teams each. Each conference has 3 divisions, each with 5 teams.

```SQL
sqlite> SELECT MAX(year) FROM teams WHERE T IS NOT NULL;
2003
```

The NHL no longer has draws.

```PY
def table_info():
    tables = load("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")["name"]
    for t in tables:
        print(f"\n{t} table info:")
        print(load(f"PRAGMA table_info({t})"))

table_info()

'''
AwardsCoaches table info:
   cid     name     type  notnull dflt_value  pk
0    0  coachID     TEXT        0       None   0
1    1    award     TEXT        0       None   0
2    2     year  INTEGER        0       None   0
3    3     lgID     TEXT        0       None   0
4    4     note     REAL        0       None   0

##  (...)

abbrev table info:
   cid      name  type  notnull dflt_value  pk
0    0      Type  TEXT        0       None   0
1    1      Code  TEXT        0       None   0
2    2  Fullname  TEXT        0       None   0
'''

```

AwardsCoaches' Note was saved as REAL instead of TEXT because the column is entirely empty.

```PY
def load(query):
    with sqlite3.connect("database/hockey.db") as connection:
        return pd.read_sql(query, connection)

print(load("SELECT year, tmID, COUNT(*) FROM Teams GROUP BY 1,2 HAVING COUNT(*) > 1;"))
print(load("SELECT COUNT(*) FROM Teams WHERE PPG > PPC;"))
print(load("SELECT COUNT(*) FROM Teams WHERE Pts != 2*W + COALESCE(T,0) + COALESCE(OTL,0);"))

'''
Empty DataFrame
Columns: [year, tmID, COUNT(*)]
Index: []

   COUNT(*)
0         0

   COUNT(*)
0         0

'''
```

No year/team pair exists more than once.
No teams have more power play goals than power play chances.
No teams have a different amount of points than the amount calculated with the point formula (2 * Win + 1 * Tie + 1 * Overtime Loss).