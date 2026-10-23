# Notes

## Findings

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

## Questions

### Question 1

Load Management: Resting key players on the final games of the season when a playoff spot has already been secured.
Tanking: Building a non-competitive roster to lose as many games as possible in order to get better draft picks on the following year. 
Games might have different weights in their importance in the season (e.g.: final games of teams who already qualified for playoffs). Does the data represent this somehow or is this overlooked?

### Question 2

What exactly is known a day before the season starts? This is what we are currently assuming for the target year and following years:

- **abbrev:** We assume that we can keep all the information from here (Type,Code,Fullname).

- **Coaches:** We assume that, since predictions are made a day before the season starts, the coach starting the season (stint 1) for a team in 2007 is known, but their results in the season are unknown (coachID,2007,tmID,lgID,1,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_). If predicting for multiple seasons, in the following season the coach would be unknown (\_\_\_\_\_\_,2008,tmID,lgID,1,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_). Rows with stint > 1 are also entirely removed so the model can't know in advance what teams will change coaches, which is the entire point of task b).

- **Teams:** We assume that information about the teams competing is known, however their results are unknown (2007,lgID,tmID,franchID,confID,divID,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,name,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_).

- **TeamsHalf:** We assume that information about the teams competing is known and that it is known that hockey games have 2 halves (year,lgID,tmID,half,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_).

- **Scoring:** We assume that information about the player is known, however his statistics are unknown (playerID,2007,1,tmID,lgID,pos,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_). Rows with stint > 1 are also entirely removed so the model can't know in advance what players will change teams.

- **Goalies:** We assume that information about the player is known, however his statistics are unknown (playerID,2007,1,tmID,lgID,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_). Rows with stint > 1 are also entirely removed so the model can't know in advance what goalies will change teams.

- **Master:** We assume that some information about all people is known (playerID,coachID,\_\_,firstName,lastName,nameNote,nameGiven,nameNick,height,weight,shootCatch,legendsID,ihdbID,hrefID,firstNHL,\_\_,firstWHA,\_\_,pos,birthYear,birthMon,birthDay,birthCountry,birthState,birthCity,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_). Some information regarding future events should be hidden:
  - The hofID is hidden to not reveal that the player at hand will be in the hall of fame in the future (and the corresponding row in HOF is also removed).
  - The lastNHL and lastWHA are hidden to not reveal the final year of a player's career for players that are yet to retire.
  - All information related to a death that occurs after the day before the start of the season should be hidden.

- **SeriesPost:** We assume that the format of the series is known but not the amount of games it ends up having or the teams participating (year,round,series,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_)

- **ScoringShootout:** Due to the lack of information on players with 0 S,G,GDG, we assume that players only make it to this list if they are involved in a shootout. Since we don't know beforehand what teams will even be involved in shootouts, it's probably best to remove all rows for the target and following years.

- **GoaliesShootout:** Similar reasoning as the previous one, remove all rows for target and following years.

- **ScoringSC, GoaliesSC, TeamsSC** We don't know which teams will make it to the Stanley Cup finals in advance, so we probably want to remove every row from these tables for the target and following years.
- **AwardsCoaches, AwardsPlayers, AwardsMisc:** It's probably best to just remove all rows for the target year and following years instead of assuming that the awards that will be given out are known in advance.
- **HOF:** Just remove all rows for the target year and following years.

<!--
- **AwardsCoaches:** We assume that the awards given out are known, however we don't know to who they'll be given (\_\_\_\_\_\_,Jack Adams,2007,NHL.\_\_).
- **AwardsPlayers:** We assume that the awards given out are known, however we don't know to who they'll be given (\_\_\_\_\_\_,First Team All-Star,2007,NHL,\_\_,RW).
- **AwardsMisc:** These awards don't have a consistent pattern, so it's probably best to just remove all rows.
-->

Still yet to define:

- TeamVsTeam (year,lgID,tmID,oppID,\_\_,\_\_,\_\_,\_\_) (depends on whether teams will face each other at least once per season or not)
- ScoringSup
- TeamSplits
- CombinedShutouts
- TeamsPost

## Answered Questions

Q: Can we get external data?
A: No.

Q: Can we use outside tools for the exploratory data analysis?
A: Yes.

## Potential Derived Columns

Win% (Teams, derived from G,W,L,T,OTL,..)