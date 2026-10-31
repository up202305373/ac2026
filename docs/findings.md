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

Some franchises went defunct so they shouldn't be considered when making predictions for a year where they no longer exist. However some franchises may have renamed. (TODO: look into this later, might be important)

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

Apparently there was an interin coach (John Fergurson (General Manager)) (which would be "stint 2") that isn't listed here. It seems like he didn't actually coach any games, given this season had 80 games and there's 21 from Tom Barry and 59 from Barry Long.

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

The NHL no longer has draws. They were removed starting in the 2005 season.

```PY
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

```PY
print(load("SELECT sum(G) FROM TEAMS"))
print(load("SELECT sum(G) FROM COACHES"))
print(load("SELECT t.year, t.tmID, t.lgID, t.G FROM Teams t WHERE NOT EXISTS ( SELECT 1 FROM Coaches c WHERE c.tmID = t.tmID AND c.year = t.year );"))
print(load("SELECT COUNT(*) AS team_seasons, SUM(CASE WHEN t.G = c.coach_g THEN 1 ELSE 0 END) AS matching FROM Teams t JOIN (SELECT tmID, year, SUM(g) AS coach_g FROM Coaches GROUP BY tmID, year) c ON c.tmID = t.tmID AND c.year = t.year;"))
print(load("SELECT t.year, t.tmID, t.G, c.coach_g FROM Teams t JOIN (SELECT tmID, year, SUM(g) AS coach_g FROM Coaches GROUP BY tmID, year) c   ON c.tmID = t.tmID AND c.year = t.year WHERE t.G <> c.coach_g;"))

   sum(G)
0  100678
     sum(G)
0  100628.0
   year tmID lgID   G
0  1909  REN  NHA  12
1  1910  REN  NHA  16
2  1912  TO1  NHA  20
3  1978  CZE  WHA   6
4  1978  FIN  WHA   1
   team_seasons  matching
0          1454      1453
   year tmID   G  coach_g
0  1998  DET  82     87.0
```

When looking into the reason why game amount in Teams and Coaches didn't match up, we found five listed team-seasons that have no coach row, all outside the NHL. However, if their coaches were documented, the amount of games would still be off by 5. To find the reason, we looked at individual team-seasons.
Coach games match team games in 1453 of 1454 team-seasons. DET 1998 is the one mismatch because two co-coaches have overlapping games (87 coach-games versus 82 team-games), which explained the discrepancy.

```PY
print(load("SELECT * FROM COACHES WHERE G IS NULL"))
print(load("SELECT * FROM COACHES WHERE NOTES IS NOT NULL"))
print(load("SELECT count(*) FROM COACHES WHERE POSTG IS NULL"))

      coachID  year tmID lgID  stint notes     g     w     l     t  postg  postw  postl  postt
0  martija02c  2001  OTT  NHL      3  None  None  None  None  None   12.0    7.0    5.0    0.0
       coachID  year tmID lgID  stint                      notes    g    w    l    t postg postw postl postt
0   agnewga01c  2006  CBS  NHL      2                    interim  5.0  0.0  4.0  1.0  None  None  None  None
1   arboual01c  2007  NYI  NHL      2                    interim  1.0  1.0  0.0  0.0  None  None  None  None
2   bassejo01c  1977  BIR  WHA      2                    interim  3.0  0.0  3.0  0.0  None  None  None  None
3   lewisda01c  1998  DET  NHL      1  co-coach with Barry Smith  5.0  4.0  1.0  0.0  None  None  None  None
4   mccarja01c  1973  MFS  WHA      2                    interim  2.0  2.0  0.0  0.0  None  None  None  None
5   mccarja01c  1974  MFS  WHA      2                    interim  1.0  0.0  1.0  0.0  None  None  None  None
6   neilsro01c  2001  OTT  NHL      2                    interim  2.0  1.0  1.0  0.0  None  None  None  None
7   paterri01c  1997  TBL  NHL      2                    interim  6.0  0.0  6.0  0.0  None  None  None  None
8   rafteje01c  1977  CIN  WHA      2                    interim  5.0  2.0  3.0  0.0  None  None  None  None
9   smithba02c  1998  DET  NHL      1   co-coach with Dave Lewis  5.0  4.0  1.0  0.0  None  None  None  None
10  wilkiia01c  1972  NRA  WHA      2                    interim  1.0  1.0  0.0  0.0  None  None  None  None
   count(*)
0       846
```

There is only one coach row without regular season information. After a closer look we realized that this coach coached this team on the first 80 games of the season (stint 1), and a different coach coached the last 2 (stint 2), before he was brought back for the post season.
Only 11 coaches have a note (9 interim, 2 co-coach).
Playoff columns are blank in 846 rows. Most definitely "no playoff games" due to not qualifying, not missing data.

```PY
print(load("SELECT sum(GF) FROM TEAMS"))
print(load("SELECT sum(GA) FROM TEAMS"))
print(load("SELECT year, SUM(GF) AS total_GF, SUM(GA) AS total_GA FROM Teams GROUP BY year HAVING SUM(GF) <> SUM(GA) ORDER BY year;"))

   sum(GF)
0   318708
   sum(GA)
0   318708
Empty DataFrame
Columns: [year, total_GF, total_GA]
Index: []
```

GF equals GA for all seasons, so goals are conserved.

```PY
print(load("SELECT YEAR, count(*) FROM TEAMS WHERE PLAYOFF IS NOT NULL GROUP BY YEAR"))
print(load("SELECT YEAR, rank, count(rank) FROM TEAMS GROUP BY YEAR, RANK"))

    year  count(*)
0   1912         2
1   1913         3
2   1914         3
3   1915         2
4   1916         3
..   ...       ...
92  2005        16
93  2006        16
94  2007        16
95  2008        16
96  2009        16

[97 rows x 2 columns]
     year  rank  count(rank)
0    1909     1            1
1    1909     2            1
2    1909     3            1
3    1909     4            1
4    1909     5            1
..    ...   ...          ...
571  2009     1            6
572  2009     2            6
573  2009     3            6
574  2009     4            6
575  2009     5            6

[576 rows x 3 columns]
```

Nowadays 16 teams make the playoffs and there are 6 divisions with ranks 1-5.

```PY
print(load("SELECT tmID, COUNT(DISTINCT NAME) AS COUNT FROM TEAMS GROUP BY tmID"))

    tmID  COUNT
0    ALB      1
1    ANA      1
2    AND      1
3    ATF      1
4    ATL      1
..   ...    ...
109  VML      1
110  VMR      1
111  WAS      1
112  WIJ      1
113  WIN      1

[114 rows x 2 columns]
```

Name isn't stable within a tmID: CHI has 2 (changes from "Black Hawks" to "Blackhawks" in 1986), and TOO changes in 1914.

```PY
print(load("SELECT TMID, YEAR, sum(G), sum(W), sum(L), sum(T), sum(GF), sum(GA), sum(half) FROM TEAMSHALF GROUP BY YEAR, TMID"))
print(load("SELECT TMID, YEAR, G, W, L, T, GF, GA FROM TEAMS WHERE YEAR > 1915 AND YEAR < 1921 GROUP BY YEAR, TMID"))

   tmID  year  sum(G)  sum(W)  sum(L)  sum(T)  sum(GF)  sum(GA)  sum(half)
0   MOC  1916      20      10      10       0       89       80          3
1   MOW  1916      20       5      15       0       94      137          3
2   OT1  1916      20      15       5       0      119       63          3
3   QU1  1916      20      10      10       0       97      126          3
4   TBS  1916      14       7       7       0       64       61          3
5   TOB  1916      14       7       7       0       73       69          3
6   MTL  1917      22      13       9       0      115       84          3
7   MTW  1917       6       1       5       0       17       35          1
8   OTS  1917      22       9      13       0      102      114          3
9   TOA  1917      22      13       9       0      108      109          3
10  MTL  1918      18      10       8       0       88       78          3
11  OTS  1918      18      12       6       0       71       53          3
12  TOA  1918      18       5      13       0       64       92          3
13  MTL  1919      24      13      11       0      129      113          3
14  OTS  1919      24      19       5       0      121       64          3
15  QUB  1919      24       4      20       0       91      177          3
16  TRS  1919      24      12      12       0      119      106          3
17  HAM  1920      24       6      18       0       92      132          3
18  MTL  1920      24      13      11       0      112       99          3
19  OTS  1920      24      14      10       0       97       75          3
20  TRS  1920      24      15       9       0      105      100          3

   tmID  year   G   W   L    T   GF   GA
0   MOC  1916  20  10  10  0.0   89   80
1   MOW  1916  20   5  15  0.0   94  137
2   OT1  1916  20  15   5  0.0  119   63
3   PO1  1916  24   9  15  0.0  114  112
4   QU1  1916  20  10  10  0.0   97  126
5   SEA  1916  24  16   8  0.0  125   80
6   SPO  1916  23   8  15  0.0   89  143
7   TBS  1916  14   7   7  0.0   64   61
8   TOB  1916  14   7   7  0.0   73   69
9   VML  1916  23  14   9  0.0  131  124
10  MTL  1917  22  13   9  0.0  115   84
11  MTW  1917   6   1   5  0.0   17   35
12  OTS  1917  22   9  13  0.0  102  114
13  PO1  1917  18   7  11  0.0   63   75
14  SEA  1917  18  11   7  0.0   67   65
15  TOA  1917  22  13   9  0.0  108  109
16  VML  1917  18   9   9  0.0   70   60
17  MTL  1918  18  10   8  0.0   88   78
18  OTS  1918  18  12   6  0.0   71   53
19  SEA  1918  20  11   9  0.0   66   46
20  TOA  1918  18   5  13  0.0   64   92
21  VA2  1918  20   7  13  0.0   44   81
22  VML  1918  20  12   8  0.0   72   55
23  MTL  1919  24  13  11  0.0  129  113
24  OTS  1919  24  19   5  0.0  121   64
25  QUB  1919  24   4  20  0.0   91  177
26  SEA  1919  22  12  10  0.0   59   55
27  TRS  1919  24  12  12  0.0  119  106
28  VA2  1919  22  10  12  0.0   57   71
29  VML  1919  22  11  11  0.0   75   65
30  HAM  1920  24   6  18  0.0   92  132
31  MTL  1920  24  13  11  0.0  112   99
32  OTS  1920  24  14  10  0.0   97   75
33  SEA  1920  24  12  11  1.0   77   68
34  TRS  1920  24  15   9  0.0  105  100
35  VA2  1920  24  10  13  1.0   72   88
36  VML  1920  24  13  11  0.0   86   79
```

Summing the two halves in TeamsHalf reproduces the statistics shown in Teams, in all team-seasons.

```PY
print(load("SELECT COUNT(*) AS total_rows, SUM(CASE WHEN s.hW + s.rW = t.W THEN 1 ELSE 0 END) AS matching_W, SUM(CASE WHEN s.hL + s.rL = t.L THEN 1 ELSE 0 END) AS matching_L, SUM(CASE WHEN s.hOTL + s.rOTL = t.OTL THEN 1 ELSE 0 END) AS matching_OTL, SUM(CASE WHEN s.hT + s.rT = t.T THEN 1 ELSE 0 END) AS matching_T FROM TeamSplits s JOIN Teams t   ON t.year = s.year  AND t.tmID = s.tmID;"))
print(load("SELECT s.year, s.tmID, t.lgID, s.hW + s.rW AS split_W, t.W AS teams_W, s.hL + s.rL AS split_L,     t.L AS teams_L, s.hT + s.rT AS split_T, t.T AS teams_T, s.hOTL + s.rOTL AS split_OTL, t.OTL AS teams_OTL FROM TeamSplits s JOIN Teams t ON t.year = s.year AND t.tmID = s.tmID WHERE (s.hW IS NOT NULL AND s.rW IS NOT NULL AND t.W IS NOT NULL AND s.hW + s.rW <> t.W) OR (s.hL IS NOT NULL AND s.rL IS NOT NULL AND t.L IS NOT NULL AND s.hL + s.rL <> t.L) OR (s.hT IS NOT NULL AND s.rT IS NOT NULL AND t.T IS NOT NULL AND s.hT + s.rT <> t.T) OR (s.hOTL IS NOT NULL AND s.rOTL IS NOT NULL AND t.OTL IS NOT NULL AND s.hOTL + s.rOTL <> t.OTL) ORDER BY s.year, s.tmID;"))

   total_rows  matching_W  matching_L  matching_OTL  matching_T
0        1459        1459        1459           298        1307
   year tmID  lgID  split_W  teams_W  split_L  teams_L  split_T  teams_T split_OTL teams_OTL
0  1921  EDE  WCHL       15       15        9        9      1.0      0.0      None      None
1  1921  REG  WCHL       14       14       10       10      1.0      0.0      None      None
```

There are two mismatches in results on TeamSplits when compared to Teams, two WCHL rows in 1921.

```PY
print(load("SELECT t.lgID, MIN(CASE WHEN s.hOTL IS NOT NULL THEN t.year END) AS first_hOTL_year, MAX(CASE WHEN s.hOTL IS NOT NULL THEN t.year END) AS last_hOTL_year, MIN(CASE WHEN s.rOTL IS NOT NULL THEN t.year END) AS first_rOTL_year, MAX(CASE WHEN s.rOTL IS NOT NULL THEN t.year END) AS last_rOTL_year, MIN(CASE WHEN s.hT IS NOT NULL THEN t.year END) AS first_hT_year, MAX(CASE WHEN s.hT IS NOT NULL THEN t.year END) AS last_hT_year, MIN(CASE WHEN s.rT IS NOT NULL THEN t.year END) AS first_rT_year, MAX(CASE WHEN s.rT IS NOT NULL THEN t.year END) AS last_rT_year FROM TeamSplits s JOIN Teams t   ON t.year = s.year  AND t.tmID = s.tmID GROUP BY t.lgID;"))

   lgID  first_hOTL_year  last_hOTL_year  first_rOTL_year  ...  first_hT_year  last_hT_year  first_rT_year  last_rT_year
0   NHA              NaN             NaN              NaN  ...           1909          1916           1909          1916
1   NHL           1999.0          2009.0           1999.0  ...           1917          2003           1917          2003
2  PCHA              NaN             NaN              NaN  ...           1911          1923           1911          1923
3  WCHL              NaN             NaN              NaN  ...           1921          1925           1921          1925
4   WHA              NaN             NaN              NaN  ...           1972          1978           1972          1978

[5 rows x 9 columns]
```

The NHL had ties from 1917 to 2003 and overtime losses from 1999 to 2009.

```PY
print(load("SELECT * FROM SCORING WHERE G + A != PTS AND G IS NOT NULL AND A IS NOT NULL AND PTS IS NOT NULL"))

Empty DataFrame
Columns: [playerID, year, stint, tmID, lgID, pos, GP, G, A, Pts, PIM, +/-, PPG, PPA, SHG, SHA, GWG, GTG, SOG, PostGP,PostG, PostA, PostPts, PostPIM, Post+/-, PostPPG, PostPPA, PostSHG, PostSHA, PostGWG, PostSOG]
Index: []
```

Points is Goals + Assists for every line in the Scoring table.

```PY
print(load("SELECT * FROM MASTER WHERE deathYear == 2007 and deathMon >= 9"))

    playerID     coachID       hofID firstName  lastName  ... deathMon deathDay deathCountry  deathState    deathCity
0  davislo01         NaN         NaN     Lorne     Davis  ...     12.0     21.0          NaN         NaN          NaN
1  hannipa01         NaN         NaN       Pat  Hannigan  ...     12.0     11.0          NaN         NaN          NaN
2  harvebu01         NaN         NaN    Buster    Harvey  ...     11.0     25.0       Canada          NB  Fredericton
3  johnsto01  johnsto01c  johnsto01h       Tom   Johnson  ...     11.0     21.0          USA          MA     Falmouth
4  mcnabma01         NaN         NaN       Max     McNab  ...      9.0      2.0          USA          NV    Las Vegas
5   metzdo01         NaN         NaN       Don      Metz  ...     11.0     16.0       Canada          SK       Regina
6  simonth01         NaN         NaN     Thain     Simon  ...      9.0     18.0       Canada          ON     Pembroke
7  warwibi01         NaN         NaN      Bill   Warwick  ...     10.0      3.0       Canada          AB     Edmonton
8  whartle01         NaN         NaN       Len   Wharton  ...      9.0     30.0          NaN         NaN          NaN
9        NaN         NaN  wirtzwi01h   William     Wirtz  ...      9.0     26.0          USA          IL     Evanston

[10 rows x 31 columns]
```

Given the season starts on late September or early October, and we don't have the exact starting day for every season in this dataset, we will assume that for the day before the 2007/2008 season, all deaths until September 2007 are known but October 2007 deaths are unknown.

```PY
print(load("SELECT count() FROM MASTER WHERE LEGENDSID IS NOT NULL"))
print(load("SELECT count() FROM MASTER WHERE HOFID IS NOT NULL"))

   count()
0     6576
   count()
0      361
```

The legends website url listed on Moodle has the string "hhof", which is most definitely an initialism for "Hockey Hall Of Fame". Since we're hiding the hofID on the test data, we made a quick sanity check to see if these 2 had the same count, which would likely mean only players in the hall of fame are on the site and therefore the information on who is on the hall of fame would still be on the table. That seems to not be the case though so the initial decision on the visibility of legendsID stands.

```PY
print(load("SELECT t.year, t.tmID, t.G, SUM(tv.W + tv.L + tv.T + tv.OTL) AS vs_games FROM Teams t JOIN TeamVsTeam tv ON tv.year = t.year AND tv.tmID = t.tmID WHERE t.lgID = 'NHL' GROUP BY t.year, t.tmID, t.G HAVING vs_games <> t.G;"))

Empty DataFrame
Columns: [year, tmID, G, vs_games]
Index: []
```

For all NHL team-seasons, the head-to-head result totals for a team-season in TeamVsTeam is equal to the corresponding Games in the Teams table, meaning TeamVsTeam tracks regular season head-to-head results. Since this is the case and schedules are known before a season starts, keeping the first 4 columns in the test table doesn’t reveal outcomes.