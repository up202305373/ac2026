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

Games might have different weights in their importance in the season (e.g.: final games of teams who already qualified for playoffs). Does the data represent this somehow or is this overlooked?

## Potential Derived Columns

Win% (Teams, derived from G,W,L,T,OTL,..)