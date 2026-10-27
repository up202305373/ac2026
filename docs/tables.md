# Tables

This file has important information regarding all tables.

## abbrev
- **What it is:** lookup of codes and their meanings, used by `Teams` and `SeriesPost`.
- **Key:** one row per (Type, Code).
- **Coverage:** 58 rows. Types: Playoffs (24), Division (17), Round (13), Conference (4).
- **Known before the season:** everything (Type, Code, Fullname).
- **Hidden in test data:** nothing.
- **Quality notes:** `Code` is not unique on its own (`CF`, `F`, `SF` appear under both Playoffs and Round), so join on (Type, Code).
- **Links to:** `Teams.playoff` (Type = Playoffs), `Teams.divID` (Type = Division), `Teams.confID` (Type = Conference) and `SeriesPost.round` (Type = Round). Every code in those columns exists in `abbrev` under the matching Type. Join on (Type, Code), never on Code alone.

## Coaches
- **What it is:** one row per coach per team per season, per stint (a coach's spell with a team that season). Holds regular-season record (`g`, `w`, `l`, `t`) and playoff record (`postg`, `postw`, `postl`, `postt`).
- **Key:** (`coachID`, `year`, `tmID`, `stint`) is unique, with 0 duplicates.
  `stint` is the order of appearance in a team's season. (`tmID`, `year`, `stint`) is not unique, because of co-coaches.
- **Coverage:** 1741 rows, 1909 to 2009. NHL 1522, WHA 111, NHA 43, PCHA 41, WCHL 24. Max 3 stints. By stint: 1455 rows with stint 1, 251 with stint 2, 35 with stint 3.
  Every `coachID` exists in `Master`.
  `year` is the season's starting year, like the other tables.
- **Known before the season (test data keeps):** rows with `stint = 1` only, and the columns `coachID`, `year`, `tmID`, `lgID`, `stint`. The coach who starts the season is assumed known.
- **Hidden in test data:** all rows with `stint > 1` (a mid-season replacement can't be known), and `notes`, `g`, `w`, `l`, `t`, `postg`, `postw`, `postl`, `postt` set to NaN in the kept rows.
- **Quality notes:**
  - Games reconcile with `Teams`: summing `g` over a team's coaches equals `Teams.G` in 1453 of 1454 team-seasons.
  - **DET 1998 is the exception.** Two co-coaches (Dave Lewis and Barry Smith) both have `stint = 1`, with 5 games each (the same 5 games counted twice, giving 87 vs. 82). The `notes` column flags it ("co-coach with ...").
    So a team can have two stint-1 rows, and "one starting coach per team" doesn't always hold.
  - **WIN 1983:** stints 1 and 3 exist and stint 2 is absent. Games still add up (21 + 59 = 80), so no games are missing.
  - **OTT 2001:** the stint-3 coach has blank `g`, `w`, `l`, `t` and only playoff games. One row has a blank regular season.
  - `notes` is almost empty: 1730 of 1741 blank, 9 `interim`, 2 co-coach.
  - Interim spells can be tiny: 1 to 6 games.
  - Playoff columns are blank in 846 rows. Likely "no playoff games", not missing data.
  - Team-seasons with no coach row: only 5, all outside the NHL (REN 1909 and 1910, TO1 1912, and the WHA's CZE and FIN in 1978). Every NHL team-season has a coach.
  - A coach can appear with two teams in one season.
  - Column names are lowercase here (`g`, `w`, `l`, `t`), and uppercase in `Teams`. Case matters in pandas, not in SQLite.
- **Links to:** `Teams` on (`year`, `tmID`) (every (`year`, `tmID`) in `Coaches` exists in `Teams`), `Master` on `coachID` (every `coachID` exists), `AwardsCoaches` on `coachID` (every awarded coach appears in `Coaches`).

## Teams
- **What it is:** one row per team per season: standings, results, goals and penalty/special-teams totals. Holds the targets for (a) (`rank`, `Pts`).
- **Key:** one team in one season. (`year`, `tmID`) is unique, and so is (`year`, `name`).
- **Coverage:** 1459 rows, 1909 to 2009, 100 distinct seasons. 2004 is the only gap (lockout).
  By league: NHL 1265 (1917 to 2009, 92 seasons), WHA 84, NHA 45, PCHA 41, WCHL 24.
  2005 to 2009: 30 teams per season, one row per franchise, 2 conferences, 6 divisions of 5.
  `year` is the season's starting year (2009 = 2009-10).
- **Known before the season (test data keeps):** `year`, `lgID`, `tmID`, `franchID`, `confID`, `divID`, `name`,`G`.
- **Hidden in test data (set to NaN):** `rank`, `playoff`, `W`, `L`, `T`, `OTL`, `Pts`, `SoW`, `SoL`, `GF`, `GA`, `PIM`, `BenchMinor`, `PPG`, `PPC`, `SHA`, `PKG`, `PKC`, `SHF`.
- **Columns:**

| Column | Meaning | Populated (NHL) |
|---|---|---|
| `rank` | Final position **within the division** (1 to 5 in 2008-09), never blank | all | 
| `playoff` | Round in which the team's playoff run ended (or the Cup win), decoded by `abbrev` (Type = Playoffs). Blank = no playoffs | see notes | 
| `G` | Games played | all | 
| `W`, `L` | Wins, regulation losses | all | 
| `T` | Ties | 1917 to 2003 | 
| `OTL` | Overtime losses | 1999 to 2009 | 
| `SoW`, `SoL` | Shootout wins and losses | 2005 to 2009 | 
| `Pts` | Standings points | all | 
| `GF`, `GA` | Goals for and against | all | 
| `PIM` | Penalty minutes | all NHL rows | 
| `BenchMinor` | Bench minor penalties | 1969 to 2009 | 
| `PPG`, `PPC` | Power-play goals, power-play chances | 1963 to 2009 | 
| `PKG`, `PKC` | Penalty-kill goals, Penalty-kill chances | 1963 to 2009 | 
| `SHF`, `SHA` | Shorthanded goals for, against | 1963 to 2009 | 

- **Quality notes:**
  - **Identities hold in every row:** `Pts = 2*W + T + OTL` (blank `T`, `OTL` read as 0), `G = W + L + T + OTL`, `PPG <= PPC` and `PKG <= PKC`.
    This means `L` counts **regulation losses only**, with overtime losses separate.
  - **GF equals GA summed over the league in all 92 NHL seasons**, so goals are conserved.
  - **`T` blank means "no ties exist".** All blanks are NHL 2005 to 2009.
  - **Eras overlap:** 1999 to 2003 have both `T` and `OTL`, and 2005 to 2009 have `OTL` and shootouts but no `T`.
  - **Schedule length by era (NHL):** 82 games in 1995 to 2009, 84 in 1992 to 1993, 80 in 1974 to 1991, 78 in 1970 to 1973, 76 in 1968 to 1969, 74 in 1967, 70 in 1949 to 1966. 1994 is the lockout year with 48. Within 1995 to 2009, every team plays the same number.
  - **`playoff`:** all 24 non-blank codes exist in `abbrev`. In the most recent seasons, exactly 16 teams per year have a code, so blank means missed playoffs there.
  - **`rank` is divisional.** In 2008 and 2009 each division has ranks 1 to 5 once. A league-wide rank doesn't exist. The only duplicates among NHL rows are two teams tied at rank 1 in 1917 (MTL and TOA, 26 points each, no division).
  - **`confID`/`divID`:** blank for the early NHL (`divID` up to 1966, `confID` up to 1973); complete for NHL seasons from 1974 (35 seasons). WHA rows have no `confID`.
  - **Special stats** `PPG`, `SHA`, `PKG`, `SHF`, `PPC` and `PKC` start in 1963 and `BenchMinor` in 1969. One WHA row (IND 1978) has `PPG`, `SHA`, `PKG`, `SHF` but blank `PPC` and `PKC`.
  - **Franchise identity:** 65 `franchID`s, 29 with more than one `tmID`, and no `tmID` belongs to two franchises.
  - **`name` isn't stable within a `tmID`:** CHI changes from "Black Hawks" to "Blackhawks" in 1986, and TOO changes in 1914. Don't group by name.
  - **Always filter on `lgID`.** WHA and other leagues share the table, and WHA seasons overlap NHL ones.
- **Links to:** `Coaches`, `TeamsPost`, `TeamsHalf`, `TeamSplits`, `TeamVsTeam`, `Scoring` and `Goalies` on (`year`, `tmID`). `SeriesPost` via `tmIDWinner`/`tmIDLoser`.

## TeamsHalf
- **What it is:** one row per team per half-season (`year`, `lgID`, `tmID`, `half`, `rank`, `G`, `W`, `L`, `T`, `GF`, `GA`).
- **Key:** (`year`, `tmID`, `half`) is unique.
- **Coverage:** 41 rows, 1916 to 1920 (12, 7, 6, 8, 8 per year). NHL 29, NHA 12. `half` = 1 in 21 rows, 2 in 20 rows. 20 team-seasons have both halves, 1 has one.
- **Known before the season (test data keeps):** `year`, `lgID`, `tmID`, `half`.
- **Hidden in test data:** `rank`, `G`, `W`, `L`, `T`, `GF`, `GA`. For any target year after 1920 the table has no rows.
- **Quality notes:**
  - Summing the two halves reproduces `Teams.G`, `Teams.W`, `Teams.L`, `Teams.T`, `Teams.GF` and `Teams.GA`, in all team-seasons.
- **Links to:** `Teams` on (`year`, `tmID`).

## TeamSplits
- **What it is:** one row per team per season with home and road records (`hW`, `hL`, `hT`, `hOTL`, `rW`, `rL`, `rT`, `rOTL`) and monthly win, loss, tie and overtime-loss counts from September to April.
- **Key:** (`year`, `tmID`) is unique, and the set of (`year`, `tmID`) is identical to `Teams`.
- **Coverage:** 1459 rows, 1909 to 2009. NHL 1265, WHA 84, NHA 45, PCHA 41, WCHL 24.
- **Known before the season (test data keeps):** `year`, `lgID`, `tmID`.
- **Hidden in test data:** all other columns set to NaN.
- **Quality notes:**
  - `hW + rW = Teams.W`, `hL + rL = Teams.L` and `hOTL + rOTL = Teams.OTL` in all 1459 rows. `hT + rT = Teams.T` in all rows except two WCHL rows in 1921 (EDE, REG).
  - `hOTL`, `rOTL` and the monthly `OL` columns are populated from 1999 to 2009. `hT` and `rT` are populated for 1917 to 2003 for the NHL.
- **Links to:** `Teams` on (`year`, `tmID`).

## Scoring
- **What it is:** one row per skater per team per season per stint (regular-season and playoff totals).
- **Key:** (`playerID`, `year`, `stint`) is unique.
- **Coverage:** 43,848 rows, 1909 to 2009. NHL 40,101, WHA 2436, NHA 586, PCHA 404, WCHL 321. By stint: 1 has 40,241 rows, 2 has 3463, 3 has 141, 4 has 3.
- **Known before the season (test data keeps):** rows with `stint = 1`, columns `playerID`, `year`, `stint`, `tmID`, `lgID`, `pos`.
- **Hidden in test data:** rows with `stint > 1`, and all statistics set to NaN in the kept rows.
- **Quality notes:**
  - `Pts = G + A` in every row where both are present.
- **Links to:** `Teams` on (`year`, `tmID`), `Master` on `playerID`, `ScoringSup` on (`playerID`, `year`).

## Goalies
- **What it is:** one row per goalie per team per season per stint (regular-season and playoff totals).
- **Key:** (`playerID`, `year`, `stint`) is unique.
- **Coverage:** 4096 rows, 1909 to 2009. NHL 3678, WHA 259, NHA 75, PCHA 47, WCHL 37. By stint: 1 has 3862 rows, 2 has 229, 3 has 5. 88 rows in 2009.
- **Known before the season (test data keeps):** rows with `stint = 1`, columns `playerID`, `year`, `stint`, `tmID`, `lgID`.
- **Hidden in test data:** rows with `stint > 1`, and all statistics set to NaN in the kept rows.
- **Quality notes:**
- **Links to:** `Teams` on (`year`, `tmID`), `Master` on `playerID`.

## Master
- **What it is:** one row per person (players, coaches and others), with biography and career-span fields.
- **Key:** `playerID`, `coachID` and `hofID` are each unique among their non-blank values.
- **Coverage:** 7469 rows, no `year` column. 6968 rows have only a `playerID`, 124 only a `coachID`, 268 both, 109 neither (all 109 have a `hofID`).
  - Ranges: `firstNHL` and `lastNHL` 1917 to 2009, `firstWHA` and `lastWHA` 1972 to 1978, `birthYear` 1841 to 1991, `height` 63 to 81, `weight` 125 to 265.
  - `pos`: D, C, L, R, G, plus F (43 rows), W (1) and 11 composite codes (21 rows, e.g. `R/C`, `C/D`). `shootCatch`: L 4604, R 2441, B 1.
- **Known before the season (test data keeps):** all columns except those under Hidden.
- **Hidden in test data:** `lastNHL`, `lastWHA`, and death information for deaths after the day before the season starts (`deathYear`, `deathMon`, `deathDay`, `deathCountry`, `deathState`, `deathCity`). `hofID` should be hidden for any person with `HOF[year]` >= `test_year`.
- **Quality notes:**
- **Links to:** `Scoring`, `Goalies`, `ScoringSC`, `GoaliesSC`, `ScoringShootout`, `GoaliesShootout`, `ScoringSup`, `AwardsPlayers`, `CombinedShutouts` on `playerID`. `Coaches`, `AwardsCoaches` on `coachID`. `HOF` on `hofID`.

## SeriesPost
- **What it is:** one row per playoff series (`year`, `round`, `series`, `tmIDWinner`, `lgIDWinner`, `tmIDLoser`, `lgIDLoser`, `W`, `L`, `T`, `GoalsWinner`, `GoalsLoser`, `note`).
- **Key:** (`year`, `round`, `tmIDWinner`) is unique.
- **Coverage:** 802 rows, 1912 to 2009, 97 seasons. By league of the winner: NHL 727, WHA 47, PCHA 13, WCHL 10, NHA 5.
- **Known before the season (test data keeps):** `year`, `round`, `series`.
- **Hidden in test data:** teams, leagues, `W`, `L`, `T`, goals and `note`.
- **Quality notes:**
  - Every year from 2005 to 2009 has exactly 15 series and the rounds are `CQF` (8 per year), `CSF` (4), `CF` (2) and `SCF` (1).
  - Since 2005, the winner has `W = 4` in every row, `L` is 0 to 3 and `T` is 0.
- **Links to:** `Teams` on (`year`, `tmIDWinner`) and (`year`, `tmIDLoser`), `abbrev` on `round` (Type = Round).

## ScoringShootout
- **What it is:** one row per skater per team per season per stint with shootout attempts (`playerID`, `year`, `stint`, `tmID`, `S`, `G`, `GDG`).
- **Key:** (`playerID`, `year`, `stint`) is unique.
- **Coverage:** 1487 rows, 2005 to 2009 (285, 316, 290, 271, 325 per year). Stint 1: 1405 rows, stint 2: 80, stint 3: 2. No blanks.
- **Known before the season:** nothing.
- **Hidden in test data:** remove all rows.
- **Quality notes:**
  - No row has `S = 0`, so players with no attempts do not appear.
- **Links to:** `Teams` on (`year`, `tmID`), `Master` on `playerID`.

## GoaliesShootout
- **What it is:** one row per goalie per team per season per stint with shootout results (`playerID`, `year`, `stint`, `tmID`, `W`, `L`, `SA`, `GA`).
- **Key:** (`playerID`, `year`, `stint`) is unique.
- **Coverage:** 346 rows, 2005 to 2009 (76, 71, 64, 66, 69 per year). Stint 1: 335 rows, stint 2: 11.
- **Known before the season:** nothing.
- **Hidden in test data:** remove all rows.
- **Quality notes:**
- **Links to:** `Teams` on (`year`, `tmID`), `Master` on `playerID`.

## CombinedShutouts
- **What it is:** one row per game in which two goalies combined for a shutout (`year`, `month`, `date`, `tmID`, `oppID`, `R/P`, `IDgoalie1`, `IDgoalie2`).
- **Key:** (`year`, `month`, `date`, `tmID`, `oppID`) is unique.
- **Coverage:** 50 rows, 1929 to 2009. 1 to 5 rows per year, 8 rows since 2005. `R/P`: R 49, P 1 (1955). No blanks.
- **Known before the season:** nothing.
- **Hidden in test data:** remove all rows.
- **Quality notes:**
- **Links to:** `Teams` on `tmID` and `oppID`, `Master` on `IDgoalie1` and `IDgoalie2`.

## TeamsPost
- **What it is:** one row per playoff team per season with playoff totals (`year`, `lgID`, `tmID`, `G`, `W`, `L`, `T`, `GF`, `GA`, `PIM`, `BenchMinor`, `PPG`, `PPC`, `SHA`, `PKG`, `PKC`, `SHF`).
- **Key:** (`year`, `tmID`) is unique.
- **Coverage:** 895 rows, 1913 to 2009. NHL 809, WHA 53, PCHA 14, WCHL 13, NHA 6. Exactly 16 rows per year for every year from 1995 to 2009.
- **Known before the season (test data keeps):** `year`, `lgID`, one row per playoff team (the number of rows is assumed known), with `tmID` hidden.
- **Hidden in test data:** `tmID` and all statistics.
- **Quality notes:**
- **Links to:** `Teams` on (`year`, `tmID`).

## TeamsSC
- **What it is:** one row per team per Stanley Cup final (`year`, `lgID`, `tmID`, `G`, `W`, `L`, `T`, `GF`, `GA`, `PIM`).
- **Key:** (`year`, `tmID`) is unique.
- **Coverage:** 30 rows, 1912 to 1925. PCHA 12, NHL 9, NHA 5, WCHL 4. Two teams per year in 12 years, three in 2 years.
- **Known before the season:** nothing.
- **Hidden in test data:** remove all rows (the table has no rows after 1925).
- **Quality notes:**
- **Links to:** `Teams` on (`year`, `tmID`).

## TeamVsTeam
- **What it is:** one row per team per opponent per season with the team's record against that opponent (`year`, `lgID`, `tmID`, `oppID`, `W`, `L`, `T`, `OTL`).
- **Key:** (`year`, `tmID`, `oppID`) is unique. No team plays itself, and every row has a reverse row (opponent against team).
- **Coverage:** 23,862 rows, 1909 to 2009. NHL 22,498, WHA 916, NHA 214, WCHL 120, PCHA 114.
- **Known before the season:** not decided.
- **Hidden in test data:** not decided.
- **Quality notes:**
- **Links to:** `Teams` on (`year`, `tmID`) and (`year`, `oppID`).

## AwardsCoaches
- **What it is:** one row per coach per award per year (`coachID`, `award`, `year`, `lgID`, `note`).
- **Key:** (`coachID`, `award`, `year`) is unique. There is also exactly one row per (`award`, `year`).
- **Coverage:** 75 rows, 1930 to 2009. NHL 68, WHA 7. Awards: Jack Adams 36 (1973 to 2009), First Team All-Star 16 and Second Team All-Star 16 (1930 to 1945), Schmertz 5 (1974 to 1978), Baldwin 2 (1972 to 1973).
- **Known before the season:** nothing.
- **Hidden in test data:** remove all rows.
- **Quality notes:**
  - `note` is blank in all 75 rows.
- **Links to:** `Master` on `coachID`, `Coaches` on `coachID`.

## AwardsMisc
- **What it is:** one row per person per award per year (`name`, `ID`, `award`, `year`, `lgID`, `note`).
- **Key:** (`name`, `award`, `year`) is unique.
- **Coverage:** 120 rows, 1965 to 2009, all NHL. The only value of `award` is `Patrick`. Between 1 and 5 rows per year.
- **Known before the season:** nothing.
- **Hidden in test data:** remove all rows.
- **Quality notes:**
  - `ID` is blank in 37 rows.
- **Links to:** `Master` on `ID` (partial).

## AwardsPlayers
- **What it is:** one row per player per award per year (`playerID`, `award`, `year`, `lgID`, `note`, `pos`).
- **Key:** (`playerID`, `award`, `year`) is unique.
- **Coverage:** 2021 rows, 1917 to 2009. NHL 1890, WHA 131. 28 distinct awards.
- **Known before the season:** nothing.
- **Hidden in test data:** remove all rows.
- **Quality notes:**
- **Links to:** `Master` on `playerID`.

## GoaliesSC
- **What it is:** one row per goalie per Stanley Cup final (`playerID`, `year`, `tmID`, `lgID`, `GP`, `Min`, `W`, `L`, `T`, `SHO`, `GA`).
- **Key:** (`playerID`, `year`) is unique.
- **Coverage:** 31 rows, 1912 to 1925. PCHA 12, NHL 10, NHA 5, WCHL 4. No blanks.
- **Known before the season:** nothing.
- **Hidden in test data:** remove all rows (the table has no rows after 1925).
- **Quality notes:**
- **Links to:** `Teams` on (`year`, `tmID`), `Master` on `playerID`.

## HOF
- **What it is:** one row per Hall of Fame member (`year`, `hofID`, `name`, `category`).
- **Key:** `hofID` is unique.
- **Coverage:** 356 rows, 1945 to 2009, 2 to 33 rows per year, 5 in 2009. Categories: Player 244, Builder 97, Referee/Linesman 15. No blanks.
- **Known before the season:** nothing.
- **Hidden in test data:** remove all rows. `Master.hofID` is also hidden.
- **Quality notes:**
- **Links to:** `Master` on `hofID`.

## ScoringSC
- **What it is:** one row per skater per Stanley Cup final (`playerID`, `year`, `tmID`, `lgID`, `pos`, `GP`, `G`, `A`, `Pts`, `PIM`).
- **Key:** (`playerID`, `year`) is unique.
- **Coverage:** 284 rows, 1912 to 1925. PCHA 107, NHL 89, NHA 46, WCHL 42. No blanks.
- **Known before the season:** nothing.
- **Hidden in test data:** remove all rows (the table has no rows after 1925).
- **Quality notes:**
- **Links to:** `Teams` on (`year`, `tmID`), `Master` on `playerID`.

## ScoringSup
- **What it is:** one row per player per year with `PPA` and `SHA` (`playerID`, `year`, `PPA`, `SHA`).
- **Key:** (`playerID`, `year`) is unique.
- **Coverage:** 137 rows, 1987 to 1990 (29, 39, 34, 35 per year).
- **Known before the season:** not decided.
- **Hidden in test data:** not decided (the table has no rows after 1990).
- **Quality notes:**
- **Links to:** `Scoring` on (`playerID`, `year`), `Master` on `playerID`.