# Tables

This file has important information regarding all tables.

## TableName
- **What it is:** one line, including what a row represents.
- **Key:** e.g. one row per (year, tmID).
- **Coverage:** first and last year, row count, leagues.
- **Known before the season:** columns the test data keeps.
- **Hidden in test data:** columns or rows removed, and why.
- **Quality notes:** gaps, blanks, oddities, with how you found them.
- **Used for:** (a), (b), both, or "not used".

## abbrev
- **What it is:** lookup of codes and their meanings, used by `Teams` and `SeriesPost`.
- **Key:** one row per (Type, Code).
- **Coverage:** 58 rows. Types: Playoffs (24), Division (17), Round (13), Conference (4).
- **Known before the season:** everything (Type, Code, Fullname).
- **Hidden in test data:** nothing.
- **Quality notes:** `Code` is not unique on its own (`CF`, `F`, `SF` appear under both Playoffs and Round), so join on (Type, Code).
- **Used for:** decoding `Teams.playoff` and `SeriesPost.round`.

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
  - `notes` is almost empty: 1730 of 1741 blank, 9 `interim`, 2 co-coach. [verified]
  - Interim spells can be tiny: 1 to 6 games.
  - Playoff columns are blank in 846 rows. Likely "no playoff games", not missing data.
  - Team-seasons with no coach row: only 5, all outside the NHL (REN 1909 and 1910, TO1 1912, and the WHA's CZE and FIN in 1978). Every NHL team-season has a coach.
  - A coach can appear with two teams in one season.
  - Column names are lowercase here (`g`, `w`, `l`, `t`), and uppercase in `Teams`. Case matters in pandas, not in SQLite.
- **Used for:**














- **Coaches:** We assume that, since predictions are made a day before the season starts, the coach starting the season (stint 1) for a team in 2007 is known, but their results in the season are unknown (coachID,2007,tmID,lgID,1,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_). If predicting for multiple seasons, in the following season the coach would be unknown (\_\_\_\_\_\_,2008,tmID,lgID,1,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_). Rows with stint > 1 are also entirely removed so the model can't know in advance what teams will change coaches, which is the entire point of task b).

- **Teams:** We assume that information about the teams competing is known, however their results are unknown (2007,lgID,tmID,franchID,confID,divID,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,name,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_).

- **TeamsHalf:** We assume that information about the teams competing is known and that it is known that hockey games have 2 halves (year,lgID,tmID,half,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_).

- **TeamSplits:** We assume that information about the teams competing is known but not their statistics (year,lgID,tmID,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_)

- **Scoring:** We assume that information about the player is known, however his statistics are unknown (playerID,2007,1,tmID,lgID,pos,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_). Rows with stint > 1 are also entirely removed so the model can't know in advance what players will change teams.

- **Goalies:** We assume that information about the player is known, however his statistics are unknown (playerID,2007,1,tmID,lgID,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_). Rows with stint > 1 are also entirely removed so the model can't know in advance what goalies will change teams.

- **Master:** We assume that some information about all people is known (playerID,coachID,\_\_,firstName,lastName,nameNote,nameGiven,nameNick,height,weight,shootCatch,legendsID,ihdbID,hrefID,firstNHL,\_\_,firstWHA,\_\_,pos,birthYear,birthMon,birthDay,birthCountry,birthState,birthCity,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_). Some information regarding future events should be hidden:
  - The hofID is hidden to not reveal that the player at hand will be in the hall of fame in the future (and the corresponding row in HOF is also removed).
  - The lastNHL and lastWHA are hidden to not reveal the final year of a player's career for players that are yet to retire.
  - All information related to a death that occurs after the day before the start of the season should be hidden.

- **SeriesPost:** We assume that the format of the series is known but not the amount of games it ends up having or the teams participating (year,round,series,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_)

- **ScoringShootout:** Due to the lack of information on players with 0 S,G,GDG, we assume that players only make it to this list if they are involved in a shootout. Since we don't know beforehand what teams will even be involved in shootouts, it's probably best to remove all rows for the target and following years.

- **GoaliesShootout, *CombinedShutouts:** Similar reasoning as the previous one, remove all rows for target and following years.

- **TeamsPost:** We don't initially know what teams reach the post season, but we assume that the amount of teams in the post season coming from a league is known before the start of the season (year,lgID,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_,\_\_)

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