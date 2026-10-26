# Questions

## Question 1

Load Management: Resting key players on the final games of the season when a playoff spot has already been secured.
Tanking: Building a non-competitive roster to lose as many games as possible in order to get better draft picks on the following year. 
Games might have different weights in their importance in the season (e.g.: final games of teams who already qualified for playoffs). Does the data represent this somehow or is this overlooked?

## Question 2

What exactly is known a day before the season starts? This is what we are currently assuming for the target year and following years:

- **abbrev:** We assume that we can keep all the information from here (Type,Code,Fullname).

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
