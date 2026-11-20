import pandas as pd

def add_games_scheduled(teams_vs_teams: pd.DataFrame) -> pd.DataFrame:
    """Add the number of games each pair of teams plays in a season."""
    """Computed from the raw W/L/T/OTL before the sampler hides them."""
    """Schedules are known months before the season starts, so there's no leakage in using this information to predict the season's results."""
    tv = teams_vs_teams.copy()
    cols = ["W", "L", "T", "OTL"]
    values = tv[cols].apply(pd.to_numeric, errors="coerce")
    tv["G"] = values.fillna(0).sum(axis=1)
    return tv

def build_derived(tables: dict) -> dict:
    tables = dict(tables)
    tables["TeamVsTeam"] = add_games_scheduled(tables["TeamVsTeam"])
    return tables