import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def prep_features(df: pd.DataFrame, 
                  splits_df: pd.DataFrame, 
                  awards_df: pd.DataFrame,
                  scoring_df: pd.DataFrame = None,
                  awards_players_df: pd.DataFrame = None,
                  goalies_df: pd.DataFrame = None,
                  master_df: pd.DataFrame = None,
                  teams_half_df: pd.DataFrame = None) -> pd.DataFrame:
    """Calculates advanced contextual metrics and handles mid-season stint realities."""
    df = df.sort_values(by=['tmID', 'year', 'stint'])
    
    # FEATURE: Coach Tenure (Years with the same team)
    df['Tenure'] = df.groupby(['tmID', 'coachID']).cumcount() + 1
    
    # FEATURE: Career Leash (Total years coached across all teams up to this point)
    df['Career_Years'] = df.groupby('coachID').cumcount() + 1
    
    # Isolate the final coach of the year for the target calculation
    last_coach = df.drop_duplicates(subset=['tmID', 'year'], keep='last').copy()
    
    # FEATURE: Coach Age (Differentiates performance firings from retirements)
    if master_df is not None and not master_df.empty:
        master_ages = master_df[['coachID', 'birthYear']].dropna(subset=['coachID']).drop_duplicates(subset=['coachID'])
        last_coach = pd.merge(last_coach, master_ages, on='coachID', how='left')
        last_coach['Coach_Age'] = last_coach['year'] - pd.to_numeric(last_coach['birthYear'], errors='coerce')
        last_coach['Coach_Age'] = last_coach['Coach_Age'].fillna(50)  # Impute average age for missing birth years
    else:
        last_coach['Coach_Age'] = 50
    
    # FEATURE: Era-Agnostic Win Percentage & Goal Differential
    last_coach['Total_Games'] = last_coach[['W', 'L', 'T', 'OTL']].fillna(0).sum(axis=1)
    last_coach['Win_Pct'] = last_coach['W'] / last_coach['Total_Games']
    last_coach['Goal_Diff'] = last_coach['GF'] - last_coach['GA']
    
    # FEATURE: Playoff Success
    last_coach['Made_Playoffs'] = last_coach['playoff'].notna() & (last_coach['playoff'] != 'N')
    last_coach['Made_Playoffs'] = last_coach['Made_Playoffs'].astype(int)
    
    last_coach['Post_Games'] = last_coach[['postw', 'postl', 'postt']].fillna(0).sum(axis=1)
    last_coach['Post_Win_Pct'] = np.where(
        last_coach['Post_Games'] > 0, 
        last_coach['postw'] / last_coach['Post_Games'], 
        0
    )
    
    # FEATURE: Special Teams Competence
    last_coach['PP_Pct'] = np.where(last_coach['PPC'] > 0, last_coach['PPG'] / last_coach['PPC'], 0)
    last_coach['PK_Pct'] = np.where(last_coach['PKC'] > 0, last_coach['PKG'] / last_coach['PKC'], 0)
    
    # FEATURE: Mid-Season Hire Leakage
    last_coach['Mid_Season_Hire'] = (last_coach['stint'] > 1).astype(int)
    
    # FEATURE: Standings Trajectory
    last_coach['Win_Pct_Last_Year'] = last_coach.groupby('tmID')['Win_Pct'].shift(1)
    last_coach['Win_Pct_Delta'] = last_coach['Win_Pct'] - last_coach['Win_Pct_Last_Year']
    
    # FEATURE: Hardware Immunity (Won a coaching award in the last 3 years)
    if awards_df is not None and not awards_df.empty:
        awards = awards_df[['coachID', 'year']].copy()
        awards['Has_Award'] = 1
        
        immunity_years = []
        for i in range(3):
            temp = awards.copy()
            temp['year'] = temp['year'] + i
            immunity_years.append(temp)
        
        immunity_df = pd.concat(immunity_years).drop_duplicates(subset=['coachID', 'year'])
        last_coach = pd.merge(last_coach, immunity_df, on=['coachID', 'year'], how='left')
        last_coach['Has_Award'] = last_coach['Has_Award'].fillna(0)
    else:
        last_coach['Has_Award'] = 0
        
    # FEATURE: Late Season Collapse
    if splits_df is not None and not splits_df.empty:
        splits_df['Early_Games'] = splits_df[['OctW', 'OctL', 'OctT', 'OctOL']].fillna(0).sum(axis=1)
        splits_df['Late_Games'] = splits_df[['MarW', 'MarL', 'MarT', 'MarOL']].fillna(0).sum(axis=1)
        
        splits_df['Early_Win_Pct'] = np.where(splits_df['Early_Games'] > 0, splits_df['OctW'] / splits_df['Early_Games'], 0)
        splits_df['Late_Win_Pct'] = np.where(splits_df['Late_Games'] > 0, splits_df['MarW'] / splits_df['Late_Games'], 0)
        
        splits_df['Late_Collapse'] = ((splits_df['Early_Win_Pct'] - splits_df['Late_Win_Pct']) > 0.2).astype(int)
        last_coach = pd.merge(last_coach, splits_df[['year', 'tmID', 'Late_Collapse']], on=['year', 'tmID'], how='left')
        last_coach['Late_Collapse'] = last_coach['Late_Collapse'].fillna(0)
    else:
        last_coach['Late_Collapse'] = 0

    # FEATURE: Goaltending Dependency (Team Save Percentage Context)
    if goalies_df is not None and not goalies_df.empty:
        team_goalies = goalies_df.groupby(['year', 'tmID'])[['SA', 'GA']].sum().reset_index()
        team_goalies['SV_Pct'] = np.where(team_goalies['SA'] > 0, (team_goalies['SA'] - team_goalies['GA']) / team_goalies['SA'], 0.900)
        last_coach = pd.merge(last_coach, team_goalies[['year', 'tmID', 'SV_Pct']], on=['year', 'tmID'], how='left')
        last_coach['SV_Pct'] = last_coach['SV_Pct'].fillna(0.900)
    else:
        last_coach['SV_Pct'] = 0.900

    # FEATURE: Star Power Expectations (Number of Elite Award-Winning Players on Roster)
    if awards_players_df is not None and scoring_df is not None:
        player_teams = scoring_df.sort_values(by=['playerID', 'year', 'stint']).drop_duplicates(subset=['playerID', 'year'], keep='last')
        player_awards = pd.merge(awards_players_df, player_teams[['playerID', 'year', 'tmID']], on=['playerID', 'year'], how='inner')
        team_stars = player_awards.groupby(['year', 'tmID']).size().reset_index(name='Star_Players')
        last_coach = pd.merge(last_coach, team_stars, on=['year', 'tmID'], how='left')
        last_coach['Star_Players'] = last_coach['Star_Players'].fillna(0)
    else:
        last_coach['Star_Players'] = 0

    # FEATURE: Second-Half Trajectory 
    if teams_half_df is not None and not teams_half_df.empty:
        half1 = teams_half_df[teams_half_df['half'] == 1].copy()
        half2 = teams_half_df[teams_half_df['half'] == 2].copy()
        
        half1['Total_Games_H1'] = half1[['W', 'L', 'T']].fillna(0).sum(axis=1)
        half1['Win_Pct_H1'] = np.where(half1['Total_Games_H1'] > 0, half1['W'] / half1['Total_Games_H1'], 0)
        
        half2['Total_Games_H2'] = half2[['W', 'L', 'T']].fillna(0).sum(axis=1)
        half2['Win_Pct_H2'] = np.where(half2['Total_Games_H2'] > 0, half2['W'] / half2['Total_Games_H2'], 0)
        
        halves = pd.merge(half1[['year', 'tmID', 'Win_Pct_H1']], half2[['year', 'tmID', 'Win_Pct_H2']], on=['year', 'tmID'], how='inner')
        halves['Second_Half_Delta'] = halves['Win_Pct_H2'] - halves['Win_Pct_H1']
        
        last_coach = pd.merge(last_coach, halves[['year', 'tmID', 'Second_Half_Delta']], on=['year', 'tmID'], how='left')
        last_coach['Second_Half_Delta'] = last_coach['Second_Half_Delta'].fillna(0)
    else:
        last_coach['Second_Half_Delta'] = 0

    return last_coach

def train_and_predict(train_data: pd.DataFrame, test_data: pd.DataFrame, 
                      train_splits: pd.DataFrame, train_awards: pd.DataFrame,
                      test_splits: pd.DataFrame, test_awards: pd.DataFrame,
                      train_scoring: pd.DataFrame = None, test_scoring: pd.DataFrame = None,
                      train_awards_players: pd.DataFrame = None, test_awards_players: pd.DataFrame = None,
                      train_goalies: pd.DataFrame = None, test_goalies: pd.DataFrame = None,
                      train_master: pd.DataFrame = None, test_master: pd.DataFrame = None,
                      train_teams_half: pd.DataFrame = None, test_teams_half: pd.DataFrame = None) -> pd.DataFrame:
                      
    # 1. Combine all train and test datasets for proper chronological feature engineering
    combined_data = pd.concat([train_data, test_data], ignore_index=True)
    combined_splits = pd.concat([train_splits, test_splits], ignore_index=True)
    combined_awards = pd.concat([train_awards, test_awards], ignore_index=True)
    
    combined_scoring = pd.concat([train_scoring, test_scoring], ignore_index=True) if train_scoring is not None else None
    combined_awards_players = pd.concat([train_awards_players, test_awards_players], ignore_index=True) if train_awards_players is not None else None
    combined_goalies = pd.concat([train_goalies, test_goalies], ignore_index=True) if train_goalies is not None else None
    combined_master = pd.concat([train_master, test_master], ignore_index=True) if train_master is not None else None
    combined_teams_half = pd.concat([train_teams_half, test_teams_half], ignore_index=True) if train_teams_half is not None else None
    
    # 2. Prep features on the combined dataset
    combined_df = prep_features(
        combined_data, combined_splits, combined_awards, 
        combined_scoring, combined_awards_players, combined_goalies, 
        combined_master, combined_teams_half
    )
    
    # 3. Split back into train and test datasets using the test year
    test_year = test_data['year'].max()
    train_df = combined_df[combined_df['year'] < test_year].copy()
    test_df = combined_df[combined_df['year'] == test_year].copy()
    
    # 4. Target Formulation for Training Data
    next_year_df = train_df[['tmID', 'year', 'coachID']].copy()
    next_year_df['year'] = next_year_df['year'] - 1 
    next_year_df = next_year_df.rename(columns={'coachID': 'Next_Year_Coach'})
    
    train_df = pd.merge(train_df, next_year_df, on=['tmID', 'year'], how='left')
    train_df = train_df.dropna(subset=['Next_Year_Coach']).copy()
    train_df['Coach_Changed'] = (train_df['coachID'] != train_df['Next_Year_Coach']).astype(int)
    
    train_df = train_df.dropna(subset=['Win_Pct', 'Win_Pct_Last_Year'])
    
    features = [
        'Tenure', 'Career_Years', 'Coach_Age', 'Win_Pct', 'Goal_Diff', 
        'Made_Playoffs', 'Post_Win_Pct', 'PP_Pct', 'PK_Pct', 
        'Mid_Season_Hire', 'Win_Pct_Last_Year', 'Win_Pct_Delta', 
        'Has_Award', 'Late_Collapse', 'SV_Pct', 'Star_Players', 
        'Second_Half_Delta'
    ]
    
    X_train = train_df[features].fillna(0)
    y_train = train_df['Coach_Changed']
    X_test = test_df[features].fillna(0)
    
    # 5. Hyperparameter Tuning implementation
    rf = RandomForestClassifier(class_weight='balanced_subsample', random_state=42)
    
    param_dist = {
        'n_estimators': [100, 150, 250],
        'max_depth': [5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # RandomizedSearch avoids exhaustive runtime costs while drastically improving parameter fit
    clf = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Predict on test year using the best estimator found
    test_df['change_probability'] = clf.predict_proba(X_test)[:, 1]
    test_df['predicted_change'] = (test_df['change_probability'] > 0.50).astype(int)
    
    return test_df[['year', 'tmID', 'change_probability', 'predicted_change']]