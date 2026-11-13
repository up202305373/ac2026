import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def prep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sorts stints and calculates the baseline Win_Pct feature."""
    df = df.sort_values(by=['tmID', 'year', 'stint'])
    last_coach = df.drop_duplicates(subset=['tmID', 'year'], keep='last').copy()
    
    last_coach['Total_Games'] = last_coach[['W', 'L', 'T', 'OTL']].fillna(0).sum(axis=1)
    last_coach['Win_Pct'] = last_coach['W'] / last_coach['Total_Games']
    
    return last_coach

def train_and_predict_rf(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    # 1. Feature Engineering for Training Data
    train_df = prep_features(train_data)
    
    # Create the Target (y): Did the coach change the following year?
    train_df['Next_Year_Coach'] = train_df.groupby('tmID')['coachID'].shift(-1)
    train_df['Coach_Changed'] = (train_df['coachID'] != train_df['Next_Year_Coach']).astype(int)
    train_df = train_df.dropna(subset=['Next_Year_Coach']).copy()
    
    X_train = train_df[['Win_Pct']].fillna(0)
    y_train = train_df['Coach_Changed']
    
    # 2. Feature Engineering for Testing Data
    test_df = prep_features(test_data)
    X_test = test_df[['Win_Pct']].fillna(0)
    
    # 3. Train the Model
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # 4. Generate Predictions
    test_df['predicted_change'] = clf.predict(X_test)
    
    # Return strictly the columns required by evaluate_coach_change
    return test_df[['year', 'tmID', 'predicted_change']]