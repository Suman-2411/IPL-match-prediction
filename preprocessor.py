import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    # removing matches where winner info is missing
    return df.dropna(subset=['match_won_by']).copy()

def encode_features(df):
    # ml models dont understand text so converting to numbers
    encoders = {}
    cols = ['batting_team','bowling_team','venue','toss_winner','toss_decision']
    
    for c in cols:
        le = LabelEncoder()
        df[c+'_enc'] = le.fit_transform(df[c])
        # saving each encoder so we can reuse later during prediction
        encoders[c] = le

    le_target = LabelEncoder()
    df['target'] = le_target.fit_transform(df['match_won_by'])
    encoders['target'] = le_target
    return df, encoders

def get_features(df):
    # these are the columns we feed into the model
    cols = ['batting_team_enc','bowling_team_enc','venue_enc',
            'toss_winner_enc','toss_decision_enc']
    return df[cols], df['target']