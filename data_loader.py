import pandas as pd

def load_data(filepath):
    # reading the csv file
    df = pd.read_csv(filepath, low_memory=False)
    return df

def get_match_data(df):
    # ball by ball data has many rows per match
    # taking first row of each match to get match level info
    match_df = df.groupby('match_id').first().reset_index()
    keep = ['match_id','date','season','batting_team','bowling_team',
            'venue','city','toss_winner','toss_decision',
            'match_won_by','win_outcome','player_of_match']
    return match_df[keep]