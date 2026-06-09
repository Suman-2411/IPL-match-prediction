import pandas as pd 

# The dataset is ballby ball.it needs to coverted to match 
def load_data(filepath):
    df=pd.read_csv(filepath,low_memory=False)
    print(f'Loaded {len(df)} rows!')
    return df

def get_match_data(df):
    # Get one row per match
    # each match has many ball by ball info rows
    # we only need match level info
    match_df = df.groupby('match_id').first().reset_index()

    # keep the rows which are useful
    cols=[
        'match_id','date','season','batting_team',
        'bowling_team','venue','city','toss_winner',
        'toss_decision','match_won_by','win_outcome','player_of_match'
    ]

    match_df = match_df[cols]

    print(f'Total matches: {len(match_df)}')
    print(match_df.head())
    return match_df

# To test it 
df = load_data('IPL.csv')
matches =get_match_data(df)

#To save match data info
matches.to_csv('matches_clean.csv',index=False)
print("Saved matches_clean.csv!")