import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess(df):
    df = df.dropna(subset=['match_won_by']).copy()
    df = df.sort_values('date').reset_index(drop=True)
    return df


def add_team_form(df, window=5):
    # how good was each team in their last 5 games
    df['batting_team_form'] = 0.0
    df['bowling_team_form'] = 0.0

    team_history = {}

    for i, row in df.iterrows():
        bat_team = row['batting_team']
        bowl_team = row['bowling_team']
        winner = row['match_won_by']

        bat_hist = team_history.get(bat_team, [])
        bowl_hist = team_history.get(bowl_team, [])

        # no history yet so just guess 50-50
        df.at[i, 'batting_team_form'] = sum(bat_hist[-window:]) / len(bat_hist[-window:]) if bat_hist else 0.5
        df.at[i, 'bowling_team_form'] = sum(bowl_hist[-window:]) / len(bowl_hist[-window:]) if bowl_hist else 0.5

        team_history.setdefault(bat_team, []).append(1 if winner == bat_team else 0)
        team_history.setdefault(bowl_team, []).append(1 if winner == bowl_team else 0)

    return df


def add_head_to_head(df):
    # these two teams played before, who usually wins
    df['h2h_win_rate'] = 0.5
    h2h_history = {}

    for i, row in df.iterrows():
        t1, t2 = row['batting_team'], row['bowling_team']
        key = tuple(sorted([t1, t2]))
        wins, total = h2h_history.get(key, [0, 0])

        if total > 0:
            rate = wins / total if key[0] == t1 else (total - wins) / total
            df.at[i, 'h2h_win_rate'] = rate

        winner = row['match_won_by']
        new_wins = wins + (1 if winner == key[0] else 0)
        h2h_history[key] = [new_wins, total + 1]

    return df


def add_venue_win_rate(df):
    # team usually wins or loses at this ground?
    df['venue_win_rate'] = 0.5
    venue_history = {}

    for i, row in df.iterrows():
        team = row['batting_team']
        venue = row['venue']
        key = (team, venue)
        wins, total = venue_history.get(key, [0, 0])

        if total > 0:
            df.at[i, 'venue_win_rate'] = wins / total

        winner = row['match_won_by']
        new_wins = wins + (1 if winner == team else 0)
        venue_history[key] = [new_wins, total + 1]

    return df


def encode_features(df):
    encoders = {}
    cols = ['batting_team', 'bowling_team', 'venue', 'toss_winner', 'toss_decision']

    for c in cols:
        le = LabelEncoder()
        df[c + '_enc'] = le.fit_transform(df[c])
        encoders[c] = le

    le_target = LabelEncoder()
    df['target'] = le_target.fit_transform(df['match_won_by'])
    encoders['target'] = le_target
    return df, encoders


def get_features(df):
    cols = ['batting_team_enc', 'bowling_team_enc', 'venue_enc',
            'toss_winner_enc', 'toss_decision_enc',
            'batting_team_form', 'bowling_team_form',
            'h2h_win_rate', 'venue_win_rate']
    return df[cols], df['target']