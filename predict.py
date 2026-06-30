import joblib
import pandas as pd
from data_loader import load_data, get_match_data
from preprocessor import preprocess, add_team_form, add_head_to_head, add_venue_win_rate

model = joblib.load('model.pkl')
encoders = joblib.load('encoders.pkl')

# need the full history to work out current form/h2h/venue stats
# loading this once when app starts, not every time someone predicts
_df = load_data('IPL.csv')
_df = get_match_data(_df)
_df = preprocess(_df)
_df = add_team_form(_df)
_df = add_head_to_head(_df)
_df = add_venue_win_rate(_df)


def get_team_form(team, window=5):
    # last 5 matches for this team, win or lose
    team_matches = _df[(_df['batting_team'] == team) | (_df['bowling_team'] == team)]
    recent = team_matches.tail(window)
    if recent.empty:
        return 0.5
    wins = (recent['match_won_by'] == team).sum()
    return wins / len(recent)


def get_h2h_rate(team1, team2):
    # how often team1 beats team2 historically
    mask = ((_df['batting_team'] == team1) & (_df['bowling_team'] == team2)) | \
           ((_df['batting_team'] == team2) & (_df['bowling_team'] == team1))
    matches = _df[mask]
    if matches.empty:
        return 0.5
    wins = (matches['match_won_by'] == team1).sum()
    return wins / len(matches)


def get_venue_rate(team, venue):
    # team's win rate at this specific ground
    mask = (_df['batting_team'] == team) & (_df['venue'] == venue)
    matches = _df[mask]
    if matches.empty:
        return 0.5
    wins = (matches['match_won_by'] == team).sum()
    return wins / len(matches)


def predict_winner(team1, team2, venue, toss_winner, toss_decision):
    try:
        t1 = encoders['batting_team'].transform([team1])[0]
        t2 = encoders['bowling_team'].transform([team2])[0]
        v  = encoders['venue'].transform([venue])[0]
        tw = encoders['toss_winner'].transform([toss_winner])[0]
        td = encoders['toss_decision'].transform([toss_decision])[0]
    except ValueError as e:
        # team or venue not seen in training data
        return None, f"unknown value: {e}"

    # model also needs form/h2h/venue now, was missing this before
    bat_form = get_team_form(team1)
    bowl_form = get_team_form(team2)
    h2h = get_h2h_rate(team1, team2)
    venue_rate = get_venue_rate(team1, venue)

    input_df = pd.DataFrame([[t1, t2, v, tw, td, bat_form, bowl_form, h2h, venue_rate]],
        columns=['batting_team_enc', 'bowling_team_enc',
                 'venue_enc', 'toss_winner_enc',
                 'toss_decision_enc', 'batting_team_form',
                 'bowling_team_form', 'h2h_win_rate', 'venue_win_rate'])

    pred = model.predict(input_df)[0]
    winner = encoders['target'].inverse_transform([pred])[0]
    return winner, None