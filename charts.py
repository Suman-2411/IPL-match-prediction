import plotly.express as px
import pandas as pd

def team_wins_chart(df):
    wins = df['match_won_by'].value_counts().reset_index()
    wins.columns = ['team', 'wins']

    fig = px.bar(wins,
                 x='team',
                 y='wins',
                 title='Total Wins by Team',
                 color='wins',
                 color_continuous_scale='reds')

    fig.update_layout(xaxis_tickangle=-45)
    return fig

def toss_win_chart(df):
    # checking if toss winner and match winner are same
    df['toss_helped'] = df['toss_winner'] == df['match_won_by']
    toss_counts = df['toss_helped'].value_counts().reset_index()
    toss_counts.columns = ['toss_helped', 'count']
    toss_counts['toss_helped'] = toss_counts['toss_helped'].map(
        {True: 'Won toss & match', False: 'Won toss lost match'}
    )

    fig = px.pie(toss_counts,
                 values='count',
                 names='toss_helped',
                 title='Does Winning Toss Help?',
                 color_discrete_sequence=['#ff4444', '#00cc66'])
    return fig

def season_wins_chart(df, team):
    team_df = df[df['match_won_by'] == team]
    season_wins = team_df['season'].value_counts().sort_index().reset_index()
    season_wins.columns = ['season', 'wins']

    fig = px.line(season_wins,
                  x='season',
                  y='wins',
                  title=f'{team} wins per season',
                  markers=True)
    return fig

def head_to_head_chart(df, team1, team2):
    # filtering matches where these two teams played each other
    mask = ((df['batting_team'] == team1) & (df['bowling_team'] == team2)) | \
           ((df['batting_team'] == team2) & (df['bowling_team'] == team1))

    h2h = df[mask]['match_won_by'].value_counts().reset_index()
    h2h.columns = ['team', 'wins']

    fig = px.bar(h2h,
                 x='team',
                 y='wins',
                 title=f'{team1} vs {team2}',
                 color='team')
    return fig