import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from data_loader import load_data, get_match_data
from predict import predict_winner
from charts import (team_wins_chart, toss_win_chart,
                    season_wins_chart, head_to_head_chart)
from metrics import calculate_metrics

@st.cache_data
def load():
    df = load_data('IPL.csv')
    return get_match_data(df)

df = load()

teams = sorted(df['match_won_by'].dropna().unique().tolist())
venues = sorted(df['venue'].dropna().unique().tolist())

page = st.sidebar.selectbox("Navigate", [
    "🏠 Home",
    "🏏 Predict Winner",
    "📊 Team Stats",
    "⚔️ Head to Head",
    "📈 Model Metrics"
])

st.markdown("""
<style>
.stApp { background-color: #0a0a0a; }
h1, h2, h3 { color: #00cc44 !important; }
p { color: #cccccc !important; }
</style>
""", unsafe_allow_html=True)


if page == "🏠 Home":
    st.title("🏏 IPL Match Predictor")
    st.caption("predicting ipl winners using machine learning")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Matches", len(df))
    with col2:
        st.metric("Teams", df['match_won_by'].nunique())
    with col3:
        st.metric("Seasons", df['season'].nunique())

    st.divider()
    st.subheader("Most Wins")
    st.plotly_chart(team_wins_chart(df), use_container_width=True)

    st.divider()
    st.subheader("Does Toss Matter?")
    st.plotly_chart(toss_win_chart(df), use_container_width=True)


elif page == "🏏 Predict Winner":
    st.title("🏏 Predict Match Winner")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Batting Team", teams)
    with col2:
        team2 = st.selectbox("Bowling Team", teams)

    venue = st.selectbox("Venue", venues)

    col1, col2 = st.columns(2)
    with col1:
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
    with col2:
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

    if st.button("Predict!"):
        if team1 == team2:
            st.warning("Please select different teams!")
        else:
            with st.spinner("predicting..."):
                winner, error = predict_winner(
                    team1, team2, venue,
                    toss_winner, toss_decision
                )

            if error:
                st.error(f"Error: {error}")
            else:
                st.success(f"🏆 Predicted Winner: {winner}")

                col1, col2 = st.columns(2)
                with col1:
                    if winner == team1:
                        st.success(f"✅ {team1}")
                    else:
                        st.error(f"❌ {team1}")
                with col2:
                    if winner == team2:
                        st.success(f"✅ {team2}")
                    else:
                        st.error(f"❌ {team2}")


elif page == "📊 Team Stats":
    st.title("📊 Team Stats")
    st.divider()

    team = st.selectbox("Select Team", teams)
    total = len(df[df['match_won_by'] == team])
    st.metric("Total Wins", total)

    st.divider()
    st.subheader(f"{team} wins per season")
    st.plotly_chart(season_wins_chart(df, team), use_container_width=True)


elif page == "⚔️ Head to Head":
    st.title("⚔️ Head to Head")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        t1 = st.selectbox("Team 1", teams)
    with col2:
        t2 = st.selectbox("Team 2", teams)

    if t1 == t2:
        st.warning("Please select different teams!")
    else:
        mask = (
            ((df['batting_team'] == t1) & (df['bowling_team'] == t2)) |
            ((df['batting_team'] == t2) & (df['bowling_team'] == t1))
        )
        st.metric("Total Matches", mask.sum())
        st.plotly_chart(head_to_head_chart(df, t1, t2), use_container_width=True)

elif page == "📈 Model Metrics":
    st.title("📈 Model Metrics")
    st.divider()

    try:
        model = joblib.load('model.pkl')
        X_test = joblib.load('X_test.pkl')
        y_test = joblib.load('y_test.pkl')

        m = calculate_metrics(model, X_test, y_test)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{m['accuracy']:.2%}")
            st.metric("F1 Score", f"{m['f1']:.2%}")
        with col2:
            st.metric("Precision", f"{m['precision']:.2%}")
            st.metric("Recall", f"{m['recall']:.2%}")

        st.divider()
        st.subheader("Feature Importance")

        features = ['batting_team', 'bowling_team', 'venue', 'toss_winner', 'toss_decision', 'batting_team_form', 'bowling_team_form', 'h2h_win_rate', 'venue_win_rate']
        fig = px.bar(x=features, y=model.feature_importances_,
                     title="which features matter most?")
        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.error("run train_model.py first to generate the pkl files!")