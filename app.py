"""
Streamlit app for the soccer prediction model.
Predicts match outcome probabilities for a chosen home vs away team.
"""
import os
import pandas as pd
import numpy as np
import glob
import requests
import streamlit as st
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Soccer Predictor", page_icon="⚽", layout="wide")


# ---------- 1. Load data (cached so it only runs once) ----------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    frames = []
    for f in sorted(glob.glob(os.path.join(data_dir, 'season-*.csv'))):
        df = pd.read_csv(f)
        frames.append(df)
    matches = pd.concat(frames, ignore_index=True)
    matches['Date'] = pd.to_datetime(matches['Date'], format='%d/%m/%Y')
    matches = matches.sort_values('Date').reset_index(drop=True)
    matches = matches.dropna(subset=['FTHG', 'FTAG', 'FTR'])
    return matches

matches = load_data()

# 2. Compute Features (Walk-forward)
@st.cache_data
def compute_features(matches_df):
    elo = {}
    K = 20
    HOME_ADV = 60
    team_history = {}
    last_played = {}

    pre_match_home_elo = []
    pre_match_away_elo = []
    home_form = []
    away_form = []
    home_rest = []
    away_rest = []

    def get_elo(team):
        return elo.get(team, 1500)

    def expected_score(r_a, r_b):
        return 1 / (1 + 10 ** ((r_b - r_a) / 400))

    def get_form(team, n=5):
        hist = team_history.get(team, [])
        if not hist:
            return 1.0
        return np.mean([p for _, p in hist[-n:]])

    for _, row in matches_df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        d = row['Date']
        
        r_h, r_a = get_elo(h), get_elo(a)
        pre_match_home_elo.append(r_h)
        pre_match_away_elo.append(r_a)
        
        s_h = 1.0 if row['FTR'] == 'H' else (0.5 if row['FTR'] == 'D' else 0.0)
        exp_h = expected_score(r_h + HOME_ADV, r_a)
        elo[h] = r_h + K * (s_h - exp_h)
        elo[a] = r_a + K * ((1 - s_h) - (1 - exp_h))
        
        home_form.append(get_form(h))
        away_form.append(get_form(a))
        
        ph = 3 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0)
        pa = 3 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 0)
        team_history.setdefault(h, []).append((d, ph))
        team_history.setdefault(a, []).append((d, pa))
        
        home_rest.append((d - last_played[h]).days if h in last_played else 14)
        away_rest.append((d - last_played[a]).days if a in last_played else 14)
        last_played[h] = d
        last_played[a] = d
    matches['rest_diff'] = [hr - ar for hr, ar in zip(home_rest, away_rest)]

    label_map = {'H': 0, 'D': 1, 'A': 2}
    matches['target'] = matches['FTR'].map(label_map)

    # Build "current state" per team: final Elo and final form as of the last match
    final_elo = dict(elo)
    final_form = {t: get_form(t) for t in elo.keys()}
    final_rest = {t: (matches_df['Date'].max() - d).days for t, d in last_played.items()}
    
    return matches_df, final_elo, final_form, final_rest

matches, final_elo, final_form, final_rest = compute_features(matches)

# 3. Load Model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.joblib')
        feature_cols = joblib.load('feature_cols.joblib')
        return model, feature_cols
    except FileNotFoundError:
        st.error("Model not found. Please run `python soccer_pipeline.py` first to train and save the model.")
        st.stop()

model, feature_cols = train_model(matches)

# ---------- 4. Build the UI ----------
st.title("⚽ Soccer Match Predictor")
st.write("Pick a home team and an away team to see the model's predicted probabilities.")

teams = sorted(final_elo.keys())
default_home = teams.index("Arsenal") if "Arsenal" in teams else 0
default_away = teams.index("Man City") if "Man City" in teams else 1
home_team = st.selectbox("Home team", teams, index=default_home)
away_team = st.selectbox("Away team", teams, index=default_away)

if home_team == away_team:
    st.warning("Home and away teams must be different.")
    st.stop()

# ---------- 5. Compute features for the chosen matchup and predict ----------
elo_diff = final_elo[home_team] - final_elo[away_team]
form_diff = final_form[home_team] - final_form[away_team]
rest_diff = final_rest[home_team] - final_rest[away_team]

X = pd.DataFrame([[elo_diff, form_diff, rest_diff]], columns=feature_cols)
proba = model.predict_proba(X)[0]

# ---------- 6. Display results ----------
st.subheader(f"Prediction: {home_team} vs {away_team}")

col1, col2, col3 = st.columns(3)
col1.metric("Home win", f"{proba[0]*100:.1f}%")
col2.metric("Draw", f"{proba[1]*100:.1f}%")
col3.metric("Away win", f"{proba[2]*100:.1f}%")

chart_data = pd.DataFrame({
    'Outcome': ['Home win', 'Draw', 'Away win'],
    'Probability': [proba[0], proba[1], proba[2]],
})
st.bar_chart(chart_data, x='Outcome', y='Probability', height=250)

with st.expander("Show features used by the model"):
    st.write(f"Elo rating — {home_team}: {final_elo[home_team]:.0f}, {away_team}: {final_elo[away_team]:.0f}")
    st.write(f"Recent form (points/game) — {home_team}: {final_form[home_team]:.2f}, {away_team}: {final_form[away_team]:.2f}")
    st.write(f"Elo difference: {elo_diff:+.0f}")
    st.write(f"Form difference: {form_diff:+.2f}")
    st.write(f"Rest-days difference: {rest_diff:+.0f}")

st.caption("⚠️ Predictions use each team's Elo and form as of the last match in the dataset "
           "(May 2026). Not intended for betting advice.")

st.caption("Model and code: [github.com/CharlieTC-blake/soccer_prediction](https://github.com/CharlieTC-blake/soccer_prediction)")