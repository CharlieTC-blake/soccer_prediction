"""
Streamlit app for the soccer prediction model.
Predicts match outcome probabilities for a chosen home vs away team.
"""
import os
import pandas as pd
import numpy as np
import glob
import streamlit as st
from sklearn.linear_model import LogisticRegression

# ---------- Setup ----------
st.set_page_config(page_title="Soccer Predictor", page_icon="⚽", layout="centered")


# ---------- 1. Load data (cached so it only runs once) ----------
@st.cache_data
def load_and_prepare():
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


matches = load_and_prepare()


# ---------- 2. Compute walk-forward features ----------
@st.cache_data
def compute_features(matches):
    elo = {}
    K = 20
    HOME_ADV = 60

    def get_elo(t):
        return elo.get(t, 1500)

    def expected_score(ra, rb):
        return 1 / (1 + 10 ** ((rb - ra) / 400))

    home_elo, away_elo = [], []
    for _, row in matches.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        rh, ra = get_elo(h), get_elo(a)
        home_elo.append(rh)
        away_elo.append(ra)
        if row['FTR'] == 'H':
            sh = 1.0
        elif row['FTR'] == 'D':
            sh = 0.5
        else:
            sh = 0.0
        exp_h = expected_score(rh + HOME_ADV, ra)
        elo[h] = rh + K * (sh - exp_h)
        elo[a] = ra + K * ((1 - sh) - (1 - exp_h))

    matches = matches.copy()
    matches['elo_diff'] = [he - ae for he, ae in zip(home_elo, away_elo)]

    team_history = {}

    def team_form(t, n=5):
        hist = team_history.get(t, [])
        if not hist:
            return 1.0
        return np.mean([p for _, p in hist[-n:]])

    home_form, away_form = [], []
    for _, row in matches.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        home_form.append(team_form(h))
        away_form.append(team_form(a))
        if row['FTR'] == 'H':
            ph, pa = 3, 0
        elif row['FTR'] == 'D':
            ph, pa = 1, 1
        else:
            ph, pa = 0, 3
        team_history.setdefault(h, []).append((row['Date'], ph))
        team_history.setdefault(a, []).append((row['Date'], pa))

    matches['form_diff'] = [hf - af for hf, af in zip(home_form, away_form)]

    last_played = {}
    home_rest, away_rest = [], []
    for _, row in matches.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        d = row['Date']
        home_rest.append((d - last_played[h]).days if h in last_played else 14)
        away_rest.append((d - last_played[a]).days if a in last_played else 14)
        last_played[h] = d
        last_played[a] = d
    matches['rest_diff'] = [hr - ar for hr, ar in zip(home_rest, away_rest)]

    label_map = {'H': 0, 'D': 1, 'A': 2}
    matches['target'] = matches['FTR'].map(label_map)

    # Build "current state" per team: final Elo and final form as of the last match
    final_elo = dict(elo)
    final_form = {t: team_form(t) for t in elo.keys()}
    final_rest = {t: (matches['Date'].max() - d).days
                  for t, d in last_played.items()}

    return matches, final_elo, final_form, final_rest


matches, final_elo, final_form, final_rest = compute_features(matches)


# ---------- 3. Train the model ----------
@st.cache_data
def train_model(matches):
    split = pd.Timestamp('2022-01-01')
    train = matches[matches['Date'] < split].copy()
    feature_cols = ['elo_diff', 'form_diff', 'rest_diff']
    model = LogisticRegression(max_iter=1000)
    model.fit(train[feature_cols], train['target'])
    return model, feature_cols


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