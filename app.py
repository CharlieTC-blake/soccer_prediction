"""
Streamlit app for the soccer prediction model.
Shows custom predictions, plus fixtures and results for any date.
"""
import os
import pandas as pd
import numpy as np
import glob
import requests
import streamlit as st
from sklearn.linear_model import LogisticRegression
from datetime import datetime

st.set_page_config(page_title="Soccer Predictor", page_icon="⚽", layout="wide")


# ---------- 1. Load historical data ----------
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


# ---------- 2. Compute walk-forward features ----------
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

    matches_df = matches_df.copy()
    matches_df['elo_diff'] = np.array(pre_match_home_elo) - np.array(pre_match_away_elo)
    matches_df['form_diff'] = np.array(home_form) - np.array(away_form)
    matches_df['rest_diff'] = np.array(home_rest) - np.array(away_rest)
    matches_df['target'] = matches_df['FTR'].map({'H': 0, 'D': 1, 'A': 2})

    final_elo = dict(elo)
    final_form = {t: get_form(t) for t in elo.keys()}
    final_rest = {t: (matches_df['Date'].max() - d).days for t, d in last_played.items()}

    return matches_df, final_elo, final_form, final_rest


matches, final_elo, final_form, final_rest = compute_features(matches)


# ---------- 3. Train the model ----------
@st.cache_data
def train_model(matches_df):
    split = pd.Timestamp('2022-01-01')
    train = matches_df[matches_df['Date'] < split].copy()
    feature_cols = ['elo_diff', 'form_diff', 'rest_diff']
    model = LogisticRegression(max_iter=1000)
    model.fit(train[feature_cols], train['target'])
    return model, feature_cols


model, feature_cols = train_model(matches)


# ---------- 4. Fetch fixtures for a date from openfootball ----------
@st.cache_data(ttl=3600)
def fetch_fixtures_for_date(date_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    if date_obj.month >= 8:
        season_year = date_obj.year
    else:
        season_year = date_obj.year - 1
    season_str = f"{season_year}-{str(season_year + 1)[2:]}"

    url = f"https://raw.githubusercontent.com/openfootball/football.json/master/{season_str}/en.1.json"

    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return None, f"No fixtures found for season {season_str}."

        data = response.json()
        all_matches = data.get("matches", [])
        day_matches = [m for m in all_matches if m.get("date") == date_str]

        if not day_matches:
            return None, f"No Premier League fixtures on {date_str}."

        formatted = []
        for m in day_matches:
            home_team = m.get("team1", "")
            away_team = m.get("team2", "")
            score = m.get("score")
            home_goals, away_goals = None, None

            if isinstance(score, dict):
                ft = score.get("ft")
                if isinstance(ft, list) and len(ft) == 2:
                    home_goals, away_goals = ft[0], ft[1]
            elif isinstance(score, list) and len(score) == 2:
                home_goals, away_goals = score[0], score[1]

            if home_goals is not None:
                status_short = "FT"
                status_long = "FINISHED"
            else:
                status_short = "NS"
                status_long = "SCHEDULED"

            formatted.append({
                "teams": {
                    "home": {"name": home_team},
                    "away": {"name": away_team},
                },
                "fixture": {
                    "status": {"short": status_short, "long": status_long},
                },
                "goals": {"home": home_goals, "away": away_goals},
            })

        return formatted, None

    except Exception as e:
        return None, f"Error fetching fixtures: {e}"


# ---------- 5. Team name translation ----------
TEAM_NAME_MAP = {
    "Manchester City FC": "Man City", "Manchester City": "Man City",
    "Manchester United FC": "Man United", "Manchester United": "Man United",
    "Newcastle United FC": "Newcastle", "Newcastle United": "Newcastle",
    "Wolverhampton Wanderers FC": "Wolves", "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest FC": "Nott'm Forest", "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur FC": "Tottenham", "Tottenham Hotspur": "Tottenham",
    "Brighton & Hove Albion FC": "Brighton", "Brighton & Hove Albion": "Brighton",
    "Sheffield United FC": "Sheffield United", "Sheffield United": "Sheffield United",
    "Leeds United FC": "Leeds", "Leeds United": "Leeds",
    "Leicester City FC": "Leicester", "Leicester City": "Leicester",
    "AFC Bournemouth": "Bournemouth", "Bournemouth": "Bournemouth",
    "West Ham United FC": "West Ham", "West Ham United": "West Ham",
    "Crystal Palace FC": "Crystal Palace", "Crystal Palace": "Crystal Palace",
    "Aston Villa FC": "Aston Villa", "Aston Villa": "Aston Villa",
    "Arsenal FC": "Arsenal", "Arsenal": "Arsenal",
    "Chelsea FC": "Chelsea", "Chelsea": "Chelsea",
    "Everton FC": "Everton", "Everton": "Everton",
    "Fulham FC": "Fulham", "Fulham": "Fulham",
    "Liverpool FC": "Liverpool", "Liverpool": "Liverpool",
    "Burnley FC": "Burnley", "Burnley": "Burnley",
    "Brentford FC": "Brentford", "Brentford": "Brentford",
    "Norwich City FC": "Norwich", "Norwich City": "Norwich",
    "Watford FC": "Watford", "Watford": "Watford",
    "Ipswich Town FC": "Ipswich", "Ipswich Town": "Ipswich",
    "Southampton FC": "Southampton", "Southampton": "Southampton",
    "Hull City AFC": "Hull", "Hull City FC": "Hull", "Hull City": "Hull",
    "Sunderland AFC": "Sunderland", "Sunderland": "Sunderland",
    "Coventry City FC": "Coventry", "Coventry City": "Coventry",
    "West Bromwich Albion FC": "West Brom", "West Bromwich Albion": "West Brom",
}


def translate_team(api_name):
    return TEAM_NAME_MAP.get(api_name, api_name)


# ---------- 6. Build the UI ----------
st.title("⚽ Soccer Match Predictor")

tab1, tab2 = st.tabs(["🔮 Custom Prediction", "📅 Fixtures by Date"])


# ---------- Tab 1: Custom prediction ----------
with tab1:
    st.write("Pick a home team and an away team to see the model's predicted probabilities.")

    teams = sorted(final_elo.keys())
    default_home = teams.index("Arsenal") if "Arsenal" in teams else 0
    default_away = teams.index("Man City") if "Man City" in teams else 1
    home_team = st.selectbox("Home team", teams, index=default_home)
    away_team = st.selectbox("Away team", teams, index=default_away)

    if home_team == away_team:
        st.warning("Home and away teams must be different.")
        st.stop()

    elo_diff = final_elo[home_team] - final_elo[away_team]
    form_diff = final_form[home_team] - final_form[away_team]
    rest_diff = final_rest[home_team] - final_rest[away_team]

    X = pd.DataFrame([[elo_diff, form_diff, rest_diff]], columns=feature_cols)
    proba = model.predict_proba(X)[0]

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
        st.write(f"Recent form — {home_team}: {final_form[home_team]:.2f}, {away_team}: {final_form[away_team]:.2f}")
        st.write(f"Elo difference: {elo_diff:+.0f}")
        st.write(f"Form difference: {form_diff:+.2f}")
        st.write(f"Rest-days difference: {rest_diff:+.0f}")


# ---------- Tab 2: Fixtures by date ----------
with tab2:
    st.write("Pick any date to see Premier League fixtures and the model's predictions.")
    st.caption("Past matches show final scores and prediction accuracy. Future matches show predictions only.")
    st.caption("Available seasons: 2020-21 through 2026-27.")

    if "selected_date" not in st.session_state:
        st.session_state.selected_date = datetime(2025, 9, 27).date()

    col_a, col_b = st.columns([2, 1])

    with col_a:
        selected_date = st.date_input(
            "Select a date",
            value=st.session_state.selected_date,
            min_value=datetime(2020, 9, 1).date(),
            max_value=datetime(2027, 5, 31).date(),
            key="date_picker",
        )

    with col_b:
        st.write("")
        st.write("")
        jump_option = st.selectbox(
            "Quick jump to",
            ["(no jump)", "2026-27 season", "2025-26 season", "2024-25 season"],
            key="jump_select",
        )
        if jump_option == "2026-27 season":
            selected_date = datetime(2026, 9, 30).date()
        elif jump_option == "2025-26 season":
            selected_date = datetime(2025, 9, 27).date()
        elif jump_option == "2024-25 season":
            selected_date = datetime(2024, 9, 21).date()

    st.session_state.selected_date = selected_date

    date_str = selected_date.strftime("%Y-%m-%d")
    day_name = selected_date.strftime("%A")
    full_date = selected_date.strftime(f"{day_name}, %d %B %Y")

    st.subheader(f"📅 {full_date}")

    fixtures, error = fetch_fixtures_for_date(date_str)

    if error:
        st.warning(error)
    elif not fixtures:
        st.info(f"No Premier League fixtures found on {full_date}.")
    else:
        st.caption(f"{len(fixtures)} match(es) on this date.")

        correct_count = 0
        scored_count = 0

        for match in fixtures:
            home_raw = match["teams"]["home"]["name"]
            away_raw = match["teams"]["away"]["name"]
            status_short = match["fixture"]["status"]["short"]
            status_long = match["fixture"]["status"]["long"]
            home_goals = match["goals"]["home"]
            away_goals = match["goals"]["away"]

            home_key = translate_team(home_raw)
            away_key = translate_team(away_raw)

            if home_key in final_elo and away_key in final_elo:
                elo_diff = final_elo[home_key] - final_elo[away_key]
                form_diff = final_form[home_key] - final_form[away_key]
                rest_diff = final_rest[home_key] - final_rest[away_key]

                X = pd.DataFrame([[elo_diff, form_diff, rest_diff]], columns=feature_cols)
                proba = model.predict_proba(X)[0]
                pred_idx = int(np.argmax(proba))
                pred_label = ["Home win", "Draw", "Away win"][pred_idx]

                st.markdown(f"**{home_raw} vs {away_raw}** — {status_long}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Home", f"{proba[0]*100:.1f}%")
                c2.metric("Draw", f"{proba[1]*100:.1f}%")
                c3.metric("Away", f"{proba[2]*100:.1f}%")

                if status_short in ("FT", "AET", "PEN") and home_goals is not None and away_goals is not None:
                    if home_goals > away_goals:
                        actual_label = "Home win"
                    elif home_goals == away_goals:
                        actual_label = "Draw"
                    else:
                        actual_label = "Away win"
                    correct = (pred_label == actual_label)
                    if correct:
                        correct_count += 1
                    scored_count += 1
                    badge = "✅ Correct" if correct else "❌ Incorrect"
                    st.caption(f"Final score: {home_goals}–{away_goals} — {actual_label}. "
                               f"Prediction: {pred_label}. {badge}")

                st.divider()
            else:
                st.markdown(f"**{home_raw} vs {away_raw}** — {status_long}")
                missing = []
                if home_key not in final_elo:
                    missing.append(home_raw)
                if away_key not in final_elo:
                    missing.append(away_raw)
                teams_str = ", ".join(missing)
                st.caption(
                    f"Prediction unavailable — {teams_str} not in training data. "
                    f"This team was promoted after 2026 and the model has never seen it play."
                )

        if scored_count > 0:
            st.info(f"Model accuracy on {scored_count} finished match(es): "
                    f"{correct_count}/{scored_count} ({100*correct_count/scored_count:.0f}%)")


st.caption("⚠️ Predictions use each team's Elo and form as of the last match in the dataset "
           "(May 2026). Not intended for betting advice.")

st.caption("Model and code: [github.com/CharlieTC-blake/soccer_prediction]"
           "(https://github.com/CharlieTC-blake/soccer_prediction)")