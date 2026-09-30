"""
Streamlit app for the soccer prediction model.

Tabs:
  1. Next Fixtures — upcoming fixtures with expandable predictions
  2. Fixtures by Date — historical and upcoming fixtures with accuracy
  3. Match Analyser — model-driven recommendations with UGX stake and EV
"""
import os
import pandas as pd
import numpy as np
import glob
import requests
import streamlit as st
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta

from soccer_pipeline import (
    compute_features, compute_sample_weights, rest_quality,
    FEATURE_COLS, SPLIT_DATE,
)
from odds_utils import (
    remove_overround_proportional, remove_overround_shin, compute_ev,
)
from match_analyser import (
    generate_recommendations, compute_ev_ugx, fair_odds,
    DEFAULT_STAKE_UGX,
)

st.set_page_config(page_title="Soccer Predictor", page_icon="⚽", layout="wide")


# ---------- Load ----------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    frames = []
    for f in sorted(glob.glob(os.path.join(data_dir, 'season-*.csv'))):
        frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], format='ISO8601')
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.dropna(subset=['FTHG', 'FTAG', 'FTR'])
    return df


@st.cache_data
def compute_features_cached(matches_df):
    return compute_features(matches_df)


@st.cache_data
def train_model(matches_df):
    train = matches_df[matches_df['Date'] < SPLIT_DATE].copy()
    weights = compute_sample_weights(train['Date'], reference_date=SPLIT_DATE)
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(train[FEATURE_COLS], train['target'], sample_weight=weights)
    return model


matches = load_data()
matches, final_state = compute_features_cached(matches)
model = train_model(matches)


# ---------- Team names ----------
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


# ---------- Fixtures ----------
@st.cache_data(ttl=3600)
def fetch_fixtures_for_date(date_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    season_year = date_obj.year if date_obj.month >= 8 else date_obj.year - 1
    season_str = f"{season_year}-{str(season_year + 1)[2:]}"
    url = (f"https://raw.githubusercontent.com/openfootball/football.json/"
           f"master/{season_str}/en.1.json")
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None, f"No fixtures found for season {season_str}."
        data = r.json()
        day_matches = [m for m in data.get("matches", [])
                       if m.get("date") == date_str]
        if not day_matches:
            return None, f"No Premier League fixtures on {date_str}."
        formatted = []
        for m in day_matches:
            score = m.get("score")
            home_goals = away_goals = None
            if isinstance(score, dict):
                ft = score.get("ft")
                if isinstance(ft, list) and len(ft) == 2:
                    home_goals, away_goals = ft[0], ft[1]
            elif isinstance(score, list) and len(score) == 2:
                home_goals, away_goals = score[0], score[1]
            status_short = "FT" if home_goals is not None else "NS"
            status_long = "FINISHED" if home_goals is not None else "SCHEDULED"
            formatted.append({
                "teams": {"home": {"name": m.get("team1", "")},
                          "away": {"name": m.get("team2", "")}},
                "fixture": {"status": {"short": status_short, "long": status_long}},
                "goals": {"home": home_goals, "away": away_goals},
                "time": m.get("time", ""),
                "date": m.get("date", ""),
            })
        return formatted, None
    except Exception as e:
        return None, f"Error fetching fixtures: {e}"


@st.cache_data(ttl=3600)
def fetch_upcoming_fixtures(days_ahead=21):
    today = datetime.now().date()
    horizon = today + timedelta(days=days_ahead)
    upcoming = []
    seasons = set()
    for d in [today, horizon]:
        seasons.add(f"{d.year}-{str(d.year + 1)[2:]}" if d.month >= 8
                    else f"{d.year - 1}-{str(d.year)[2:]}")
    for season in seasons:
        url = (f"https://raw.githubusercontent.com/openfootball/football.json/"
               f"master/{season}/en.1.json")
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                continue
            data = r.json()
        except Exception:
            continue
        for m in data.get("matches", []):
            try:
                mdate = datetime.strptime(m["date"], "%Y-%m-%d").date()
            except Exception:
                continue
            if today <= mdate <= horizon:
                upcoming.append({
                    "date": mdate,
                    "home": m.get("team1", ""),
                    "away": m.get("team2", ""),
                    "time": m.get("time", ""),
                })
    upcoming.sort(key=lambda x: (x["date"], x.get("time", "")))
    return upcoming[:10]


# ---------- Prediction helpers ----------
def build_feature_dict(home_key, away_key, match_date=None):
    if match_date is None:
        match_date = datetime.now().date()
    elo_h = final_state['elo'].get(home_key, 1500)
    elo_a = final_state['elo'].get(away_key, 1500)
    form_h = final_state['form'].get(home_key, 1.0)
    form_a = final_state['form'].get(away_key, 1.0)
    hgs, hgc = final_state['goals'].get(home_key, (1.0, 1.0))
    ags, agc = final_state['goals'].get(away_key, (1.0, 1.0))
    lp_h = final_state['last_played'].get(home_key)
    lp_a = final_state['last_played'].get(away_key)
    h_rest = (match_date - lp_h.date()).days if lp_h is not None else None
    a_rest = (match_date - lp_a.date()).days if lp_a is not None else None
    h2h = final_state['h2h'].get((home_key, away_key), 1.0)
    return {
        'elo_diff': elo_h - elo_a,
        'form_diff': form_h - form_a,
        'rest_quality_diff': rest_quality(h_rest) - rest_quality(a_rest),
        'goals_scored_diff': hgs - ags,
        'goals_conceded_diff': hgc - agc,
        'h2h_home_points_L3': h2h,
        'xg_created_diff': 0.0,
        'xg_conceded_diff': 0.0,
    }


def predict(home_key, away_key, match_date):
    """Returns (p_home, p_draw, p_away) or None if team unknown."""
    if home_key not in final_state['elo'] or away_key not in final_state['elo']:
        return None
    feats = build_feature_dict(home_key, away_key, match_date)
    X = pd.DataFrame([feats], columns=FEATURE_COLS)
    proba = model.predict_proba(X)[0]
    return float(proba[0]), float(proba[1]), float(proba[2])


def get_odds_from_data(home_key, away_key, match_date):
    """Look up historical Bet365 odds for a specific match."""
    mask = ((matches['Date'].dt.date == match_date) &
            (matches['HomeTeam'] == home_key) &
            (matches['AwayTeam'] == away_key))
    row = matches[mask]
    if len(row) == 0:
        return None, None, None
    r = row.iloc[0]
    oh = r.get('B365H')
    od = r.get('B365D')
    oa = r.get('B365A')
    if pd.isna(oh) or pd.isna(od) or pd.isna(oa):
        return None, None, None
    return float(oh), float(od), float(oa)


# ---------- UI ----------
st.title("⚽ Soccer Match Predictor")
tab1, tab2, tab3 = st.tabs(["📅 Next Fixtures", "🗓️ Fixtures by Date", "📊 Match Analyser"])


# ============ Tab 1: Next Fixtures ============
with tab1:
    st.write("Upcoming Premier League fixtures. Expand any match to see the "
             "model's prediction and calculate Expected Value.")

    fixtures = fetch_upcoming_fixtures(days_ahead=21)

    if not fixtures:
        st.info("No upcoming fixtures found in the next 3 weeks.")
    else:
        for fx in fixtures:
            date_str = fx["date"].strftime("%A, %d %B %Y")
            label = (f"**{fx['home']}** vs **{fx['away']}** — "
                     f"{date_str} {fx.get('time', '')}")
            with st.expander(label, expanded=False):
                home_key = translate_team(fx["home"])
                away_key = translate_team(fx["away"])

                result = predict(home_key, away_key, fx["date"])
                if result is None:
                    st.warning(f"Prediction unavailable — one or both teams "
                               f"not in training data ({home_key}, {away_key}).")
                    continue

                p_home, p_draw, p_away = result

                c1, c2, c3 = st.columns(3)
                c1.metric("Home win", f"{p_home*100:.1f}%")
                c2.metric("Draw", f"{p_draw*100:.1f}%")
                c3.metric("Away win", f"{p_away*100:.1f}%")

                with st.expander("Features used by the model"):
                    feats = build_feature_dict(home_key, away_key, fx["date"])
                    st.dataframe(pd.DataFrame([feats]).T.rename(columns={0: "value"}))

                st.markdown("**Enter bookmaker's odds to calculate EV**")
                o1, o2, o3 = st.columns(3)
                key_suffix = f"{fx['date']}_{fx['home']}_{fx['away']}"
                with o1:
                    oh = st.number_input("Home odds", min_value=1.01, value=2.00,
                                          step=0.01, key=f"n_oh_{key_suffix}")
                with o2:
                    od = st.number_input("Draw odds", min_value=1.01, value=3.40,
                                          step=0.01, key=f"n_od_{key_suffix}")
                with o3:
                    oa = st.number_input("Away odds", min_value=1.01, value=3.50,
                                          step=0.01, key=f"n_oa_{key_suffix}")

                odds = [oh, od, oa]
                try:
                    book_probs, overround = remove_overround_proportional(odds)
                except ValueError as e:
                    st.error(f"Invalid odds: {e}")
                    continue

                st.caption(f"Bookmaker overround: {overround*100:.2f}%")
                st.caption(f"Fair probabilities — "
                           f"Home {book_probs[0]*100:.1f}% | "
                           f"Draw {book_probs[1]*100:.1f}% | "
                           f"Away {book_probs[2]*100:.1f}%")

                evs = compute_ev([p_home, p_draw, p_away], odds)
                st.markdown("**Expected Value per £1 stake**")
                ec1, ec2, ec3 = st.columns(3)
                for col, name, ev in zip([ec1, ec2, ec3],
                                          ["Home", "Draw", "Away"], evs):
                    colour = "🟢" if ev > 0 else "🔴"
                    col.metric(f"{name} EV", f"{ev:+.3f}",
                               help=f"{colour} {'Positive' if ev > 0 else 'Negative'} EV")


# ============ Tab 2: Fixtures by Date ============
with tab2:
    st.write("Pick any date to see Premier League fixtures and predictions.")
    st.caption("Past matches show final scores and prediction accuracy.")
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
            key="date_picker_t2",
        )
    with col_b:
        st.write("")
        st.write("")
        jump_option = st.selectbox(
            "Quick jump to",
            ["(no jump)", "2026-27 season", "2025-26 season", "2024-25 season"],
            key="jump_select_t2",
        )
        if jump_option == "2026-27 season":
            selected_date = datetime(2026, 9, 30).date()
        elif jump_option == "2025-26 season":
            selected_date = datetime(2025, 9, 27).date()
        elif jump_option == "2024-25 season":
            selected_date = datetime(2024, 9, 21).date()

    st.session_state.selected_date = selected_date
    date_str = selected_date.strftime("%Y-%m-%d")
    full_date = selected_date.strftime(f"{selected_date.strftime('%A')}, %d %B %Y")
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

            result = predict(home_key, away_key, selected_date)
            if result is not None:
                p_home, p_draw, p_away = result
                proba = [p_home, p_draw, p_away]
                pred_idx = int(np.argmax(proba))
                pred_label = ["Home win", "Draw", "Away win"][pred_idx]

                st.markdown(f"**{home_raw} vs {away_raw}** — {status_long}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Home", f"{p_home*100:.1f}%")
                c2.metric("Draw", f"{p_draw*100:.1f}%")
                c3.metric("Away", f"{p_away*100:.1f}%")

                if (status_short in ("FT", "AET", "PEN")
                        and home_goals is not None and away_goals is not None):
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
                    st.caption(f"Final score: {home_goals}–{away_goals} — "
                               f"{actual_label}. Prediction: {pred_label}. {badge}")
                st.divider()
            else:
                st.markdown(f"**{home_raw} vs {away_raw}** — {status_long}")
                missing = [t for k, t in [(home_key, home_raw), (away_key, away_raw)]
                           if k not in final_state['elo']]
                st.caption(f"Prediction unavailable — {', '.join(missing)} "
                           f"not in training data.")

        if scored_count > 0:
            st.info(f"Model accuracy on {scored_count} finished match(es): "
                    f"{correct_count}/{scored_count} "
                    f"({100*correct_count/scored_count:.0f}%)")


# ============ Tab 3: Match Analyser ============
with tab3:
    st.write("Model-driven recommendations for each fixture. "
             "Enter your bookmaker's odds to see Expected Value.")
    st.caption("Recommendations are derived from the model, not the bookmaker. "
               "Not financial advice.")

    # Global stake input
    stake_col, _ = st.columns([1, 3])
    with stake_col:
        stake_ugx = st.number_input(
            "Stake per bet (UGX)",
            min_value=100, max_value=10_000_000,
            value=DEFAULT_STAKE_UGX, step=100,
            key="analyser_stake",
        )

    # Date picker
    if "analyser_date" not in st.session_state:
        st.session_state.analyser_date = datetime(2025, 9, 27).date()

    analyser_date = st.date_input(
        "Select a date",
        value=st.session_state.analyser_date,
        min_value=datetime(2020, 9, 1).date(),
        max_value=datetime(2027, 5, 31).date(),
        key="analyser_date_picker",
    )
    st.session_state.analyser_date = analyser_date

    date_str = analyser_date.strftime("%Y-%m-%d")
    full_date = analyser_date.strftime(f"{analyser_date.strftime('%A')}, %d %B %Y")
    st.subheader(f"📅 {full_date}")

    fixtures, error = fetch_fixtures_for_date(date_str)
    if error:
        st.warning(error)
    elif not fixtures:
        st.info(f"No Premier League fixtures found on {full_date}.")
    else:
        st.caption(f"{len(fixtures)} match(es) on this date. "
                   f"Expand any match to see recommendations.")

        for match in fixtures:
            home_raw = match["teams"]["home"]["name"]
            away_raw = match["teams"]["away"]["name"]
            status_short = match["fixture"]["status"]["short"]
            status_long = match["fixture"]["status"]["long"]
            home_goals = match["goals"]["home"]
            away_goals = match["goals"]["away"]

            home_key = translate_team(home_raw)
            away_key = translate_team(away_raw)

            score_str = ""
            if home_goals is not None and away_goals is not None:
                score_str = f" — Final: {home_goals}–{away_goals}"

            label = f"**{home_raw}** vs **{away_raw}** — {status_long}{score_str}"

            with st.expander(label, expanded=False):
                result = predict(home_key, away_key, analyser_date)
                if result is None:
                    st.warning(f"Prediction unavailable — one or both teams "
                               f"not in training data ({home_key}, {away_key}).")
                    continue

                p_home, p_draw, p_away = result

                # Try to find historical odds
                hist_oh, hist_od, hist_oa = get_odds_from_data(
                    home_key, away_key, analyser_date
                )

                st.markdown("**Model probabilities**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Home win", f"{p_home*100:.1f}%")
                c2.metric("Draw", f"{p_draw*100:.1f}%")
                c3.metric("Away win", f"{p_away*100:.1f}%")

                st.markdown("**Bookmaker odds** *(edit to try different prices)*")
                key_suffix = f"ma_{date_str}_{home_key}_{away_key}"
                o1, o2, o3 = st.columns(3)
                with o1:
                    oh = st.number_input(
                        "Home odds", min_value=1.01,
                        value=float(hist_oh) if hist_oh is not None else 2.00,
                        step=0.01, key=f"ma_oh_{key_suffix}",
                    )
                with o2:
                    od = st.number_input(
                        "Draw odds", min_value=1.01,
                        value=float(hist_od) if hist_od is not None else 3.40,
                        step=0.01, key=f"ma_od_{key_suffix}",
                    )
                with o3:
                    oa = st.number_input(
                        "Away odds", min_value=1.01,
                        value=float(hist_oa) if hist_oa is not None else 3.50,
                        step=0.01, key=f"ma_oa_{key_suffix}",
                    )

                if hist_oh is None:
                    st.caption("⚠️ No historical odds for this match — "
                               "enter your bookmaker's current prices above.")

                # Generate recommendations
                recs = generate_recommendations(
                    home_raw, away_raw, p_home, p_draw, p_away,
                    odds_home=oh, odds_draw=od, odds_away=oa,
                    stake_ugx=stake_ugx,
                )

                st.markdown(f"**Recommendations for a stake of UGX {stake_ugx:,}**")

                # Top pick
                tp = recs['top_pick']
                st.markdown(f"**🥇 Top pick: {tp['label']}**")
                st.write(f"Model probability: **{tp['prob']*100:.1f}%**")
                st.write(f"Fair odds (break-even): **{tp['fair_odds']:.2f}**")
                if tp['ev_ugx'] is not None:
                    ev_colour = "🟢" if tp['ev_ugx'] > 0 else "🔴"
                    st.write(f"{ev_colour} Expected Value: **UGX {tp['ev_ugx']:+,.0f}**")
                    if tp['ev_ugx'] < 0:
                        st.caption("Negative EV means the odds are worse than "
                                   "the model's probability suggests.")
                st.divider()

                # Top double chance
                tdc = recs['top_double_chance']
                st.markdown(f"**🥈 Top double chance: {tdc['label']}**")
                st.write(f"Combined probability: **{tdc['prob']*100:.1f}%**")
                st.write(f"Fair odds (break-even): **{tdc['fair_odds']:.2f}**")
                st.divider()

                # Avoid
                av = recs['avoid']
                st.markdown(f"**🚫 Avoid: {av['label']}** "
                            f"({av['prob']*100:.1f}% model probability)")

                st.caption("⚠️ Recommendations are model-derived. "
                           "Not financial advice. Bet only what you can afford to lose.")


st.caption("⚠️ Predictions use each team's Elo and form as of the last match "
           "in the dataset. Not intended for betting advice.")
st.caption("Model and code: [github.com/CharlieTC-blake/soccer_prediction]"
           "(https://github.com/CharlieTC-blake/soccer_prediction)")
