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
import joblib
from datetime import datetime

st.set_page_config(page_title="Soccer Predictor", page_icon="⚽", layout="wide")

# 1. Load Data
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

    matches_df = matches_df.copy()
    matches_df['elo_diff'] = np.array(pre_match_home_elo) - np.array(pre_match_away_elo)
    matches_df['form_diff'] = np.array(home_form) - np.array(away_form)
    matches_df['rest_diff'] = np.array(home_rest) - np.array(away_rest)
    
    # Store final state for future predictions
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

model, feature_cols = load_model()

# 4. Fetch Fixtures
@st.cache_data(ttl=300)
@st.cache_data(ttl=300)
@st.cache_data(ttl=3600)  # cache for 1 hour
@st.cache_data(ttl=3600)
def fetch_fixtures_for_date(date_str):
    """Fetch Premier League fixtures for a specific date from openfootball."""
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

        # Filter to the requested date
        day_matches = [m for m in all_matches if m.get("date") == date_str]

        if not day_matches:
            return None, f"No Premier League fixtures on {date_str}."

        # Convert to the format the rest of the app expects
        formatted = []
        for m in day_matches:
            home_team = m.get("team1", "")
            away_team = m.get("team2", "")

            # Score can be a dict ({"ft": [2,1]}) or a flat list ([2,1])
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

TEAM_NAME_MAP = {
    # Man City
    "Manchester City": "Man City", "Manchester City FC": "Man City",
    # Man United
    "Manchester United": "Man United", "Manchester United FC": "Man United",
    # Newcastle
    "Newcastle United": "Newcastle", "Newcastle United FC": "Newcastle",
    # Wolves
    "Wolverhampton Wanderers": "Wolves", "Wolverhampton Wanderers FC": "Wolves",
    # Nottingham Forest
    "Nottingham Forest": "Nott'm Forest", "Nottingham Forest FC": "Nott'm Forest",
    # Tottenham
    "Tottenham Hotspur": "Tottenham", "Tottenham Hotspur FC": "Tottenham",
    # Brighton
    "Brighton & Hove Albion": "Brighton", "Brighton & Hove Albion FC": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    # Sheffield United
    "Sheffield United": "Sheffield United", "Sheffield United FC": "Sheffield United",
    # Leeds
    "Leeds United": "Leeds", "Leeds United FC": "Leeds",
    # Leicester
    "Leicester City": "Leicester", "Leicester City FC": "Leicester",
    # Bournemouth
    "AFC Bournemouth": "Bournemouth", "Bournemouth": "Bournemouth",
    # West Ham
    "West Ham United": "West Ham", "West Ham United FC": "West Ham",
    # Crystal Palace
    "Crystal Palace": "Crystal Palace", "Crystal Palace FC": "Crystal Palace",
    # Aston Villa
    "Aston Villa": "Aston Villa", "Aston Villa FC": "Aston Villa",
    # Arsenal
    "Arsenal": "Arsenal", "Arsenal FC": "Arsenal",
    # Chelsea
    "Chelsea": "Chelsea", "Chelsea FC": "Chelsea",
    # Everton
    "Everton": "Everton", "Everton FC": "Everton",
    # Fulham
    "Fulham": "Fulham", "Fulham FC": "Fulham",
    # Liverpool
    "Liverpool": "Liverpool", "Liverpool FC": "Liverpool",
    # Burnley
    "Burnley": "Burnley", "Burnley FC": "Burnley",
    # Brentford
    "Brentford": "Brentford", "Brentford FC": "Brentford",
    # Norwich
    "Norwich City": "Norwich", "Norwich City FC": "Norwich",
    # Watford
    "Watford": "Watford", "Watford FC": "Watford",
    # Ipswich
    "Ipswich Town": "Ipswich", "Ipswich Town FC": "Ipswich",
    # Southampton
    "Southampton": "Southampton", "Southampton FC": "Southampton",
    # Hull
    "Hull City": "Hull", "Hull City FC": "Hull",
    # Sunderland
    "Sunderland": "Sunderland", "Sunderland AFC": "Sunderland",
    # Coventry
    "Coventry City": "Coventry", "Coventry City FC": "Coventry",
    # Others that might appear
    "West Bromwich Albion": "West Brom", "West Bromwich Albion FC": "West Brom",
    "Queens Park Rangers": "QPR", "Queens Park Rangers FC": "QPR",
    "Cardiff City": "Cardiff", "Cardiff City FC": "Cardiff",
    "Swansea City": "Swansea", "Swansea City FC": "Swansea",
    "Stoke City": "Stoke", "Stoke City FC": "Stoke",
    "Middlesbrough": "Middlesbrough", "Middlesbrough FC": "Middlesbrough",
    "Blackburn Rovers": "Blackburn", "Blackburn Rovers FC": "Blackburn",
    "Preston North End": "Preston", "Preston North End FC": "Preston",
    "Bristol City": "Bristol City", "Bristol City FC": "Bristol City",
    "Millwall": "Millwall", "Millwall FC": "Millwall",
    "Luton Town": "Luton", "Luton Town FC": "Luton",
    "Wigan Athletic": "Wigan", "Wigan Athletic FC": "Wigan",
    "Reading": "Reading", "Reading FC": "Reading",
    "Derby County": "Derby", "Derby County FC": "Derby",
    "Sheffield Wednesday": "Sheffield Wed", "Sheffield Wednesday FC": "Sheffield Wed",
    "Charlton Athletic": "Charlton", "Charlton Athletic FC": "Charlton",
    "Wimbledon": "Wimbledon", "Wimbledon FC": "Wimbledon",
    "MK Dons": "MK Dons", "Milton Keynes Dons": "MK Dons",
    "Oldham Athletic": "Oldham", "Oldham Athletic FC": "Oldham",
    "Barnsley": "Barnsley", "Barnsley FC": "Barnsley",
    "Swindon Town": "Swindon", "Swindon Town FC": "Swindon",
    "Oxford United": "Oxford", "Oxford United FC": "Oxford",
    "Cambridge United": "Cambridge", "Cambridge United FC": "Cambridge",
    "Peterborough United": "Peterborough", "Peterborough United FC": "Peterborough",
    "Rotherham United": "Rotherham", "Rotherham United FC": "Rotherham",
    "Doncaster Rovers": "Doncaster", "Doncaster Rovers FC": "Doncaster",
    "Crewe Alexandra": "Crewe", "Crewe Alexandra FC": "Crewe",
    "Port Vale": "Port Vale", "Port Vale FC": "Port Vale",
    "Bury": "Bury", "Bury FC": "Bury",
    "Bolton Wanderers": "Bolton", "Bolton Wanderers FC": "Bolton",
    "Blackpool": "Blackpool", "Blackpool FC": "Blackpool",
    "Fleetwood Town": "Fleetwood", "Fleetwood Town FC": "Fleetwood",
    "Accrington Stanley": "Accrington", "Accrington Stanley FC": "Accrington",
    "Burton Albion": "Burton", "Burton Albion FC": "Burton",
    "Wycombe Wanderers": "Wycombe", "Wycombe Wanderers FC": "Wycombe",
    "Gillingham": "Gillingham", "Gillingham FC": "Gillingham",
    "Southend United": "Southend", "Southend United FC": "Southend",
    "Colchester United": "Colchester", "Colchester United FC": "Colchester",
    "Leyton Orient": "Leyton Orient", "Leyton Orient FC": "Leyton Orient",
    "Northampton Town": "Northampton", "Northampton Town FC": "Northampton",
    "Stevenage": "Stevenage", "Stevenage FC": "Stevenage",
    "Cheltenham Town": "Cheltenham", "Cheltenham Town FC": "Cheltenham",
    "Forest Green Rovers": "Forest Green", "Forest Green Rovers FC": "Forest Green",
    "Salford City": "Salford", "Salford City FC": "Salford",
    "Mansfield Town": "Mansfield", "Mansfield Town FC": "Mansfield",
    "Walsall": "Walsall", "Walsall FC": "Walsall",
    "Tranmere Rovers": "Tranmere", "Tranmere Rovers FC": "Tranmere",
    "Notts County": "Notts County", "Notts County FC": "Notts County",
    "Chesterfield": "Chesterfield", "Chesterfield FC": "Chesterfield",
    "Wrexham": "Wrexham", "Wrexham AFC": "Wrexham",
    "Stockport County": "Stockport", "Stockport County FC": "Stockport",
    "Grimsby Town": "Grimsby", "Grimsby Town FC": "Grimsby",
    "Hartlepool United": "Hartlepool", "Hartlepool United FC": "Hartlepool",
    "Barrow": "Barrow", "Barrow AFC": "Barrow",
    "Sutton United": "Sutton", "Sutton United FC": "Sutton",
    "Newport County": "Newport", "Newport County AFC": "Newport",
    "Exeter City": "Exeter", "Exeter City FC": "Exeter",
    "Bristol Rovers": "Bristol Rovers", "Bristol Rovers FC": "Bristol Rovers",
    "Plymouth Argyle": "Plymouth", "Plymouth Argyle FC": "Plymouth",
    "Torquay United": "Torquay", "Torquay United FC": "Torquay",
    "Yeovil Town": "Yeovil", "Yeovil Town FC": "Yeovil",
    "Wealdstone": "Wealdstone", "Wealdstone FC": "Wealdstone",
    "Barnet": "Barnet", "Barnet FC": "Barnet",
    "Boreham Wood": "Boreham Wood", "Boreham Wood FC": "Boreham Wood",
    "Halifax Town": "Halifax", "FC Halifax Town": "Halifax",
    "Altrincham": "Altrincham", "Altrincham FC": "Altrincham",
    "Solihull Moors": "Solihull Moors", "Solihull Moors FC": "Solihull Moors",
    "Dagenham & Redbridge": "Dag & Red", "Dagenham & Redbridge FC": "Dag & Red",
    "Maidenhead United": "Maidenhead", "Maidenhead United FC": "Maidenhead",
    "Aldershot Town": "Aldershot", "Aldershot Town FC": "Aldershot",
    "Woking": "Woking", "Woking FC": "Woking",
    "Eastleigh": "Eastleigh", "Eastleigh FC": "Eastleigh",
    "Bromley": "Bromley", "Bromley FC": "Bromley",
    "Dover Athletic": "Dover", "Dover Athletic FC": "Dover",
    "Chester": "Chester", "Chester FC": "Chester",
    "Kidderminster Harriers": "Kidderminster", "Kidderminster Harriers FC": "Kidderminster",
    "Hereford": "Hereford", "Hereford FC": "Hereford",
    "York City": "York", "York City FC": "York",
    "Boston United": "Boston", "Boston United FC": "Boston",
    "Kettering Town": "Kettering", "Kettering Town FC": "Kettering",
    "Brackley Town": "Brackley", "Brackley Town FC": "Brackley",
    "Chorley": "Chorley", "Chorley FC": "Chorley",
    "Curzon Ashton": "Curzon Ashton", "Curzon Ashton FC": "Curzon Ashton",
    "Spennymoor Town": "Spennymoor", "Spennymoor Town FC": "Spennymoor",
    "Southport": "Southport", "Southport FC": "Southport",
    "Darlington": "Darlington", "Darlington FC": "Darlington",
    "Blyth Spartans": "Blyth", "Blyth Spartans AFC": "Blyth",
    "Farsley Celtic": "Farsley", "Farsley Celtic FC": "Farsley",
    "Guiseley": "Guiseley", "Guiseley AFC": "Guiseley",
    "Bradford Park Avenue": "Bradford PA", "Bradford (Park Avenue) AFC": "Bradford PA",
    "Leamington": "Leamington", "Leamington FC": "Leamington",
    "Kidderminster": "Kidderminster", "Kidderminster Harriers": "Kidderminster",
    "AFC Telford United": "Telford", "Telford United": "Telford",
    "Alfreton Town": "Alfreton", "Alfreton Town FC": "Alfreton",
    "Buxton": "Buxton", "Buxton FC": "Buxton",
    "Gateshead": "Gateshead", "Gateshead FC": "Gateshead",
    "Ashton United": "Ashton", "Ashton United FC": "Ashton",
    "Mickleover": "Mickleover", "Mickleover FC": "Mickleover",
    "Nantwich Town": "Nantwich", "Nantwich Town FC": "Nantwich",
    "Stafford Rangers": "Stafford", "Stafford Rangers FC": "Stafford",
    "Stalybridge Celtic": "Stalybridge", "Stalybridge Celtic FC": "Stalybridge",
    "Warrington Town": "Warrington", "Warrington Town FC": "Warrington",
    "Whitby Town": "Whitby", "Whitby Town FC": "Whitby",
    "Bamber Bridge": "Bamber Bridge", "Bamber Bridge FC": "Bamber Bridge",
    "FC United of Manchester": "FC United", "United of Manchester": "FC United",
    "Scarborough Athletic": "Scarborough", "Scarborough Athletic FC": "Scarborough",
    "Atherton Collieries": "Atherton", "Atherton Collieries FC": "Atherton",
    "Basford United": "Basford", "Basford United FC": "Basford",
    "Matlock Town": "Matlock", "Matlock Town FC": "Matlock",
    "Hyde United": "Hyde", "Hyde United FC": "Hyde",
    "Witton Albion": "Witton", "Witton Albion FC": "Witton",
    "Radcliffe": "Radcliffe", "Radcliffe FC": "Radcliffe",
    "Morpeth Town": "Morpeth", "Morpeth Town FC": "Morpeth",
    "Marske United": "Marske", "Marske United FC": "Marske",
    "Liversedge": "Liversedge", "Liversedge FC": "Liversedge",
    "Bridlington Town": "Bridlington", "Bridlington Town FC": "Bridlington",
    "Ossett United": "Ossett", "Ossett United FC": "Ossett",
    "Pontefract Collieries": "Pontefract", "Pontefract Collieries FC": "Pontefract",
    "Consett": "Consett", "Consett AFC": "Consett",
    "Hebburn Town": "Hebburn", "Hebburn Town FC": "Hebburn",
    "Newton Aycliffe": "Newton Aycliffe", "Newton Aycliffe FC": "Newton Aycliffe",
    "North Shields": "North Shields", "North Shields FC": "North Shields",
    "Whickham": "Whickham", "Whickham FC": "Whickham",
    "West Auckland Town": "West Auckland", "West Auckland Town FC": "West Auckland",
    "Shildon": "Shildon", "Shildon AFC": "Shildon",
    "Bishop Auckland": "Bishop Auckland", "Bishop Auckland FC": "Bishop Auckland",
    "Crook Town": "Crook Town", "Crook Town AFC": "Crook Town",
    "Penrith": "Penrith", "Penrith AFC": "Penrith",
    "Carlisle City": "Carlisle City", "Carlisle City FC": "Carlisle City",
    "Sunderland RCA": "Sunderland RCA", "Sunderland RCA FC": "Sunderland RCA",
    "Ryhope CW": "Ryhope", "Ryhope Colliery Welfare FC": "Ryhope",
    "Seaham Red Star": "Seaham", "Seaham Red Star FC": "Seaham",
    "Easington Colliery": "Easington", "Easington Colliery AFC": "Easington",
    "Tow Law Town": "Tow Law", "Tow Law Town AFC": "Tow Law",
    "Willington": "Willington", "Willington AFC": "Willington",
    "Chester-le-Street Town": "Chester-le-Street", "Chester-le-Street Town FC": "Chester-le-Street",
    "Birtley Town": "Birtley", "Birtley Town FC": "Birtley",
    "Jarrow": "Jarrow", "Jarrow FC": "Jarrow",
    "Boldon CA": "Boldon", "Boldon Community Association FC": "Boldon",
    "Washington": "Washington", "Washington FC": "Washington",
    "Horden CW": "Horden", "Horden Community Welfare FC": "Horden",
    "Esh Winning": "Esh Winning", "Esh Winning FC": "Esh Winning",
    "Brandon United": "Brandon", "Brandon United FC": "Brandon",
    "Durham City": "Durham City", "Durham City AFC": "Durham City",
    "Willington": "Willington", "Willington AFC": "Willington",
    "Crook Town": "Crook Town", "Crook Town AFC": "Crook Town",
    "Bishop Auckland": "Bishop Auckland", "Bishop Auckland FC": "Bishop Auckland",
    "Shildon": "Shildon", "Shildon AFC": "Shildon",
    "West Auckland Town": "West Auckland", "West Auckland Town FC": "West Auckland",
    "Newton Aycliffe": "Newton Aycliffe", "Newton Aycliffe FC": "Newton Aycliffe",
    "North Shields": "North Shields", "North Shields FC": "North Shields",
    "Whickham": "Whickham", "Whickham FC": "Whickham",
    "Consett": "Consett", "Consett AFC": "Consett",
    "Hebburn Town": "Hebburn", "Hebburn Town FC": "Hebburn",
    "Ossett United": "Ossett", "Ossett United FC": "Ossett",
    "Pontefract Collieries": "Pontefract", "Pontefract Collieries FC": "Pontefract",
    "Liversedge": "Liversedge", "Liversedge FC": "Liversedge",
    "Bridlington Town": "Bridlington", "Bridlington Town FC": "Bridlington",
    "Marske United": "Marske", "Marske United FC": "Marske",
    "Morpeth Town": "Morpeth", "Morpeth Town FC": "Morpeth",
    "Radcliffe": "Radcliffe", "Radcliffe FC": "Radcliffe",
    "Hyde United": "Hyde", "Hyde United FC": "Hyde",
    "Witton Albion": "Witton", "Witton Albion FC": "Witton",
    "Matlock Town": "Matlock", "Matlock Town FC": "Matlock",
    "Basford United": "Basford", "Basford United FC": "Basford",
    "Atherton Collieries": "Atherton", "Atherton Collieries FC": "Atherton",
    "Scarborough Athletic": "Scarborough", "Scarborough Athletic FC": "Scarborough",
    "FC United of Manchester": "FC United", "United of Manchester": "FC United",
    "Bamber Bridge": "Bamber Bridge", "Bamber Bridge FC": "Bamber Bridge",
    "Whitby Town": "Whitby", "Whitby Town FC": "Whitby",
    "Warrington Town": "Warrington", "Warrington Town FC": "Warrington",
    "Stalybridge Celtic": "Stalybridge", "Stalybridge Celtic FC": "Stalybridge",
    "Stafford Rangers": "Stafford", "Stafford Rangers FC": "Stafford",
    "Nantwich Town": "Nantwich", "Nantwich Town FC": "Nantwich",
    "Mickleover": "Mickleover", "Mickleover FC": "Mickleover",
    "Ashton United": "Ashton", "Ashton United FC": "Ashton",
    "Gateshead": "Gateshead", "Gateshead FC": "Gateshead",
    "Buxton": "Buxton", "Buxton FC": "Buxton",
    "Alfreton Town": "Alfreton", "Alfreton Town FC": "Alfreton",
    "AFC Telford United": "Telford", "Telford United": "Telford",
    "Kidderminster Harriers": "Kidderminster", "Kidderminster Harriers FC": "Kidderminster",
    "Leamington": "Leamington", "Leamington FC": "Leamington",
    "Bradford Park Avenue": "Bradford PA", "Bradford (Park Avenue) AFC": "Bradford PA",
    "Guiseley": "Guiseley", "Guiseley AFC": "Guiseley",
    "Farsley Celtic": "Farsley", "Farsley Celtic FC": "Farsley",
    "Blyth Spartans": "Blyth", "Blyth Spartans AFC": "Blyth",
    "Darlington": "Darlington", "Darlington FC": "Darlington",
    "Southport": "Southport", "Southport FC": "Southport",
    "Spennymoor Town": "Spennymoor", "Spennymoor Town FC": "Spennymoor",
    "Curzon Ashton": "Curzon Ashton", "Curzon Ashton FC": "Curzon Ashton",
    "Chorley": "Chorley", "Chorley FC": "Chorley",
    "Brackley Town": "Brackley", "Brackley Town FC": "Brackley",
    "Kettering Town": "Kettering", "Kettering Town FC": "Kettering",
    "Boston United": "Boston", "Boston United FC": "Boston",
    "York City": "York", "York City FC": "York",
    "Hereford": "Hereford", "Hereford FC": "Hereford",
    "Kidderminster": "Kidderminster", "Kidderminster Harriers": "Kidderminster",
    "Chester": "Chester", "Chester FC": "Chester",
    "Dover Athletic": "Dover", "Dover Athletic FC": "Dover",
    "Bromley": "Bromley", "Bromley FC": "Bromley",
    "Eastleigh": "Eastleigh", "Eastleigh FC": "Eastleigh",
    "Woking": "Woking", "Woking FC": "Woking",
    "Aldershot Town": "Aldershot", "Aldershot Town FC": "Aldershot",
    "Maidenhead United": "Maidenhead", "Maidenhead United FC": "Maidenhead",
    "Dagenham & Redbridge": "Dag & Red", "Dagenham & Redbridge FC": "Dag & Red",
    "Solihull Moors": "Solihull Moors", "Solihull Moors FC": "Solihull Moors",
    "Altrincham": "Altrincham", "Altrincham FC": "Altrincham",
    "FC Halifax Town": "Halifax", "Halifax Town": "Halifax",
    "Boreham Wood": "Boreham Wood", "Boreham Wood FC": "Boreham Wood",
    "Barnet": "Barnet", "Barnet FC": "Barnet",
    "Wealdstone": "Wealdstone", "Wealdstone FC": "Wealdstone",
    "Yeovil Town": "Yeovil", "Yeovil Town FC": "Yeovil",
    "Torquay United": "Torquay", "Torquay United FC": "Torquay",
    "Plymouth Argyle": "Plymouth", "Plymouth Argyle FC": "Plymouth",
    "Bristol Rovers": "Bristol Rovers", "Bristol Rovers FC": "Bristol Rovers",
    "Exeter City": "Exeter", "Exeter City FC": "Exeter",
    "Newport County": "Newport", "Newport County AFC": "Newport",
    "Sutton United": "Sutton", "Sutton United FC": "Sutton",
    "Barrow": "Barrow", "Barrow AFC": "Barrow",
    "Hartlepool United": "Hartlepool", "Hartlepool United FC": "Hartlepool",
    "Grimsby Town": "Grimsby", "Grimsby Town FC": "Grimsby",
    "Stockport County": "Stockport", "Stockport County FC": "Stockport",
    "Wrexham": "Wrexham", "Wrexham AFC": "Wrexham",
    "Chesterfield": "Chesterfield", "Chesterfield FC": "Chesterfield",
    "Notts County": "Notts County", "Notts County FC": "Notts County",
    "Tranmere Rovers": "Tranmere", "Tranmere Rovers FC": "Tranmere",
    "Walsall": "Walsall", "Walsall FC": "Walsall",
    "Mansfield Town": "Mansfield", "Mansfield Town FC": "Mansfield",
    "Salford City": "Salford", "Salford City FC": "Salford",
    "Forest Green Rovers": "Forest Green", "Forest Green Rovers FC": "Forest Green",
    "Cheltenham Town": "Cheltenham", "Cheltenham Town FC": "Cheltenham",
    "Stevenage": "Stevenage", "Stevenage FC": "Stevenage",
    "Northampton Town": "Northampton", "Northampton Town FC": "Northampton",
    "Leyton Orient": "Leyton Orient", "Leyton Orient FC": "Leyton Orient",
    "Colchester United": "Colchester", "Colchester United FC": "Colchester",
    "Southend United": "Southend", "Southend United FC": "Southend",
    "Gillingham": "Gillingham", "Gillingham FC": "Gillingham",
    "Wycombe Wanderers": "Wycombe", "Wycombe Wanderers FC": "Wycombe",
    "Burton Albion": "Burton", "Burton Albion FC": "Burton",
    "Accrington Stanley": "Accrington", "Accrington Stanley FC": "Accrington",
    "Fleetwood Town": "Fleetwood", "Fleetwood Town FC": "Fleetwood",
    "Blackpool": "Blackpool", "Blackpool FC": "Blackpool",
    "Bolton Wanderers": "Bolton", "Bolton Wanderers FC": "Bolton",
    "Bury": "Bury", "Bury FC": "Bury",
    "Port Vale": "Port Vale", "Port Vale FC": "Port Vale",
    "Crewe Alexandra": "Crewe", "Crewe Alexandra FC": "Crewe",
    "Doncaster Rovers": "Doncaster", "Doncaster Rovers FC": "Doncaster",
    "Rotherham United": "Rotherham", "Rotherham United FC": "Rotherham",
    "Peterborough United": "Peterborough", "Peterborough United FC": "Peterborough",
    "Cambridge United": "Cambridge", "Cambridge United FC": "Cambridge",
    "Oxford United": "Oxford", "Oxford United FC": "Oxford",
    "Swindon Town": "Swindon", "Swindon Town FC": "Swindon",
    "Barnsley": "Barnsley", "Barnsley FC": "Barnsley",
    "Oldham Athletic": "Oldham", "Oldham Athletic FC": "Oldham",
    "MK Dons": "MK Dons", "Milton Keynes Dons": "MK Dons",
    "Wimbledon": "Wimbledon", "Wimbledon FC": "Wimbledon",
    "Charlton Athletic": "Charlton", "Charlton Athletic FC": "Charlton",
    "Sheffield Wednesday": "Sheffield Wed", "Sheffield Wednesday FC": "Sheffield Wed",
    "Derby County": "Derby", "Derby County FC": "Derby",
    "Reading": "Reading", "Reading FC": "Reading",
    "Wigan Athletic": "Wigan", "Wigan Athletic FC": "Wigan",
    "Luton Town": "Luton", "Luton Town FC": "Luton",
    "Millwall": "Millwall", "Millwall FC": "Millwall",
    "Bristol City": "Bristol City", "Bristol City FC": "Bristol City",
    "Preston North End": "Preston", "Preston North End FC": "Preston",
    "Blackburn Rovers": "Blackburn", "Blackburn Rovers FC": "Blackburn",
    "Middlesbrough": "Middlesbrough", "Middlesbrough FC": "Middlesbrough",
    "Stoke City": "Stoke", "Stoke City FC": "Stoke",
    "Swansea City": "Swansea", "Swansea City FC": "Swansea",
    "Cardiff City": "Cardiff", "Cardiff City FC": "Cardiff",
    "Queens Park Rangers": "QPR", "Queens Park Rangers FC": "QPR",
    "West Bromwich Albion": "West Brom", "West Bromwich Albion FC": "West Brom",
    "Hull City": "Hull", "Hull City FC": "Hull",
    "Sunderland": "Sunderland", "Sunderland AFC": "Sunderland",
    "Coventry City": "Coventry", "Coventry City FC": "Coventry",
    "Ipswich Town": "Ipswich", "Ipswich Town FC": "Ipswich",
    "Southampton": "Southampton", "Southampton FC": "Southampton",
    "Watford": "Watford", "Watford FC": "Watford",
    "Norwich City": "Norwich", "Norwich City FC": "Norwich",
    "Brentford": "Brentford", "Brentford FC": "Brentford",
    "Burnley": "Burnley", "Burnley FC": "Burnley",
    "Liverpool": "Liverpool", "Liverpool FC": "Liverpool",
    "Fulham": "Fulham", "Fulham FC": "Fulham",
    "Everton": "Everton", "Everton FC": "Everton",
    "Chelsea": "Chelsea", "Chelsea FC": "Chelsea",
    "Arsenal": "Arsenal", "Arsenal FC": "Arsenal",
    "Aston Villa": "Aston Villa", "Aston Villa FC": "Aston Villa",
    "Crystal Palace": "Crystal Palace", "Crystal Palace FC": "Crystal Palace",
    "West Ham United": "West Ham", "West Ham United FC": "West Ham",
    "AFC Bournemouth": "Bournemouth", "Bournemouth": "Bournemouth",
    "Leicester City": "Leicester", "Leicester City FC": "Leicester",
    "Leeds United": "Leeds", "Leeds United FC": "Leeds",
    "Sheffield United": "Sheffield United", "Sheffield United FC": "Sheffield United",
    "Brighton & Hove Albion": "Brighton", "Brighton & Hove Albion FC": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Tottenham Hotspur": "Tottenham", "Tottenham Hotspur FC": "Tottenham",
    "Nottingham Forest": "Nott'm Forest", "Nottingham Forest FC": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves", "Wolverhampton Wanderers FC": "Wolves",
    "Newcastle United": "Newcastle", "Newcastle United FC": "Newcastle",
    "Manchester United": "Man United", "Manchester United FC": "Man United",
    "Manchester City": "Man City", "Manchester City FC": "Man City",
}

def translate_team(api_name):
    return TEAM_NAME_MAP.get(api_name, api_name)

# 5. UI
st.title("⚽ Soccer Match Predictor")
tab1, tab2 = st.tabs(["🔮 Custom Prediction", "📅 Fixtures by Date"])

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
    c1, c2, c3 = st.columns(3)
    c1.metric("Home win", f"{proba[0]*100:.1f}%")
    c2.metric("Draw", f"{proba[1]*100:.1f}%")
    c3.metric("Away win", f"{proba[2]*100:.1f}%")
    
    chart_data = pd.DataFrame({
        'Outcome': ['Home win', 'Draw', 'Away win'],
        'Probability': [proba[0], proba[1], proba[2]],
    })
    st.bar_chart(chart_data, x='Outcome', y='Probability', height=250)
    
    with st.expander("Show features used by the model"):
        st.write(f"**Elo rating** — {home_team}: {final_elo[home_team]:.0f}, {away_team}: {final_elo[away_team]:.0f}")
        st.write(f"**Recent form** — {home_team}: {final_form[home_team]:.2f}, {away_team}: {final_form[away_team]:.2f}")
        st.write(f"**Elo difference**: {elo_diff:+.0f}")
        st.write(f"**Form difference**: {form_diff:+.2f}")
        st.write(f"**Rest-days difference**: {rest_diff:+.0f}")

with tab2:
    st.write("Pick any date to see Premier League fixtures and the model's predictions.")
    st.caption("Past matches show final scores and prediction accuracy. Future matches show predictions only.")
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        today = datetime.now().date()
        selected_date = st.date_input(
            "Select a date",
            value=today,
            min_value=datetime(2020, 9, 1).date(),
            max_value=datetime(2027, 5, 31).date(),
        )
    with col_b:
        st.write("")
        st.write("")
        jump_option = st.selectbox(
            "Quick jump to",
            ["(no jump)", "Today", "This season start", "Last season start"],
        )
        if jump_option == "Today":
            selected_date = today
        elif jump_option == "This season start":
            selected_date = datetime(2026, 8, 15).date()
        elif jump_option == "Last season start":
            selected_date = datetime(2025, 8, 15).date()
            
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
                    st.caption(f"Final score: {home_goals}–{away_goals} — {actual_label}. Prediction: {pred_label}. {badge}")
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
            
st.caption("⚠️ Predictions use each team's Elo and form as of the last match in the dataset. Not intended for betting advice.")