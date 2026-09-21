"""
Betting backtest for the soccer prediction model.

Uses the same walk-forward features as soccer_pipeline.py, then compares
the model's probabilities against Bet365 odds to see whether betting on
the model's picks would have made money.
"""
import pandas as pd
import numpy as np
import glob
from sklearn.linear_model import LogisticRegression

# ---------- 1. Load data ----------
frames = []
for f in sorted(glob.glob('data/season-*.csv')):
    df = pd.read_csv(f)
    frames.append(df)
matches = pd.concat(frames, ignore_index=True)
matches['Date'] = pd.to_datetime(matches['Date'], format='%d/%m/%Y')
matches = matches.sort_values('Date').reset_index(drop=True)
matches = matches.dropna(subset=['FTHG', 'FTAG', 'FTR'])
matches = matches.dropna(subset=['B365H', 'B365D', 'B365A'])

print(f"Loaded {len(matches)} matches, {matches['Date'].min().date()} to {matches['Date'].max().date()}")

# ---------- 2. Elo ratings (walk-forward) ----------
elo = {}
K = 20
HOME_ADV = 60

def get_elo(team):
    return elo.get(team, 1500)

def expected_score(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

home_elo, away_elo = [], []
for _, row in matches.iterrows():
    h, a = row['HomeTeam'], row['AwayTeam']
    r_h, r_a = get_elo(h), get_elo(a)
    home_elo.append(r_h)
    away_elo.append(r_a)

    if row['FTR'] == 'H':
        s_h = 1.0
    elif row['FTR'] == 'D':
        s_h = 0.5
    else:
        s_h = 0.0

    exp_h = expected_score(r_h + HOME_ADV, r_a)
    elo[h] = r_h + K * (s_h - exp_h)
    elo[a] = r_a + K * ((1 - s_h) - (1 - exp_h))

matches['elo_diff'] = [he - ae for he, ae in zip(home_elo, away_elo)]

# ---------- 3. Rolling form (walk-forward) ----------
team_history = {}

def team_form(team, n=5):
    hist = team_history.get(team, [])
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

# ---------- 4. Rest days ----------
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

# ---------- 5. Target and split ----------
label_map = {'H': 0, 'D': 1, 'A': 2}
matches['target'] = matches['FTR'].map(label_map)

split_date = pd.Timestamp('2022-01-01')
train = matches[matches['Date'] < split_date].copy()
test = matches[matches['Date'] >= split_date].copy()
print(f"Train: {len(train)} matches")
print(f"Test:  {len(test)} matches")

# ---------- 6. Train the model ----------
feature_cols = ['elo_diff', 'form_diff', 'rest_diff']
X_train, y_train = train[feature_cols], train['target']
X_test = test[feature_cols]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------- 7. Get model predictions for the test set ----------
proba_test = model.predict_proba(X_test)

test = test.copy()
test['p_home'] = proba_test[:, 0]
test['p_draw'] = proba_test[:, 1]
test['p_away'] = proba_test[:, 2]

# ---------- 8. Convert Bet365 odds to fair implied probabilities ----------
# Step 1: raw implied probabilities (1 / odds)
test['raw_home'] = 1 / test['B365H']
test['raw_draw'] = 1 / test['B365D']
test['raw_away'] = 1 / test['B365A']

# Step 2: sum them (this is >1 because of the overround)
test['raw_sum'] = test['raw_home'] + test['raw_draw'] + test['raw_away']

# Step 3: normalize so they sum to 1
test['book_home'] = test['raw_home'] / test['raw_sum']
test['book_draw'] = test['raw_draw'] / test['raw_sum']
test['book_away'] = test['raw_away'] / test['raw_sum']

# Report the average overround across all test matches
avg_overround = (test['raw_sum'] - 1).mean()
print(f"\nAverage bookmaker overround: {avg_overround * 100:.2f}%")

# ---------- 9. Betting simulation (multiple configurations) ----------
STAKE = 1.0

def run_backtest(threshold, include_home=True, include_draw=True, include_away=True):
    """Run a betting simulation with the given settings. Returns a summary dict."""
    bets = []
    for _, row in test.iterrows():
        actual = row['FTR']
        outcomes = [
            ('H', row['p_home'], row['book_home'], row['B365H'], 'Home', include_home),
            ('D', row['p_draw'], row['book_draw'], row['B365D'], 'Draw', include_draw),
            ('A', row['p_away'], row['book_away'], row['B365A'], 'Away', include_away),
        ]
        for outcome, model_p, book_p, odds, label, enabled in outcomes:
            if not enabled:
                continue
            if model_p > book_p + threshold:
                won = (actual == outcome)
                profit = STAKE * (odds - 1) if won else -STAKE
                bets.append({
                    'Bet': label, 'ModelP': model_p, 'BookP': book_p,
                    'Edge': model_p - book_p, 'Odds': odds,
                    'Actual': actual, 'Won': won, 'Profit': profit,
                })

    if not bets:
        return None
    df = pd.DataFrame(bets)
    total_staked = len(df) * STAKE
    total_profit = df['Profit'].sum()
    return {
        'threshold': threshold,
        'include_home': include_home,
        'include_draw': include_draw,
        'include_away': include_away,
        'n_bets': len(df),
        'profit': total_profit,
        'roi': total_profit / total_staked * 100,
        'win_rate': df['Won'].mean() * 100,
        'df': df,
    }

# --- Configuration A: vary threshold, bet on everything ---
print("\n=== EXPERIMENT 1: Varying the edge threshold (bet all outcomes) ===")
print(f"{'Threshold':>10s} {'Bets':>6s} {'WinRate':>9s} {'Profit':>9s} {'ROI':>8s}")
for t in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
    result = run_backtest(t)
    if result:
        print(f"{t*100:>9.1f}% {result['n_bets']:>6d} "
              f"{result['win_rate']:>8.1f}% "
              f"£{result['profit']:>+7.2f} "
              f"{result['roi']:>+7.2f}%")

# --- Configuration B: exclude away bets ---
print("\n=== EXPERIMENT 2: Excluding away bets ===")
print(f"{'Threshold':>10s} {'Bets':>6s} {'WinRate':>9s} {'Profit':>9s} {'ROI':>8s}")
for t in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
    result = run_backtest(t, include_home=True, include_draw=True, include_away=False)
    if result:
        print(f"{t*100:>9.1f}% {result['n_bets']:>6d} "
              f"{result['win_rate']:>8.1f}% "
              f"£{result['profit']:>+7.2f} "
              f"{result['roi']:>+7.2f}%")

# --- Configuration C: home bets only ---
print("\n=== EXPERIMENT 3: Home bets only ===")
print(f"{'Threshold':>10s} {'Bets':>6s} {'WinRate':>9s} {'Profit':>9s} {'ROI':>8s}")
for t in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
    result = run_backtest(t, include_home=True, include_draw=False, include_away=False)
    if result:
        print(f"{t*100:>9.1f}% {result['n_bets']:>6d} "
              f"{result['win_rate']:>8.1f}% "
              f"£{result['profit']:>+7.2f} "
              f"{result['roi']:>+7.2f}%")

# --- Configuration D: draw bets only ---
print("\n=== EXPERIMENT 4: Draw bets only ===")
print(f"{'Threshold':>10s} {'Bets':>6s} {'WinRate':>9s} {'Profit':>9s} {'ROI':>8s}")
for t in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
    result = run_backtest(t, include_home=False, include_draw=True, include_away=False)
    if result:
        print(f"{t*100:>9.1f}% {result['n_bets']:>6d} "
              f"{result['win_rate']:>8.1f}% "
              f"£{result['profit']:>+7.2f} "
              f"{result['roi']:>+7.2f}%")

print("\nStage C3c complete. Experiments above.")