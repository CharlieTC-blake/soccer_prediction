"""
Soccer match outcome prediction pipeline.
Author: Charlie Blake
Data: football-data.co.uk EPL results...
"""
# Section 0: The Imports
import pandas as pd
import numpy as np
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

pd.set_option('display.width', 120) 

# ---------- 1. Load ----------
frames = []
for f in sorted(glob.glob('data/season-*.csv')):
    df = pd.read_csv(f)
    frames.append(df)
matches = pd.concat(frames, ignore_index=True)
matches['Date'] = pd.to_datetime(matches['Date'], format='%d/%m/%Y')
matches = matches.sort_values('Date').reset_index(drop=True)
matches = matches.dropna(subset=['FTHG', 'FTAG', 'FTR'])

print(f"Loaded {len(matches)} matches, {matches['Date'].min().date()} to {matches['Date'].max().date()}")

# ---------- 2. Elo ratings (computed walk-forward, so no leakage) ----------
elo = {}
K = 20
HOME_ADV = 60  # elo points of home advantage, standard approximation

def get_elo(team):
    return elo.get(team, 1500)

def expected_score(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

pre_match_home_elo = []
pre_match_away_elo = []

for _, row in matches.iterrows():
    h, a = row['HomeTeam'], row['AwayTeam']
    r_h, r_a = get_elo(h), get_elo(a)
    pre_match_home_elo.append(r_h)
    pre_match_away_elo.append(r_a)

    # actual result as a score: 1 = home win, 0.5 = draw, 0 = away win
    if row['FTR'] == 'H':
        s_h = 1.0
    elif row['FTR'] == 'D':
        s_h = 0.5
    else:
        s_h = 0.0

    exp_h = expected_score(r_h + HOME_ADV, r_a)
    elo[h] = r_h + K * (s_h - exp_h)
    elo[a] = r_a + K * ((1 - s_h) - (1 - exp_h))

matches['home_elo'] = pre_match_home_elo
matches['away_elo'] = pre_match_away_elo
matches['elo_diff'] = matches['home_elo'] - matches['away_elo']

# ---------- 3. Rolling form (points from last 5 matches, computed walk-forward) ----------
team_history = {}  # team -> list of (date, points_earned)

def team_form(team, n=5):
    hist = team_history.get(team, [])
    if not hist:
        return 1.0  # neutral prior: ~1 point/game average
    recent = hist[-n:]
    return np.mean([p for _, p in recent])

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

matches['home_form'] = home_form
matches['away_form'] = away_form
matches['form_diff'] = matches['home_form'] - matches['away_form']

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
matches['home_rest'] = home_rest
matches['away_rest'] = away_rest
matches['rest_diff'] = matches['home_rest'] - matches['away_rest']

# ---------- 5. Target ----------
label_map = {'H': 0, 'D': 1, 'A': 2}
matches['target'] = matches['FTR'].map(label_map)

# ---------- 6. Walk-forward split: train on everything up to 2022-01-01, test after ----------
split_date = pd.Timestamp('2022-01-01')
train = matches[matches['Date'] < split_date].copy()
test = matches[matches['Date'] >= split_date].copy()
print(f"Train: {len(train)} matches (through {train['Date'].max().date()})")
print(f"Test:  {len(test)} matches ({test['Date'].min().date()} to {test['Date'].max().date()})")

feature_cols = ['elo_diff', 'form_diff', 'rest_diff']
X_train, y_train = train[feature_cols], train['target']
X_test, y_test = test[feature_cols], test['target']

# ---------- 7. Train ----------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------- 8. Predict & evaluate ----------
proba_test = model.predict_proba(X_test)
pred_test = model.predict(X_test)

acc = accuracy_score(y_test, pred_test)
ll = log_loss(y_test, proba_test, labels=[0, 1, 2])

# multiclass brier score (one-vs-rest average)
y_test_onehot = np.zeros((len(y_test), 3))
y_test_onehot[np.arange(len(y_test)), y_test] = 1
brier = np.mean(np.sum((proba_test - y_test_onehot) ** 2, axis=1))

# baseline: always predict "home win" (most common class)
baseline_pred = np.zeros(len(y_test), dtype=int)  # class 0 = H
baseline_acc = accuracy_score(y_test, baseline_pred)

print("\n--- RESULTS (held-out test set, matches from 2022 onward) ---")
print(f"Model accuracy:            {acc:.3f}")
print(f"Baseline ('always home'):  {baseline_acc:.3f}")
print(f"Log loss:                  {ll:.3f}  (lower is better; random guessing ~1.10)")
print(f"Brier score (multiclass):  {brier:.3f}  (lower is better; random guessing ~0.67)")

print("\nModel coefficients (feature -> class weight, classes are [Home, Draw, Away]):")
coef_df = pd.DataFrame(model.coef_, columns=feature_cols, index=['Home', 'Draw', 'Away'])
print(coef_df.round(4))

print("\nSample predictions (first 10 test matches):")
sample = test[['Date', 'HomeTeam', 'AwayTeam', 'FTR']].head(10).copy()
sample_proba = proba_test[:10]
sample['P(Home)'] = sample_proba[:, 0].round(2)
sample['P(Draw)'] = sample_proba[:, 1].round(2)
sample['P(Away)'] = sample_proba[:, 2].round(2)
print(sample.to_string(index=False))

# Save results summary for the writeup
with open('results_summary.txt', 'w') as f:
    f.write(f"Matches total: {len(matches)}\n")
    f.write(f"Train matches: {len(train)}\n")
    f.write(f"Test matches: {len(test)}\n")
    f.write(f"Model accuracy: {acc:.4f}\n")
    f.write(f"Baseline accuracy (always home): {baseline_acc:.4f}\n")
    f.write(f"Log loss: {ll:.4f}\n")
    f.write(f"Brier score: {brier:.4f}\n")
    f.write("\nCoefficients:\n")
    f.write(coef_df.round(4).to_string())
    f.write("\n\nSample predictions:\n")
    f.write(sample.to_string(index=False))

print("\nSaved results_summary.txt")
