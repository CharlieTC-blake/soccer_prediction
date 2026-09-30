"""
Soccer match outcome prediction pipeline.
Author: Charlie Blake
Data: football-data.co.uk EPL results...
"""
"""
Soccer match outcome prediction pipeline.
Uses Logistic Regression for stable, well-calibrated probabilities on small datasets.
"""
import pandas as pd
import numpy as np
import glob
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

print("Loading data...")
frames = []
for f in sorted(glob.glob('data/season-*.csv')):
    frames.append(pd.read_csv(f))
matches = pd.concat(frames, ignore_index=True)
matches['Date'] = pd.to_datetime(matches['Date'], format='%d/%m/%Y')
matches = matches.sort_values('Date').reset_index(drop=True)
matches = matches.dropna(subset=['FTHG', 'FTAG', 'FTR'])
print(f"Loaded {len(matches)} matches, {matches['Date'].min().date()} to {matches['Date'].max().date()}")

print("Engineering features (walk-forward)...")
elo = {}
K = 20
HOME_ADV = 60
team_history = {}
last_played = {}

# Use a dictionary to collect new columns to avoid DataFrame fragmentation warnings
new_cols = {
    'pre_home_elo': [], 'pre_away_elo': [],
    'home_form': [], 'away_form': [],
    'home_rest': [], 'away_rest': []
}

for _, row in matches.iterrows():
    h, a, d = row['HomeTeam'], row['AwayTeam'], row['Date']
    
    # Elo
    r_h, r_a = elo.get(h, 1500), elo.get(a, 1500)
    new_cols['pre_home_elo'].append(r_h)
    new_cols['pre_away_elo'].append(r_a)
    
    s_h = 1.0 if row['FTR'] == 'H' else (0.5 if row['FTR'] == 'D' else 0.0)
    exp_h = 1 / (1 + 10 ** ((r_a - (r_h + HOME_ADV)) / 400))
    elo[h] = r_h + K * (s_h - exp_h)
    elo[a] = r_a + K * ((1 - s_h) - (1 - exp_h))
    
    # Form
    hist_h = team_history.get(h, [])
    hist_a = team_history.get(a, [])
    new_cols['home_form'].append(np.mean([p for _, p in hist_h[-5:]]) if hist_h else 1.0)
    new_cols['away_form'].append(np.mean([p for _, p in hist_a[-5:]]) if hist_a else 1.0)
    
    ph = 3 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0)
    pa = 3 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 0)
    team_history.setdefault(h, []).append((d, ph))
    team_history.setdefault(a, []).append((d, pa))
    
    # Rest
    new_cols['home_rest'].append((d - last_played[h]).days if h in last_played else 14)
    new_cols['away_rest'].append((d - last_played[a]).days if a in last_played else 14)
    last_played[h] = d
    last_played[a] = d

# Assign all new columns at once to prevent fragmentation
matches = pd.concat([matches, pd.DataFrame(new_cols)], axis=1)
matches['elo_diff'] = matches['pre_home_elo'] - matches['pre_away_elo']
matches['form_diff'] = matches['home_form'] - matches['away_form']
matches['rest_diff'] = matches['home_rest'] - matches['away_rest']
matches['target'] = matches['FTR'].map({'H': 0, 'D': 1, 'A': 2})

print("Splitting data...")
split_date = pd.Timestamp('2022-01-01')
train = matches[matches['Date'] < split_date].copy()
test = matches[matches['Date'] >= split_date].copy()
print(f"Train: {len(train)} matches, Test: {len(test)} matches")

feature_cols = ['elo_diff', 'form_diff', 'rest_diff']
X_train, y_train = train[feature_cols], train['target']
X_test, y_test = test[feature_cols], test['target']

print("Training Logistic Regression Model...")
# Logistic Regression natively outputs well-calibrated probabilities and is robust on small datasets
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print("Evaluating...")
proba_test = model.predict_proba(X_test)
pred_test = model.predict(X_test)

acc = accuracy_score(y_test, pred_test)
ll = log_loss(y_test, proba_test, labels=[0, 1, 2])
baseline_acc = accuracy_score(y_test, np.zeros(len(y_test), dtype=int))

y_test_onehot = np.zeros((len(y_test), 3))
y_test_onehot[np.arange(len(y_test)), y_test] = 1
brier = np.mean(np.sum((proba_test - y_test_onehot) ** 2, axis=1))

print("\n--- RESULTS ---")
print(f"Model accuracy:   {acc:.3f}")
print(f"Baseline accuracy:{baseline_acc:.3f}")
print(f"Log loss:         {ll:.3f}")
print(f"Brier score:      {brier:.3f}")

joblib.dump(model, 'model.joblib')
joblib.dump(feature_cols, 'feature_cols.joblib')
print("\nSaved model.joblib and feature_cols.joblib")