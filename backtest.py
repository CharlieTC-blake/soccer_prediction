"""
Betting backtest for the soccer prediction model.

Uses the same walk-forward features as soccer_pipeline.py, then compares
the model's probabilities against Bet365 odds to see whether betting on
the model's picks would have made money.

Betting backtest for the soccer prediction model.
Selects the SINGLE outcome with the highest positive EV per match to avoid contradictory bets.
"""
import pandas as pd
import numpy as np
import glob
import joblib
from sklearn.linear_model import LogisticRegression

print("Loading data...")
frames = []
for f in sorted(glob.glob('data/season-*.csv')):
    frames.append(pd.read_csv(f))
matches = pd.concat(frames, ignore_index=True)
matches['Date'] = pd.to_datetime(matches['Date'], format='%d/%m/%Y')
matches = matches.sort_values('Date').reset_index(drop=True)
matches = matches.dropna(subset=['FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A'])

print("Engineering features...")
elo = {}
K = 20
HOME_ADV = 60
team_history = {}
last_played = {}

new_cols = {'pre_home_elo': [], 'pre_away_elo': [], 'home_form': [], 'away_form': [], 'home_rest': [], 'away_rest': []}

for _, row in matches.iterrows():
    h, a, d = row['HomeTeam'], row['AwayTeam'], row['Date']
    r_h, r_a = elo.get(h, 1500), elo.get(a, 1500)
    new_cols['pre_home_elo'].append(r_h)
    new_cols['pre_away_elo'].append(r_a)
    
    s_h = 1.0 if row['FTR'] == 'H' else (0.5 if row['FTR'] == 'D' else 0.0)
    exp_h = 1 / (1 + 10 ** ((r_a - (r_h + HOME_ADV)) / 400))
    elo[h] = r_h + K * (s_h - exp_h)
    elo[a] = r_a + K * ((1 - s_h) - (1 - exp_h))
    
    hist_h = team_history.get(h, [])
    hist_a = team_history.get(a, [])
    new_cols['home_form'].append(np.mean([p for _, p in hist_h[-5:]]) if hist_h else 1.0)
    new_cols['away_form'].append(np.mean([p for _, p in hist_a[-5:]]) if hist_a else 1.0)
    
    ph = 3 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0)
    pa = 3 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 0)
    team_history.setdefault(h, []).append((d, ph))
    team_history.setdefault(a, []).append((d, pa))
    
    new_cols['home_rest'].append((d - last_played[h]).days if h in last_played else 14)
    new_cols['away_rest'].append((d - last_played[a]).days if a in last_played else 14)
    last_played[h] = d
    last_played[a] = d

matches = pd.concat([matches, pd.DataFrame(new_cols)], axis=1)
matches['elo_diff'] = matches['pre_home_elo'] - matches['pre_away_elo']
matches['form_diff'] = matches['home_form'] - matches['away_form']
matches['rest_diff'] = matches['home_rest'] - matches['away_rest']
matches['target'] = matches['FTR'].map({'H': 0, 'D': 1, 'A': 2})

split_date = pd.Timestamp('2022-01-01')
train = matches[matches['Date'] < split_date].copy()
test = matches[matches['Date'] >= split_date].copy()

feature_cols = ['elo_diff', 'form_diff', 'rest_diff']
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(train[feature_cols], train['target'])

proba_test = model.predict_proba(test[feature_cols])
test = test.copy()
test['p_home'] = proba_test[:, 0]
test['p_draw'] = proba_test[:, 1]
test['p_away'] = proba_test[:, 2]

test['raw_sum'] = (1 / test['B365H']) + (1 / test['B365D']) + (1 / test['B365A'])
test['book_home'] = (1 / test['B365H']) / test['raw_sum']
test['book_draw'] = (1 / test['B365D']) / test['raw_sum']
test['book_away'] = (1 / test['B365A']) / test['raw_sum']

print(f"\nAverage bookmaker overround: {(test['raw_sum'].mean() - 1) * 100:.2f}%\n")

def run_kelly_backtest(kelly_fraction=0.25, min_ev_threshold=0.0):
    bankroll = 1000.0
    bets = []
    
    for _, row in test.iterrows():
        actual = row['FTR']
        # Evaluate all 3 outcomes
        outcomes = [
            ('H', row['p_home'], row['book_home'], row['B365H']),
            ('D', row['p_draw'], row['book_draw'], row['B365D']),
            ('A', row['p_away'], row['book_away'], row['B365A'])
        ]
        
        # Find the SINGLE outcome with the highest EV that exceeds the threshold
        best_bet = None
        max_ev = -999
        
        for outcome, model_p, book_p, odds in outcomes:
            if pd.isna(odds) or odds <= 1.0:
                continue
            ev = (model_p * odds) - 1.0
            if ev > max_ev and ev > min_ev_threshold:
                max_ev = ev
                best_bet = (outcome, model_p, odds, ev)
        
        # Place bet ONLY on the best outcome
        if best_bet:
            outcome, model_p, odds, ev = best_bet
            b = odds - 1.0
            q = 1.0 - model_p
            kelly_stake_fraction = (b * model_p - q) / b
            
            # Fractional Kelly, capped at 5% of bankroll
            actual_stake_fraction = min(kelly_stake_fraction * kelly_fraction, 0.05)
            
            if actual_stake_fraction > 0:
                stake = bankroll * actual_stake_fraction
                won = (actual == outcome)
                profit = stake * (odds - 1.0) if won else -stake
                bankroll += profit
                
                bets.append({
                    'Match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                    'Outcome': outcome,
                    'ModelProb': model_p,
                    'Odds': odds,
                    'EV': ev,
                    'Stake': stake,
                    'Profit': profit,
                    'Won': won
                })
    
    if not bets:
        return None
        
    df_bets = pd.DataFrame(bets)
    total_staked = df_bets['Stake'].sum()
    total_profit = df_bets['Profit'].sum()
    roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0.0
    
    return {
        'kelly_fraction': kelly_fraction,
        'min_ev': min_ev_threshold,
        'bets_placed': len(df_bets),
        'total_staked': total_staked,
        'total_profit': total_profit,
        'roi': roi,
        'win_rate': (df_bets['Won'].sum() / len(df_bets)) * 100,
        'final_bankroll': bankroll
    }

print("=== QUARTER KELLY BACKTEST (Single Best Bet per Match) ===")
print(f"{'Min EV': >6s} {'Bets': >6s} {'WinRate': >9s} {'ROI': >9s} {'End Bankroll': >12s}")
for ev_thresh in [0.00, 0.03, 0.05, 0.08, 0.10]:
    result = run_kelly_backtest(kelly_fraction=0.25, min_ev_threshold=ev_thresh)
    if result:
        print(f"{ev_thresh*100: >5.0f}%    {result['bets_placed']: >6d}  "
              f"{result['win_rate']: >8.1f}%  "
              f"{result['roi']: >+7.2f}%  "
              f"£{result['final_bankroll']: >10.2f}")
    else:
        print(f"{ev_thresh*100: >5.0f}%    {'0': >6s}  {'0.0%': >9s}  {'0.00%': >8s}  {'£1000.00': >12s}")