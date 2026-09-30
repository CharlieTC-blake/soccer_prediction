"""
Betting backtest for the soccer prediction model.

Uses the shared pipeline features and fractional Kelly staking with
single-best-bet-per-match selection.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from soccer_pipeline import (
    compute_features, compute_sample_weights, load_matches,
    FEATURE_COLS, SPLIT_DATE,
)
from odds_utils import kelly_stake_fraction


def run_kelly_backtest(test, kelly_fraction=0.25, min_ev_threshold=0.0):
    bankroll = 1000.0
    bets = []
    for _, row in test.iterrows():
        actual = row['FTR']
        outcomes = [
            ('H', row['p_home'], row['B365H']),
            ('D', row['p_draw'], row['B365D']),
            ('A', row['p_away'], row['B365A']),
        ]
        best_bet = None
        max_ev = -999
        for outcome, model_p, odds in outcomes:
            if pd.isna(odds) or odds <= 1.0:
                continue
            ev = model_p * odds - 1.0
            if ev > max_ev and ev > min_ev_threshold:
                max_ev = ev
                best_bet = (outcome, model_p, odds, ev)

        if best_bet:
            outcome, model_p, odds, ev = best_bet
            stake_frac = kelly_stake_fraction(
                model_p, odds, fraction=kelly_fraction, cap=0.05
            )
            if stake_frac > 0:
                stake = bankroll * stake_frac
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
                    'Won': won,
                })
    if not bets:
        return None
    df_bets = pd.DataFrame(bets)
    total_staked = df_bets['Stake'].sum()
    total_profit = df_bets['Profit'].sum()
    roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0.0
    return {
        'bets_placed': len(df_bets),
        'total_staked': total_staked,
        'total_profit': total_profit,
        'roi': roi,
        'win_rate': (df_bets['Won'].sum() / len(df_bets)) * 100,
        'final_bankroll': bankroll,
    }


def main():
    print("Loading data...")
    matches = load_matches()
    matches = matches.dropna(subset=['B365H', 'B365D', 'B365A'])

    print("Engineering features...")
    matches, _ = compute_features(matches)

    train = matches[matches['Date'] < SPLIT_DATE].copy()
    test = matches[matches['Date'] >= SPLIT_DATE].copy()

    weights = compute_sample_weights(train['Date'], reference_date=SPLIT_DATE)
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(train[FEATURE_COLS], train['target'], sample_weight=weights)

    proba_test = model.predict_proba(test[FEATURE_COLS])
    test = test.copy()
    test['p_home'] = proba_test[:, 0]
    test['p_draw'] = proba_test[:, 1]
    test['p_away'] = proba_test[:, 2]

    raw = 1.0 / test['B365H'] + 1.0 / test['B365D'] + 1.0 / test['B365A']
    print(f"\nAverage bookmaker overround: {(raw.mean() - 1) * 100:.2f}%\n")

    print("=== QUARTER KELLY BACKTEST (Single Best Bet per Match) ===")
    print(f"{'Min EV':>7s} {'Bets':>6s} {'WinRate':>9s} {'ROI':>9s} {'End Bankroll':>14s}")
    for ev_thresh in [0.00, 0.03, 0.05, 0.08, 0.10]:
        result = run_kelly_backtest(test, kelly_fraction=0.25,
                                     min_ev_threshold=ev_thresh)
        if result:
            print(f"{ev_thresh*100:>6.0f}%  "
                  f"{result['bets_placed']:>6d}  "
                  f"{result['win_rate']:>8.1f}%  "
                  f"{result['roi']:>+8.2f}%  "
                  f"£{result['final_bankroll']:>12.2f}")
        else:
            print(f"{ev_thresh*100:>6.0f}%  {0:>6d}  {0.0:>8.1f}%  "
                  f"{0.0:>+8.2f}%  £{1000.0:>12.2f}")


if __name__ == '__main__':
    main()