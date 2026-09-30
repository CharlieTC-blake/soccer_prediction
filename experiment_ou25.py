"""
Experiment: Over/Under 2.5 goals binary prediction.

Same features as the 1X2 model, but:
  - Target: total_goals > 2.5 (binary)
  - Odds: B365>2.5 (Over) and B365<2.5 (Under)
  - Backtest: same Kelly approach, applied to the 2-outcome market
"""
import pandas as pd
import numpy as np
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from odds_utils import remove_overround_proportional, kelly_stake_fraction


SPLIT_DATE = pd.Timestamp('2022-01-01')
HALF_LIFE_YEARS = 6.0

FEATURE_COLS = [
    'elo_diff', 'form_diff', 'rest_quality_diff',
    'goals_scored_diff', 'goals_conceded_diff', 'h2h_home_points_L3',
    'xg_created_diff', 'xg_conceded_diff',
]


def rest_quality(days):
    if days is None or pd.isna(days):
        return 0.7
    if days < 3:
        return 0.3 + 0.2 * days
    if days <= 6:
        return 1.0
    if days <= 10:
        return 1.0 - 0.05 * (days - 6)
    return max(0.5, 0.8 - 0.03 * (days - 10))


def compute_features(matches_df):
    elo = {}
    K = 20
    HOME_ADV = 60
    team_history = {}
    team_goals = {}
    team_xg = {}
    last_played = {}
    h2h_history = {}

    cols = {k: [] for k in FEATURE_COLS}

    def get_elo(t):
        return elo.get(t, 1500)

    def exp_score(ra, rb):
        return 1 / (1 + 10 ** ((rb - ra) / 400))

    def rolling_avg(hist, n):
        if not hist:
            return None
        return float(np.mean([h[1] for h in hist[-n:]]))

    def rolling_goals(team, n=5):
        hist = team_goals.get(team, [])
        if not hist:
            return 1.0, 1.0
        recent = hist[-n:]
        return (float(np.mean([h[1] for h in recent])),
                float(np.mean([h[2] for h in recent])))

    def rolling_xg(team, n=10):
        hist = team_xg.get(team, [])
        if not hist:
            return 1.2, 1.2
        recent = hist[-n:]
        return (float(np.mean([h[1] for h in recent])),
                float(np.mean([h[2] for h in recent])))

    for _, row in matches_df.iterrows():
        h, a, d = row['HomeTeam'], row['AwayTeam'], row['Date']
        ftr = row['FTR']
        fthg, ftag = row['FTHG'], row['FTAG']
        home_xg = row.get('home_xg', np.nan)
        away_xg = row.get('away_xg', np.nan)

        r_h, r_a = get_elo(h), get_elo(a)
        cols['elo_diff'].append(r_h - r_a)
        s_h = 1.0 if ftr == 'H' else (0.5 if ftr == 'D' else 0.0)
        exp_h = exp_score(r_h + HOME_ADV, r_a)
        elo[h] = r_h + K * (s_h - exp_h)
        elo[a] = r_a + K * ((1 - s_h) - (1 - exp_h))

        hf = rolling_avg(team_history.get(h, []), 5)
        af = rolling_avg(team_history.get(a, []), 5)
        cols['form_diff'].append((hf if hf is not None else 1.0) -
                                  (af if af is not None else 1.0))

        hgs, hgc = rolling_goals(h)
        ags, agc = rolling_goals(a)
        cols['goals_scored_diff'].append(hgs - ags)
        cols['goals_conceded_diff'].append(hgc - agc)

        hxc, hxcn = rolling_xg(h)
        axc, axcn = rolling_xg(a)
        cols['xg_created_diff'].append(hxc - axc)
        cols['xg_conceded_diff'].append(hxcn - axcn)

        h_rest = (d - last_played[h]).days if h in last_played else None
        a_rest = (d - last_played[a]).days if a in last_played else None
        cols['rest_quality_diff'].append(rest_quality(h_rest) -
                                          rest_quality(a_rest))

        past = h2h_history.get((h, a), [])[-3:]
        cols['h2h_home_points_L3'].append(
            float(np.mean(past)) if past else 1.0
        )

        ph = 3 if ftr == 'H' else (1 if ftr == 'D' else 0)
        pa = 3 if ftr == 'A' else (1 if ftr == 'D' else 0)
        team_history.setdefault(h, []).append((d, ph))
        team_history.setdefault(a, []).append((d, pa))

        team_goals.setdefault(h, []).append((d, fthg, ftag))
        team_goals.setdefault(a, []).append((d, ftag, fthg))

        if pd.notna(home_xg) and pd.notna(away_xg):
            team_xg.setdefault(h, []).append((d, home_xg, away_xg))
            team_xg.setdefault(a, []).append((d, away_xg, home_xg))

        h2h_history.setdefault((h, a), []).append(ph)
        h2h_history.setdefault((a, h), []).append(pa)

        last_played[h] = d
        last_played[a] = d

    feature_df = pd.DataFrame(cols, index=matches_df.index)
    matches_df = pd.concat([matches_df, feature_df], axis=1)
    return matches_df


def compute_sample_weights(dates, reference_date):
    ages = (reference_date - dates).dt.days / 365.25
    return 0.5 ** (ages / HALF_LIFE_YEARS)


def kelly_binary_backtest(test_df, kelly_fraction=0.25, min_ev_threshold=0.0):
    bankroll = 1000.0
    bets = []

    for _, row in test_df.iterrows():
        over_odds = row['B365>2.5']
        under_odds = row['B365<2.5']
        if pd.isna(over_odds) or pd.isna(under_odds) or over_odds <= 1 or under_odds <= 1:
            continue

        p_over = row['p_over']
        p_under = 1.0 - p_over

        # EV calculation for binary market
        over_ev = p_over * over_odds - 1.0
        under_ev = p_under * under_odds - 1.0

        best_bet = None
        max_ev = -999
        if over_ev > min_ev_threshold and over_ev > max_ev:
            max_ev = over_ev
            best_bet = ('OVER', p_over, over_odds, over_ev)
        if under_ev > min_ev_threshold and under_ev > max_ev:
            max_ev = under_ev
            best_bet = ('UNDER', p_under, under_odds, under_ev)

        if best_bet:
            outcome, model_p, odds, ev = best_bet
            stake_frac = kelly_stake_fraction(model_p, odds,
                                              fraction=kelly_fraction, cap=0.05)
            if stake_frac > 0:
                stake = bankroll * stake_frac
                won = (outcome == 'OVER' and row['actual_over'] == 1) or \
                      (outcome == 'UNDER' and row['actual_over'] == 0)
                profit = stake * (odds - 1.0) if won else -stake
                bankroll += profit
                bets.append({
                    'Outcome': outcome, 'ModelProb': model_p, 'Odds': odds,
                    'EV': ev, 'Stake': stake, 'Profit': profit, 'Won': won,
                })

    if not bets:
        return None
    df = pd.DataFrame(bets)
    total_staked = df['Stake'].sum()
    total_profit = df['Profit'].sum()
    return {
        'bets_placed': len(df),
        'total_staked': total_staked,
        'total_profit': total_profit,
        'roi': total_profit / total_staked * 100 if total_staked > 0 else 0,
        'win_rate': df['Won'].mean() * 100,
        'final_bankroll': bankroll,
    }


def main():
    print("Loading data...")
    frames = []
    for f in sorted(glob.glob('data/season-*.csv')):
        frames.append(pd.read_csv(f))
    matches = pd.concat(frames, ignore_index=True)
    matches['Date'] = pd.to_datetime(matches['Date'], format='ISO8601')
    matches = matches.sort_values('Date').reset_index(drop=True)
    matches = matches.dropna(subset=['FTHG', 'FTAG', 'FTR'])

    print("Computing features...")
    matches = compute_features(matches)

    # Binary target: 1 if total goals > 2.5, else 0
    matches['total_goals'] = matches['FTHG'] + matches['FTAG']
    matches['actual_over'] = (matches['total_goals'] > 2.5).astype(int)

    print(f"Overall Over 2.5 rate: {matches['actual_over'].mean():.3f}")

    train = matches[matches['Date'] < SPLIT_DATE]
    test = matches[matches['Date'] >= SPLIT_DATE].copy()
    print(f"Train: {len(train)}, Test: {len(test)}")

    X_train, y_train = train[FEATURE_COLS], train['actual_over']
    X_test, y_test = test[FEATURE_COLS], test['actual_over']
    weights = compute_sample_weights(train['Date'], SPLIT_DATE)

    print("\nTraining binary Logistic Regression...")
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)

    p_over_test = model.predict_proba(X_test)[:, 1]
    test['p_over'] = p_over_test

    # Metrics
    pred = (p_over_test > 0.5).astype(int)
    acc = accuracy_score(y_test, pred)
    ll = log_loss(y_test, p_over_test)
    bs = brier_score_loss(y_test, p_over_test)
    baseline_acc = accuracy_score(y_test,
                                   np.ones(len(y_test), dtype=int))

    print("\n--- OVER/UNDER 2.5 METRICS ---")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Baseline (always Over): {baseline_acc:.4f}")
    print(f"Log loss:           {ll:.4f}")
    print(f"Brier score:        {bs:.4f}")

    print("\nCoefficients:")
    coef_df = pd.DataFrame(model.coef_, columns=FEATURE_COLS,
                            index=['Over_2.5'])
    print(coef_df.round(4))

    # Backtest
    print("\n=== KELLY BACKTEST (Over/Under 2.5) ===")
    print(f"{'Min EV':>7s} {'Bets':>6s} {'WinRate':>9s} {'ROI':>9s} {'End Bankroll':>14s}")
    for ev_thresh in [0.00, 0.02, 0.05, 0.08, 0.10]:
        result = kelly_binary_backtest(test, kelly_fraction=0.25,
                                       min_ev_threshold=ev_thresh)
        if result:
            print(f"{ev_thresh*100:>6.0f}%  "
                  f"{result['bets_placed']:>6d}  "
                  f"{result['win_rate']:>8.1f}%  "
                  f"{result['roi']:>+8.2f}%  "
                  f"£{result['final_bankroll']:>12.2f}")


if __name__ == '__main__':
    main()
