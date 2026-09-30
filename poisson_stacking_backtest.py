"""
Backtest the stacked model against the LR baseline.

Uses the same walk-forward folds as poisson_stacking.py, then runs
a Kelly backtest for both models on the same test matches.
"""
import pandas as pd
import numpy as np
import glob
from sklearn.linear_model import LogisticRegression
from penaltyblog.models import DixonColesGoalModel

from soccer_pipeline import (
    compute_features, compute_sample_weights,
    FEATURE_COLS, SPLIT_DATE,
)
from odds_utils import kelly_stake_fraction


def season_of(date):
    year = date.year
    return (f"{year}-{str(year + 1)[2:]}" if date.month >= 8
            else f"{year - 1}-{str(year)[2:]}")


def load_matches():
    frames = []
    for f in sorted(glob.glob('data/season-*.csv')):
        frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], format='ISO8601')
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.dropna(subset=['FTHG', 'FTAG', 'FTR',
                           'B365H', 'B365D', 'B365A'])
    df['Season'] = df['Date'].apply(season_of)
    return df


def recency_weights(dates, reference_date, half_life_years=2.0):
    ages = (reference_date - dates).dt.days / 365.25
    return np.array(0.5 ** (ages / half_life_years), dtype=np.float64, copy=True)


def fit_lr(train, ref_date):
    weights = compute_sample_weights(train['Date'], reference_date=ref_date)
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(train[FEATURE_COLS], train['target'], sample_weight=weights)
    return model


def fit_dc(train, ref_date):
    w = recency_weights(train['Date'], ref_date, half_life_years=2.0)
    model = DixonColesGoalModel(
        goals_home=train['FTHG'].values.copy(),
        goals_away=train['FTAG'].values.copy(),
        teams_home=train['HomeTeam'].values.copy(),
        teams_away=train['AwayTeam'].values.copy(),
        weights=w,
    )
    model.fit()
    return model


def get_base_rates(train):
    counts = train['FTR'].value_counts()
    total = counts.sum()
    return np.array([
        counts.get('H', 0) / total,
        counts.get('D', 0) / total,
        counts.get('A', 0) / total,
    ])


def predict_dc(model, test, base_fallback):
    out = []
    for _, row in test.iterrows():
        try:
            p = model.predict(row['HomeTeam'], row['AwayTeam'])
            out.append([p.home_win, p.draw, p.away_win])
        except Exception:
            out.append(base_fallback.tolist())
    return np.array(out)


def kelly_backtest(test_df, proba_cols,
                   kelly_fraction=0.25, min_ev_threshold=0.0):
    """
    test_df must have columns: B365H, B365D, B365A, FTR, and proba_cols
    (list of three column names for H, D, A probabilities).
    """
    bankroll = 1000.0
    bets = []
    for _, row in test_df.iterrows():
        oh = row['B365H']
        od = row['B365D']
        oa = row['B365A']
        if pd.isna(oh) or pd.isna(od) or pd.isna(oa):
            continue

        p_h = row[proba_cols[0]]
        p_d = row[proba_cols[1]]
        p_a = row[proba_cols[2]]

        outcomes = [
            ('H', p_h, oh),
            ('D', p_d, od),
            ('A', p_a, oa),
        ]

        best = None
        max_ev = -999
        for label, model_p, odds in outcomes:
            if odds <= 1.0:
                continue
            ev = model_p * odds - 1.0
            if ev > max_ev and ev > min_ev_threshold:
                max_ev = ev
                best = (label, model_p, odds, ev)

        if best:
            label, model_p, odds, ev = best
            stake_frac = kelly_stake_fraction(
                model_p, odds, fraction=kelly_fraction, cap=0.05
            )
            if stake_frac > 0:
                stake = bankroll * stake_frac
                won = (row['FTR'] == label)
                profit = stake * (odds - 1.0) if won else -stake
                bankroll += profit
                bets.append({'stake': stake, 'profit': profit, 'won': won})

    if not bets:
        return None
    df = pd.DataFrame(bets)
    total_staked = df['stake'].sum()
    total_profit = df['profit'].sum()
    return {
        'bets': len(df),
        'total_staked': total_staked,
        'profit': total_profit,
        'roi': total_profit / total_staked * 100 if total_staked > 0 else 0,
        'win_rate': df['won'].mean() * 100,
        'bankroll': bankroll,
    }


def main():
    print("Loading data...")
    matches = load_matches()
    print(f"Loaded {len(matches)} matches with odds")

    print("\nEngineering features (walk-forward)...")
    matches, _ = compute_features(matches)

    seasons = ['2020-21', '2021-22', '2022-23',
               '2023-24', '2024-25', '2025-26']
    test_seasons = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']

    folds = []

    for test_season in test_seasons:
        train_seasons = [s for s in seasons if s < test_season]
        train = matches[matches['Season'].isin(train_seasons)].copy()
        test = matches[matches['Season'] == test_season].copy()

        if len(train) == 0 or len(test) == 0:
            continue

        ref_date = test['Date'].min()
        print(f"\nFold: {test_season} — train {len(train)}, test {len(test)}")

        lr = fit_lr(train, ref_date)
        lr_probs = lr.predict_proba(test[FEATURE_COLS])

        dc = fit_dc(train, ref_date)
        base = get_base_rates(train)
        dc_probs = predict_dc(dc, test, base)
        base_probs = np.tile(base, (len(test), 1))

        test = test.copy()
        test['lr_H'] = lr_probs[:, 0]
        test['lr_D'] = lr_probs[:, 1]
        test['lr_A'] = lr_probs[:, 2]
        test['dc_H'] = dc_probs[:, 0]
        test['dc_D'] = dc_probs[:, 1]
        test['dc_A'] = dc_probs[:, 2]
        test['base_H'] = base_probs[:, 0]
        test['base_D'] = base_probs[:, 1]
        test['base_A'] = base_probs[:, 2]

        folds.append(test)

    # Train meta-model on fold-1 stacked features
    first = folds[0]
    X_meta_train = np.hstack([
        first[['lr_H', 'lr_D', 'lr_A']].values,
        first[['dc_H', 'dc_D', 'dc_A']].values,
        first[['base_H', 'base_D', 'base_A']].values,
    ])
    y_meta_train = first['FTR'].map({'H': 0, 'D': 1, 'A': 2}).values

    meta = LogisticRegression(max_iter=2000, random_state=42)
    meta.fit(X_meta_train, y_meta_train)

    # Add stacked probabilities to folds 2-5
    folds_rest = folds[1:]
    test_all = pd.concat(folds_rest, ignore_index=True)

    X_meta_test = np.hstack([
        test_all[['lr_H', 'lr_D', 'lr_A']].values,
        test_all[['dc_H', 'dc_D', 'dc_A']].values,
        test_all[['base_H', 'base_D', 'base_A']].values,
    ])
    stacked = meta.predict_proba(X_meta_test)
    test_all['stack_H'] = stacked[:, 0]
    test_all['stack_D'] = stacked[:, 1]
    test_all['stack_A'] = stacked[:, 2]

    print(f"\nBacktest on {len(test_all)} matches (folds 2-5)")

    print("\n=== KELLY BACKTEST (5% threshold) ===")
    print(f"{'Model':<25s} {'Bets':>6s} {'WinRate':>9s} {'ROI':>9s} {'End Bankroll':>14s}")

    for name, cols in [
        ("LR alone", ['lr_H', 'lr_D', 'lr_A']),
        ("Dixon-Coles alone", ['dc_H', 'dc_D', 'dc_A']),
        ("Stacked", ['stack_H', 'stack_D', 'stack_A']),
    ]:
        result = kelly_backtest(test_all, cols,
                                kelly_fraction=0.25,
                                min_ev_threshold=0.05)
        if result:
            print(f"{name:<25s} "
                  f"{result['bets']:>6d}  "
                  f"{result['win_rate']:>8.1f}%  "
                  f"{result['roi']:>+8.2f}%  "
                  f"£{result['bankroll']:>12.2f}")


if __name__ == '__main__':
    main()
