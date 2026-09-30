"""
Test Dixon-Coles on Over/Under 2.5 and Both Teams To Score.

Uses penaltyblog's native market methods (no manual score matrix extraction).
"""
import pandas as pd
import numpy as np
import glob
from sklearn.metrics import log_loss, brier_score_loss

from penaltyblog.models import DixonColesGoalModel


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
    df = df.dropna(subset=['FTHG', 'FTAG', 'FTR'])
    df['Season'] = df['Date'].apply(season_of)
    return df


def recency_weights(dates, reference_date, half_life_years=2.0):
    ages = (reference_date - dates).dt.days / 365.25
    return np.array(0.5 ** (ages / half_life_years), dtype=np.float64, copy=True)


def fit_model(train_df, weights):
    model = DixonColesGoalModel(
        goals_home=train_df['FTHG'].values.copy(),
        goals_away=train_df['FTAG'].values.copy(),
        teams_home=train_df['HomeTeam'].values.copy(),
        teams_away=train_df['AwayTeam'].values.copy(),
        weights=weights,
    )
    model.fit()
    return model


def predict_markets(model, home, away):
    """Return P(Over 2.5) and P(BTTS) or None on failure."""
    try:
        p = model.predict(home, away)
        under, push, over = p.totals(2.5)
        return {'p_over': float(over), 'p_btts': float(p.btts_yes)}
    except Exception:
        return None


def evaluate_binary(y_true, y_pred, label):
    ll = log_loss(y_true, y_pred)
    bs = brier_score_loss(y_true, y_pred)
    acc = ((y_pred > 0.5) == y_true).mean()
    print(f"  {label:35s} acc={acc:.4f}  ll={ll:.4f}  brier={bs:.4f}")


def main():
    print("Loading data...")
    matches = load_matches()
    print(f"Loaded {len(matches)} matches\n")

    seasons = ['2020-21', '2021-22', '2022-23',
               '2023-24', '2024-25', '2025-26']
    test_seasons = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']

    rows = []  # each: actual_over, actual_btts, p_over, p_btts, base_over, base_btts

    for test_season in test_seasons:
        train_seasons = [s for s in seasons if s < test_season]
        train = matches[matches['Season'].isin(train_seasons)]
        test = matches[matches['Season'] == test_season]

        if len(train) == 0 or len(test) == 0:
            continue

        print(f"Fold: test on {test_season} — train {len(train)}, test {len(test)}")
        ref_date = test['Date'].min()
        w = recency_weights(train['Date'], ref_date, half_life_years=2.0)
        model = fit_model(train, weights=w)

        train_over_rate = float((train['FTHG'] + train['FTAG'] > 2.5).mean())
        train_btts_rate = float(((train['FTHG'] >= 1) & (train['FTAG'] >= 1)).mean())

        n_ok = 0
        n_fail = 0
        for _, row in test.iterrows():
            h = row['HomeTeam']
            a = row['AwayTeam']
            hg, ag = int(row['FTHG']), int(row['FTAG'])
            actual_over = int(hg + ag > 2.5)
            actual_btts = int(hg >= 1 and ag >= 1)

            m = predict_markets(model, h, a)
            if m is None:
                n_fail += 1
                rows.append((actual_over, actual_btts,
                             train_over_rate, train_btts_rate,
                             train_over_rate, train_btts_rate))
            else:
                n_ok += 1
                rows.append((actual_over, actual_btts,
                             m['p_over'], m['p_btts'],
                             train_over_rate, train_btts_rate))
        print(f"  OK: {n_ok}, Failed: {n_fail}")

    arr = np.array(rows)
    y_over = arr[:, 0].astype(int)
    y_btts = arr[:, 1].astype(int)
    p_over = arr[:, 2]
    p_btts = arr[:, 3]
    base_over = arr[:, 4]
    base_btts = arr[:, 5]

    print(f"\nTotal test matches: {len(y_over)}")

    print("\n--- OVER 2.5 ---")
    evaluate_binary(y_over, base_over, "Baseline (train rate)")
    evaluate_binary(y_over, p_over, "Dixon-Coles")

    print("\n--- BTTS ---")
    evaluate_binary(y_btts, base_btts, "Baseline (train rate)")
    evaluate_binary(y_btts, p_btts, "Dixon-Coles")


if __name__ == '__main__':
    main()
