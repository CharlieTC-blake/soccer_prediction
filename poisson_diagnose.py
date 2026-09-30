"""
Dixon-Coles: diagnose the accuracy gap.

Tests 4 variants against the LR baseline:
  A: baseline (uniform weights, uniform fallback)
  B: base-rate fallback only
  C: recency weights only
  D: recency weights + base-rate fallback

Warning: this fits 4 model sets (5 folds each) = 20 fits.
Expect 10-20 minutes.
"""
import pandas as pd
import numpy as np
import glob
import joblib
from sklearn.metrics import accuracy_score, log_loss

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


def brier_multiclass(proba, y_true):
    n = len(y_true)
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), y_true] = 1
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def recency_weights(dates, reference_date, half_life_years=2.0):
    ages = (reference_date - dates).dt.days / 365.25
    return (0.5 ** (ages / half_life_years)).values


def fit_model(train_df, weights=None):
    model = DixonColesGoalModel(
        goals_home=train_df['FTHG'].values,
        goals_away=train_df['FTAG'].values,
        teams_home=train_df['HomeTeam'].values,
        teams_away=train_df['AwayTeam'].values,
        weights=weights,
    )
    model.fit()
    return model


def base_rate(train_df):
    counts = train_df['FTR'].value_counts()
    total = counts.sum()
    return np.array([
        counts.get('H', 0) / total,
        counts.get('D', 0) / total,
        counts.get('A', 0) / total,
    ])


def evaluate_variant(matches, seasons, test_seasons,
                     use_weights, use_fallback, label):
    all_y_true = []
    all_probas = []
    total_errors = 0

    for test_season in test_seasons:
        train_seasons = [s for s in seasons if s < test_season]
        train = matches[matches['Season'].isin(train_seasons)]
        test = matches[matches['Season'] == test_season]

        if len(train) == 0 or len(test) == 0:
            continue

        if use_weights:
            ref_date = test['Date'].min()
            w = recency_weights(train['Date'], ref_date, half_life_years=2.0)
        else:
            w = None

        model = fit_model(train, weights=w)
        fallback = base_rate(train) if use_fallback else np.array([1/3, 1/3, 1/3])

        y_true = test['FTR'].map({'H': 0, 'D': 1, 'A': 2}).values
        for _, row in test.iterrows():
            h = row['HomeTeam']
            a = row['AwayTeam']
            try:
                pred = model.predict(h, a)
                all_probas.append([pred.home_win, pred.draw, pred.away_win])
            except Exception:
                total_errors += 1
                all_probas.append(fallback.tolist())
        all_y_true.extend(y_true.tolist())

    y_true = np.array(all_y_true)
    probas = np.array(all_probas)
    pred_classes = np.argmax(probas, axis=1)
    acc = accuracy_score(y_true, pred_classes)
    ll = log_loss(y_true, probas, labels=[0, 1, 2])
    bs = brier_multiclass(probas, y_true)

    print(f"  {label:45s} acc={acc:.4f}  ll={ll:.4f}  brier={bs:.4f}  err={total_errors}")
    return acc, ll, bs


def main():
    print("Loading data...")
    matches = load_matches()
    print(f"Loaded {len(matches)} matches\n")

    seasons = ['2020-21', '2021-22', '2022-23',
               '2023-24', '2024-25', '2025-26']
    test_seasons = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']

    print("--- Variant comparison (LR baseline for reference) ---")
    print(f"  {'LR baseline':45s} acc=0.5294  ll=0.9958  brier=0.5923  err=—")

    evaluate_variant(matches, seasons, test_seasons,
                     use_weights=False, use_fallback=False,
                     label="A: uniform weights, uniform fallback")
    evaluate_variant(matches, seasons, test_seasons,
                     use_weights=False, use_fallback=True,
                     label="B: uniform weights, base-rate fallback")
    evaluate_variant(matches, seasons, test_seasons,
                     use_weights=True, use_fallback=False,
                     label="C: recency weights, uniform fallback")
    evaluate_variant(matches, seasons, test_seasons,
                     use_weights=True, use_fallback=True,
                     label="D: recency weights, base-rate fallback")


if __name__ == '__main__':
    main()
