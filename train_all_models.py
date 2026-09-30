"""
Train and save all three components of the stacked model:
  1. Logistic Regression (LR) on features
  2. Dixon-Coles (DC) on goals
  3. Meta-model that combines LR + DC + base rates

Saves:
  model.joblib            — the LR model
  feature_cols.joblib     — the feature column names
  poisson_model.joblib    — the Dixon-Coles model
  meta_model.joblib       — the meta-model
  base_rates.joblib       — the base rates (H/D/A frequencies)
"""
import pandas as pd
import numpy as np
import glob
import joblib
from sklearn.linear_model import LogisticRegression
from penaltyblog.models import DixonColesGoalModel

from soccer_pipeline import (
    compute_features, compute_sample_weights,
    FEATURE_COLS, SPLIT_DATE,
)


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
    return np.array(0.5 ** (ages / half_life_years),
                    dtype=np.float64, copy=True)


def main():
    print("Loading data...")
    matches = load_matches()
    print(f"Loaded {len(matches)} matches")

    print("\nEngineering features (walk-forward)...")
    matches, _ = compute_features(matches)

    # ---------- Train LR on all pre-2022 data ----------
    train = matches[matches['Date'] < SPLIT_DATE].copy()
    print(f"Training on {len(train)} matches (before {SPLIT_DATE.date()})")

    lr_weights = compute_sample_weights(train['Date'],
                                         reference_date=SPLIT_DATE)
    lr_model = LogisticRegression(max_iter=2000, random_state=42)
    lr_model.fit(train[FEATURE_COLS], train['target'],
                  sample_weight=lr_weights)
    print("Fitted LR")

    # ---------- Train Dixon-Coles on same data ----------
    dc_weights = recency_weights(train['Date'], SPLIT_DATE,
                                  half_life_years=2.0)
    dc_model = DixonColesGoalModel(
        goals_home=train['FTHG'].values.copy(),
        goals_away=train['FTAG'].values.copy(),
        teams_home=train['HomeTeam'].values.copy(),
        teams_away=train['AwayTeam'].values.copy(),
        weights=dc_weights,
    )
    dc_model.fit()
    print("Fitted Dixon-Coles")

    # ---------- Compute base rates ----------
    counts = train['FTR'].value_counts()
    total = counts.sum()
    base_rates = np.array([
        counts.get('H', 0) / total,
        counts.get('D', 0) / total,
        counts.get('A', 0) / total,
    ])
    print(f"Base rates: H={base_rates[0]:.3f}  D={base_rates[1]:.3f}  "
          f"A={base_rates[2]:.3f}")

    # ---------- Train meta-model on out-of-fold predictions ----------
    # Use fold-1 (test on 2021-22) predictions to train the meta-model.
    print("\nTraining meta-model on fold-1 predictions...")
    seasons = ['2020-21', '2021-22', '2022-23',
               '2023-24', '2024-25', '2025-26']
    test_seasons = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']

    meta_train_X = []
    meta_train_y = []

    for test_season in test_seasons:
        train_seasons = [s for s in seasons if s < test_season]
        tr = matches[matches['Season'].isin(train_seasons)].copy()
        te = matches[matches['Season'] == test_season].copy()
        if len(tr) == 0 or len(te) == 0:
            continue

        ref_date = te['Date'].min()

        # Fit fold-specific models
        w = compute_sample_weights(tr['Date'], reference_date=ref_date)
        m_lr = LogisticRegression(max_iter=2000, random_state=42)
        m_lr.fit(tr[FEATURE_COLS], tr['target'], sample_weight=w)
        lr_probs = m_lr.predict_proba(te[FEATURE_COLS])

        w2 = recency_weights(tr['Date'], ref_date, half_life_years=2.0)
        m_dc = DixonColesGoalModel(
            goals_home=tr['FTHG'].values.copy(),
            goals_away=tr['FTAG'].values.copy(),
            teams_home=tr['HomeTeam'].values.copy(),
            teams_away=tr['AwayTeam'].values.copy(),
            weights=w2,
        )
        m_dc.fit()

        base = np.array([
            (tr['FTR'] == 'H').mean(),
            (tr['FTR'] == 'D').mean(),
            (tr['FTR'] == 'A').mean(),
        ])

        dc_probs = []
        for _, row in te.iterrows():
            try:
                p = m_dc.predict(row['HomeTeam'], row['AwayTeam'])
                dc_probs.append([p.home_win, p.draw, p.away_win])
            except Exception:
                dc_probs.append(base.tolist())
        dc_probs = np.array(dc_probs)

        base_probs = np.tile(base, (len(te), 1))

        X = np.hstack([lr_probs, dc_probs, base_probs])
        y = te['FTR'].map({'H': 0, 'D': 1, 'A': 2}).values

        meta_train_X.append(X)
        meta_train_y.append(y)

    meta_X = np.vstack(meta_train_X)
    meta_y = np.concatenate(meta_train_y)
    print(f"Meta training set: {meta_X.shape}")

    meta_model = LogisticRegression(max_iter=2000, random_state=42)
    meta_model.fit(meta_X, meta_y)
    print("Fitted meta-model")

    # ---------- Save everything ----------
    joblib.dump(lr_model, 'model.joblib')
    joblib.dump(FEATURE_COLS, 'feature_cols.joblib')
    joblib.dump(dc_model, 'poisson_model.joblib')
    joblib.dump(meta_model, 'meta_model.joblib')
    joblib.dump(base_rates, 'base_rates.joblib')

    print("\nSaved:")
    print("  model.joblib            (LR)")
    print("  feature_cols.joblib     (feature names)")
    print("  poisson_model.joblib    (Dixon-Coles)")
    print("  meta_model.joblib       (meta-model)")
    print("  base_rates.joblib       (base rates)")


if __name__ == '__main__':
    main()
