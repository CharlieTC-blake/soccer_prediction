"""
Stacking experiment: combine LR, Dixon-Coles, and base rates into a
meta-model and evaluate whether it beats the LR baseline.

Walk-forward folds (same as before):
  Fold 1: train 2020-21, test 2021-22
  Fold 2: train 2020-22, test 2022-23
  Fold 3: train 2020-23, test 2023-24
  Fold 4: train 2020-24, test 2024-25
  Fold 5: train 2020-25, test 2025-26

For each fold:
  - Fit LR and Dixon-Coles on training data
  - Predict on the fold's test set
  - Store both predictions for meta-model training

Then train the meta-model on the accumulated base predictions
and evaluate on the same folds.

Note: the meta-model is trained on out-of-fold predictions. Since
we only have one pass, we use the fold-1 predictions to train the
meta and evaluate on folds 2-5.
"""
import pandas as pd
import numpy as np
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
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
    return np.array(0.5 ** (ages / half_life_years), dtype=np.float64, copy=True)


def brier_multiclass(proba, y_true):
    n = len(y_true)
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), y_true] = 1
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def fit_lr(train_with_features, ref_date):
    weights = compute_sample_weights(train_with_features['Date'],
                                     reference_date=ref_date)
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(train_with_features[FEATURE_COLS],
              train_with_features['target'],
              sample_weight=weights)
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


def predict_lr(model, X):
    return model.predict_proba(X)


def predict_dc(model, test, base_fallback):
    out = []
    for _, row in test.iterrows():
        try:
            p = model.predict(row['HomeTeam'], row['AwayTeam'])
            out.append([p.home_win, p.draw, p.away_win])
        except Exception:
            out.append(base_fallback.tolist())
    return np.array(out)


def main():
    print("Loading data...")
    matches = load_matches()
    print(f"Loaded {len(matches)} matches")

    print("\nEngineering features (walk-forward)...")
    matches, _ = compute_features(matches)

    seasons = ['2020-21', '2021-22', '2022-23',
               '2023-24', '2024-25', '2025-26']
    test_seasons = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']

    # Collect per-fold base predictions
    fold_data = []  # list of dicts: {test_season, y_true, lr_probs, dc_probs, base_probs}

    for test_season in test_seasons:
        train_seasons = [s for s in seasons if s < test_season]
        train = matches[matches['Season'].isin(train_seasons)].copy()
        test = matches[matches['Season'] == test_season].copy()

        if len(train) == 0 or len(test) == 0:
            continue

        ref_date = test['Date'].min()
        print(f"\nFold: {test_season} — train {len(train)}, test {len(test)}")

        print("  Fitting LR...")
        lr = fit_lr(train, ref_date)
        lr_probs = predict_lr(lr, test[FEATURE_COLS])

        print("  Fitting Dixon-Coles...")
        dc = fit_dc(train, ref_date)
        base = get_base_rates(train)
        dc_probs = predict_dc(dc, test, base)

        y_true = test['FTR'].map({'H': 0, 'D': 1, 'A': 2}).values

        fold_data.append({
            'test_season': test_season,
            'y_true': y_true,
            'lr_probs': lr_probs,
            'dc_probs': dc_probs,
            'base_probs': np.tile(base, (len(test), 1)),
        })

    # ---------- Evaluate base models ----------
    y_all = np.concatenate([f['y_true'] for f in fold_data])
    lr_all = np.concatenate([f['lr_probs'] for f in fold_data])
    dc_all = np.concatenate([f['dc_probs'] for f in fold_data])
    base_all = np.concatenate([f['base_probs'] for f in fold_data])

    def report(name, proba):
        pred = np.argmax(proba, axis=1)
        acc = accuracy_score(y_all, pred)
        ll = log_loss(y_all, proba, labels=[0, 1, 2])
        bs = brier_multiclass(proba, y_all)
        print(f"  {name:25s} acc={acc:.4f}  ll={ll:.4f}  brier={bs:.4f}")

    print("\n=== BASE MODEL PERFORMANCE (all folds combined) ===")
    report("Logistic Regression", lr_all)
    report("Dixon-Coles", dc_all)
    report("Base rates", base_all)

    # ---------- Stacking ----------
    # Meta-features: concatenate all three prediction vectors
    X_meta = np.hstack([lr_all, dc_all, base_all])

    # Train meta-model on fold-1 data, test on folds 2-5
    first = fold_data[0]
    rest = fold_data[1:]

    X_meta_train = np.hstack([first['lr_probs'], first['dc_probs'],
                              first['base_probs']])
    y_meta_train = first['y_true']

    X_meta_test = np.hstack([
        np.concatenate([f['lr_probs'] for f in rest]),
        np.concatenate([f['dc_probs'] for f in rest]),
        np.concatenate([f['base_probs'] for f in rest]),
    ])
    y_meta_test = np.concatenate([f['y_true'] for f in rest])

    print(f"\nMeta-model: train on {len(y_meta_train)} matches (fold 1), "
          f"test on {len(y_meta_test)} matches (folds 2-5)")

    meta = LogisticRegression(max_iter=2000, random_state=42)
    meta.fit(X_meta_train, y_meta_train)

    # Predict on test
    stacked_probs = meta.predict_proba(X_meta_test)

    # For comparison, evaluate LR on the same folds 2-5
    lr_test = np.concatenate([f['lr_probs'] for f in rest])
    dc_test = np.concatenate([f['dc_probs'] for f in rest])
    base_test = np.concatenate([f['base_probs'] for f in rest])

    def report_subset(name, proba):
        pred = np.argmax(proba, axis=1)
        acc = accuracy_score(y_meta_test, pred)
        ll = log_loss(y_meta_test, proba, labels=[0, 1, 2])
        bs = brier_multiclass(proba, y_meta_test)
        print(f"  {name:25s} acc={acc:.4f}  ll={ll:.4f}  brier={bs:.4f}")

    print("\n=== SAME SUBSET (folds 2-5, where meta was evaluated) ===")
    report_subset("LR (no stacking)", lr_test)
    report_subset("Dixon-Coles (no stacking)", dc_test)
    report_subset("Base rates", base_test)
    report_subset("Stacked (LR+DC+base)", stacked_probs)

    print("\nMeta-model coefficients:")
    coef_df = pd.DataFrame(meta.coef_, columns=(
        [f'LR_{n}' for n in ['H','D','A']] +
        [f'DC_{n}' for n in ['H','D','A']] +
        [f'Base_{n}' for n in ['H','D','A']]
    ), index=['Home', 'Draw', 'Away'])
    print(coef_df.round(3))


if __name__ == '__main__':
    main()
