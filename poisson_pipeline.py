"""
Dixon-Coles Poisson pipeline with walk-forward evaluation.

Folds (by season, August to July):
  Fold 1: train 2020-21, test 2021-22
  Fold 2: train 2020-22, test 2022-23
  Fold 3: train 2020-23, test 2023-24
  Fold 4: train 2020-24, test 2024-25
  Fold 5: train 2020-25, test 2025-26

Each fold ensures every test match involves teams seen in training.
"""
import pandas as pd
import numpy as np
import glob
import joblib
from sklearn.metrics import accuracy_score, log_loss

from penaltyblog.models import DixonColesGoalModel


# Season boundaries (Aug of start year to Jul of end year)
SEASON_STARTS = {
    '2020-21': ('2020-08-01', '2021-07-31'),
    '2021-22': ('2021-08-01', '2022-07-31'),
    '2022-23': ('2022-08-01', '2023-07-31'),
    '2023-24': ('2023-08-01', '2024-07-31'),
    '2024-25': ('2024-08-01', '2025-07-31'),
    '2025-26': ('2025-08-01', '2026-07-31'),
}


def season_of(date):
    """Return the season string for a given date."""
    year = date.year
    if date.month >= 8:
        return f"{year}-{str(year + 1)[2:]}"
    else:
        return f"{year - 1}-{str(year)[2:]}"


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


def fit_model(train_df):
    model = DixonColesGoalModel(
        goals_home=train_df['FTHG'].values,
        goals_away=train_df['FTAG'].values,
        teams_home=train_df['HomeTeam'].values,
        teams_away=train_df['AwayTeam'].values,
    )
    model.fit()
    return model


def predict_match(model, home_team, away_team):
    """Return (p_home, p_draw, p_away) or None on failure."""
    try:
        pred = model.predict(home_team, away_team)
        return float(pred.home_win), float(pred.draw), float(pred.away_win)
    except Exception:
        return None


def main():
    print("Loading data...")
    matches = load_matches()
    print(f"Loaded {len(matches)} matches, "
          f"{matches['Date'].min().date()} to {matches['Date'].max().date()}")

    seasons = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
    test_seasons = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']

    all_y_true = []
    all_probas = []
    total_errors = 0

    print("\n=== WALK-FORWARD EVALUATION ===")
    for test_season in test_seasons:
        train_seasons = [s for s in seasons
                         if s < test_season]
        train_mask = matches['Season'].isin(train_seasons)
        test_mask = matches['Season'] == test_season

        train = matches[train_mask]
        test = matches[test_mask]

        if len(train) == 0 or len(test) == 0:
            print(f"  {test_season}: skip (empty train or test)")
            continue

        print(f"\n  Fold: test on {test_season}")
        print(f"    Train: {len(train)} matches "
              f"(seasons {train_seasons[0]} to {train_seasons[-1]})")
        print(f"    Test:  {len(test)} matches")

        model = fit_model(train)

        y_true = test['FTR'].map({'H': 0, 'D': 1, 'A': 2}).values
        errors = 0
        for _, row in test.iterrows():
            h = row['HomeTeam']
            a = row['AwayTeam']
            p = predict_match(model, h, a)
            if p is None:
                errors += 1
                all_probas.append([1/3, 1/3, 1/3])
            else:
                all_probas.append(list(p))

        total_errors += errors
        all_y_true.extend(y_true.tolist())
        print(f"    Errors: {errors}")

    y_true = np.array(all_y_true)
    probas = np.array(all_probas)

    pred_classes = np.argmax(probas, axis=1)
    acc = accuracy_score(y_true, pred_classes)
    ll = log_loss(y_true, probas, labels=[0, 1, 2])
    bs = brier_multiclass(probas, y_true)
    baseline_acc = accuracy_score(y_true, np.zeros(len(y_true), dtype=int))

    print(f"\nTotal test matches: {len(y_true)}")
    print(f"Total errors: {total_errors}")

    print("\n--- DIXON-COLES WALK-FORWARD RESULTS ---")
    print(f"Accuracy:            {acc:.4f}")
    print(f"Baseline (always H): {baseline_acc:.4f}")
    print(f"Log loss:            {ll:.4f}")
    print(f"Brier:               {bs:.4f}")

    print("\n--- COMPARISON ---")
    print(f"{'Model':<28s} {'Accuracy':>10s} {'Log loss':>10s} {'Brier':>10s}")
    print(f"{'Logistic Regression':<28s} {0.5294:>10.4f} {0.9958:>10.4f} {0.5923:>10.4f}")
    print(f"{'Dixon-Coles (walk-fwd)':<28s} {acc:>10.4f} {ll:>10.4f} {bs:>10.4f}")

    # ---------- Save final model (trained on all data) ----------
    print("\nFitting final model on all data...")
    final_model = fit_model(matches)
    joblib.dump(final_model, 'poisson_model.joblib')
    print("Saved poisson_model.joblib")

    # ---------- Sample predictions ----------
    print("\nSample predictions (5 matches from each test season):")
    for test_season in test_seasons:
        test = matches[matches['Season'] == test_season].head(3)
        # Reuse the model from that fold
        train_seasons = [s for s in seasons if s < test_season]
        train = matches[matches['Season'].isin(train_seasons)]
        model = fit_model(train)
        for _, row in test.iterrows():
            h = row['HomeTeam']
            a = row['AwayTeam']
            actual = row['FTR']
            p = predict_match(model, h, a)
            if p:
                ph, pdd, pa = p
                print(f"  [{test_season}] {h:15s} vs {a:15s} — "
                      f"H {ph*100:5.1f}%  D {pdd*100:5.1f}%  A {pa*100:5.1f}%  "
                      f"(actual: {actual})")


if __name__ == '__main__':
    main()
