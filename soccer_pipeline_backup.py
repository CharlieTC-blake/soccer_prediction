"""
Soccer match outcome prediction pipeline.

Features:
  - Elo rating difference
  - Recent form (last 5) difference
  - Non-linear rest quality difference
  - Rolling goals scored / conceded (last 5) difference
  - Head-to-head points (last 3 meetings) from home perspective

Training uses exponential decay sample weights (concept-drift mitigation).
"""
import pandas as pd
import numpy as np
import glob
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss


SPLIT_DATE = pd.Timestamp('2022-01-01')
HALF_LIFE_YEARS = 6.0

FEATURE_COLS = [
    'elo_diff',
    'form_diff',
    'rest_quality_diff',
    'goals_scored_diff',
    'goals_conceded_diff',
    'h2h_home_points_L3',
]


def rest_quality(days):
    """
    Non-linear rest quality score in [0, 1].

    - < 3 days: fatigue penalty
    - 4-6 days: optimal (1.0)
    - 7-10 days: slight rust
    - > 10 days: significant rust (floor 0.5)
    """
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
    """
    Walk-forward feature engineering.

    Returns (matches_df, final_state) where final_state contains the
    most recent state of every team for predicting future fixtures.
    """
    elo = {}
    K = 20
    HOME_ADV = 60

    team_history = {}
    team_goals = {}
    last_played = {}
    h2h_history = {}

    cols = {k: [] for k in [
        'elo_diff', 'form_diff', 'rest_quality_diff',
        'home_goals_scored_L5', 'home_goals_conceded_L5',
        'away_goals_scored_L5', 'away_goals_conceded_L5',
        'goals_scored_diff', 'goals_conceded_diff',
        'h2h_home_points_L3',
    ]}

    def get_elo(t):
        return elo.get(t, 1500)

    def expected_score(ra, rb):
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

    for _, row in matches_df.iterrows():
        h, a, d = row['HomeTeam'], row['AwayTeam'], row['Date']
        ftr = row['FTR']
        fthg, ftag = row['FTHG'], row['FTAG']

        r_h, r_a = get_elo(h), get_elo(a)
        cols['elo_diff'].append(r_h - r_a)
        s_h = 1.0 if ftr == 'H' else (0.5 if ftr == 'D' else 0.0)
        exp_h = expected_score(r_h + HOME_ADV, r_a)
        elo[h] = r_h + K * (s_h - exp_h)
        elo[a] = r_a + K * ((1 - s_h) - (1 - exp_h))

        hf = rolling_avg(team_history.get(h, []), 5)
        af = rolling_avg(team_history.get(a, []), 5)
        cols['form_diff'].append((hf if hf is not None else 1.0) -
                                  (af if af is not None else 1.0))

        hgs, hgc = rolling_goals(h)
        ags, agc = rolling_goals(a)
        cols['home_goals_scored_L5'].append(hgs)
        cols['home_goals_conceded_L5'].append(hgc)
        cols['away_goals_scored_L5'].append(ags)
        cols['away_goals_conceded_L5'].append(agc)
        cols['goals_scored_diff'].append(hgs - ags)
        cols['goals_conceded_diff'].append(hgc - agc)

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

        h2h_history.setdefault((h, a), []).append(ph)
        h2h_history.setdefault((a, h), []).append(pa)

        last_played[h] = d
        last_played[a] = d

    feature_df = pd.DataFrame(cols, index=matches_df.index)
    matches_df = pd.concat([matches_df, feature_df], axis=1)
    matches_df = matches_df.copy()
    matches_df['target'] = matches_df['FTR'].map({'H': 0, 'D': 1, 'A': 2})

    final_state = {
        'elo': dict(elo),
        'form': {t: (rolling_avg(hist, 5) if rolling_avg(hist, 5) is not None else 1.0)
                 for t, hist in team_history.items()},
        'goals': {t: rolling_goals(t) for t in team_goals.keys()},
        'last_played': dict(last_played),
        'h2h': {k: (float(np.mean(v[-3:])) if v else 1.0)
                for k, v in h2h_history.items()},
        'max_date': matches_df['Date'].max(),
    }

    return matches_df, final_state


def compute_sample_weights(dates, reference_date, half_life_years=HALF_LIFE_YEARS):
    """Exponential decay weight: 0.5 ** (age_years / half_life_years)."""
    ages_years = (reference_date - dates).dt.days / 365.25
    return 0.5 ** (ages_years / half_life_years)


def load_matches():
    frames = []
    for f in sorted(glob.glob('data/season-*.csv')):
        frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.dropna(subset=['FTHG', 'FTAG', 'FTR'])
    return df


def main():
    print("Loading data...")
    matches = load_matches()
    print(f"Loaded {len(matches)} matches, "
          f"{matches['Date'].min().date()} to {matches['Date'].max().date()}")

    print("Engineering features (walk-forward)...")
    matches, _ = compute_features(matches)

    print("Splitting data...")
    train = matches[matches['Date'] < SPLIT_DATE].copy()
    test = matches[matches['Date'] >= SPLIT_DATE].copy()
    print(f"Train: {len(train)}, Test: {len(test)}")

    X_train, y_train = train[FEATURE_COLS], train['target']
    X_test, y_test = test[FEATURE_COLS], test['target']

    weights = compute_sample_weights(train['Date'], reference_date=SPLIT_DATE)

    print("Training Logistic Regression with sample weights...")
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)

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
    print(f"Model accuracy:    {acc:.3f}")
    print(f"Baseline accuracy: {baseline_acc:.3f}")
    print(f"Log loss:          {ll:.3f}")
    print(f"Brier score:       {brier:.3f}")

    print("\nCoefficients:")
    coef_df = pd.DataFrame(model.coef_, columns=FEATURE_COLS,
                            index=['Home', 'Draw', 'Away'])
    print(coef_df.round(4))

    joblib.dump(model, 'model.joblib')
    joblib.dump(FEATURE_COLS, 'feature_cols.joblib')
    print("\nSaved model.joblib and feature_cols.joblib")


if __name__ == '__main__':
    main()