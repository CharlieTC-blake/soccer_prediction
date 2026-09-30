"""
Experiment: test different xG feature variants.

Baseline (no xG):       log loss 0.997
Current xG-5 features:  log loss 0.998

Tries:
  A) xG rolling window of 3 (instead of 5)
  B) xG rolling window of 10 (instead of 5)
  C) xG overperformance (goals - xG) rolling window of 5
  D) xG rolling window of 5, but only xG conceded (drop xG created)
  E) xG rolling window of 5, but only xG created (drop xG conceded)
"""
import pandas as pd
import numpy as np
import glob
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss


SPLIT_DATE = pd.Timestamp('2022-01-01')
HALF_LIFE_YEARS = 6.0

BASE_FEATURES = [
    'elo_diff', 'form_diff', 'rest_quality_diff',
    'goals_scored_diff', 'goals_conceded_diff', 'h2h_home_points_L3',
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


def compute_base_features(matches_df):
    """Base features: Elo, form, rest, goals, H2H. No xG."""
    elo = {}
    K = 20
    HOME_ADV = 60
    team_history = {}
    team_goals = {}
    last_played = {}
    h2h_history = {}

    cols = {k: [] for k in [
        'elo_diff', 'form_diff', 'rest_quality_diff',
        'goals_scored_diff', 'goals_conceded_diff', 'h2h_home_points_L3',
    ]}

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

    for _, row in matches_df.iterrows():
        h, a, d = row['HomeTeam'], row['AwayTeam'], row['Date']
        ftr = row['FTR']
        fthg, ftag = row['FTHG'], row['FTAG']

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
    return pd.concat([matches_df, feature_df], axis=1)


def compute_xg_variant(matches_df, variant):
    """Compute xG features for the specified variant."""
    team_xg = {}
    team_goals_vs_xg = {}

    cols = {}

    def rolling_xg(team, n):
        hist = team_xg.get(team, [])
        if not hist:
            return 1.2, 1.2
        recent = hist[-n:]
        return (float(np.mean([h[1] for h in recent])),
                float(np.mean([h[2] for h in recent])))

    def rolling_overperf(team, n):
        hist = team_goals_vs_xg.get(team, [])
        if not hist:
            return 0.0
        recent = hist[-n:]
        return float(np.mean([h[1] for h in recent]))

    if variant == 'A':
        cols['xg_created_diff'] = []
        cols['xg_conceded_diff'] = []
    elif variant == 'B':
        cols['xg_created_diff'] = []
        cols['xg_conceded_diff'] = []
    elif variant == 'C':
        cols['xg_overperf_diff'] = []
    elif variant == 'D':
        cols['xg_conceded_diff'] = []
    elif variant == 'E':
        cols['xg_created_diff'] = []

    for _, row in matches_df.iterrows():
        h, a, d = row['HomeTeam'], row['AwayTeam'], row['Date']
        fthg, ftag = row['FTHG'], row['FTAG']
        home_xg = row.get('home_xg', np.nan)
        away_xg = row.get('away_xg', np.nan)

        if variant == 'A':
            hxc, hxcn = rolling_xg(h, 3)
            axc, axcn = rolling_xg(a, 3)
            cols['xg_created_diff'].append(hxc - axc)
            cols['xg_conceded_diff'].append(hxcn - axcn)
        elif variant == 'B':
            hxc, hxcn = rolling_xg(h, 10)
            axc, axcn = rolling_xg(a, 10)
            cols['xg_created_diff'].append(hxc - axc)
            cols['xg_conceded_diff'].append(hxcn - axcn)
        elif variant == 'C':
            hop = rolling_overperf(h, 5)
            aop = rolling_overperf(a, 5)
            cols['xg_overperf_diff'].append(hop - aop)
        elif variant == 'D':
            _, hxcn = rolling_xg(h, 5)
            _, axcn = rolling_xg(a, 5)
            cols['xg_conceded_diff'].append(hxcn - axcn)
        elif variant == 'E':
            hxc, _ = rolling_xg(h, 5)
            axc, _ = rolling_xg(a, 5)
            cols['xg_created_diff'].append(hxc - axc)

        if pd.notna(home_xg) and pd.notna(away_xg):
            team_xg.setdefault(h, []).append((d, home_xg, away_xg))
            team_xg.setdefault(a, []).append((d, away_xg, home_xg))
            team_goals_vs_xg.setdefault(h, []).append((d, fthg - home_xg))
            team_goals_vs_xg.setdefault(a, []).append((d, ftag - away_xg))

    feature_df = pd.DataFrame(cols, index=matches_df.index)
    return pd.concat([matches_df, feature_df], axis=1)


def compute_sample_weights(dates, reference_date):
    ages = (reference_date - dates).dt.days / 365.25
    return 0.5 ** (ages / HALF_LIFE_YEARS)


def evaluate(matches_df, feature_cols, label):
    train = matches_df[matches_df['Date'] < SPLIT_DATE]
    test = matches_df[matches_df['Date'] >= SPLIT_DATE]
    X_train, y_train = train[feature_cols], train['target']
    X_test, y_test = test[feature_cols], test['target']
    weights = compute_sample_weights(train['Date'], SPLIT_DATE)
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)
    proba = model.predict_proba(X_test)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    ll = log_loss(y_test, proba, labels=[0, 1, 2])
    print(f"  {label:45s} acc={acc:.4f}  logloss={ll:.4f}")
    return ll


def main():
    print("Loading data...")
    frames = []
    for f in sorted(glob.glob('data/season-*.csv')):
        frames.append(pd.read_csv(f))
    matches = pd.concat(frames, ignore_index=True)
    matches['Date'] = pd.to_datetime(matches['Date'], format='ISO8601')
    matches = matches.sort_values('Date').reset_index(drop=True)
    matches = matches.dropna(subset=['FTHG', 'FTAG', 'FTR'])

    print("Computing base features...")
    matches = compute_base_features(matches)
    matches['target'] = matches['FTR'].map({'H': 0, 'D': 1, 'A': 2})

    print("\n--- BASELINE (no xG) ---")
    evaluate(matches, BASE_FEATURES, "No xG (6 features)")

    print("\n--- xG VARIANTS ---")
    for variant, label, cols in [
        ('A', 'xG window=3 (xg_created_diff, xg_conceded_diff)',
              ['xg_created_diff', 'xg_conceded_diff']),
        ('B', 'xG window=10 (xg_created_diff, xg_conceded_diff)',
              ['xg_created_diff', 'xg_conceded_diff']),
        ('C', 'xG overperformance (goals-xG, window=5)',
              ['xg_overperf_diff']),
        ('D', 'xG conceded only (window=5)', ['xg_conceded_diff']),
        ('E', 'xG created only (window=5)', ['xg_created_diff']),
    ]:
        m2 = compute_xg_variant(matches, variant)
        feature_cols = BASE_FEATURES + cols
        evaluate(m2, feature_cols, label)


if __name__ == '__main__':
    main()
