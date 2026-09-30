"""
Experiment: probability calibration via isotonic regression.
"""
import pandas as pd
import numpy as np
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss


TRAIN_END = pd.Timestamp('2022-01-01')
CALIB_END = pd.Timestamp('2023-01-01')
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

    cols = {k: [] for k in [
        'elo_diff', 'form_diff', 'rest_quality_diff',
        'goals_scored_diff', 'goals_conceded_diff', 'h2h_home_points_L3',
        'xg_created_diff', 'xg_conceded_diff',
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
    matches_df['target'] = matches_df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    return matches_df


def compute_sample_weights(dates, reference_date):
    ages = (reference_date - dates).dt.days / 365.25
    return 0.5 ** (ages / HALF_LIFE_YEARS)


def brier_multiclass(proba, y_true):
    n = len(y_true)
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), y_true] = 1
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def calibrate_isotonic(base_proba, y_true):
    iso_models = []
    for k in range(3):
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(base_proba[:, k], (y_true == k).astype(float))
        iso_models.append(iso)
    def calibrate(new_proba):
        out = np.zeros_like(new_proba)
        for k in range(3):
            out[:, k] = iso_models[k].predict(new_proba[:, k])
        out = out / out.sum(axis=1, keepdims=True)
        return out
    return calibrate


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

    train = matches[matches['Date'] < TRAIN_END]
    calib = matches[(matches['Date'] >= TRAIN_END) & (matches['Date'] < CALIB_END)]
    test = matches[matches['Date'] >= CALIB_END]

    print(f"Train:       {len(train)} matches")
    print(f"Calibration: {len(calib)} matches")
    print(f"Test:        {len(test)} matches")
    print()

    X_train, y_train = train[FEATURE_COLS], train['target']
    X_calib, y_calib = calib[FEATURE_COLS], calib['target']
    X_test, y_test = test[FEATURE_COLS], test['target']

    weights = compute_sample_weights(train['Date'], TRAIN_END)

    print("Training base model...")
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)

    base_proba_calib = model.predict_proba(X_calib)
    base_proba_test = model.predict_proba(X_test)

    print("\n--- UNCALIBRATED MODEL ---")
    pred_test = model.predict(X_test)
    acc = accuracy_score(y_test, pred_test)
    ll = log_loss(y_test, base_proba_test, labels=[0, 1, 2])
    bs = brier_multiclass(base_proba_test, y_test.values)
    print(f"Accuracy:    {acc:.4f}")
    print(f"Log loss:    {ll:.4f}")
    print(f"Brier:       {bs:.4f}")

    print("\nFitting isotonic calibration on calibration set...")
    calibrate = calibrate_isotonic(base_proba_calib, y_calib.values)
    cal_proba_test = calibrate(base_proba_test)

    print("\n--- CALIBRATED MODEL ---")
    cal_pred = np.argmax(cal_proba_test, axis=1)
    acc_cal = accuracy_score(y_test, cal_pred)
    ll_cal = log_loss(y_test, cal_proba_test, labels=[0, 1, 2])
    bs_cal = brier_multiclass(cal_proba_test, y_test.values)
    print(f"Accuracy:    {acc_cal:.4f}")
    print(f"Log loss:    {ll_cal:.4f}")
    print(f"Brier:       {bs_cal:.4f}")

    print("\n--- DELTA ---")
    print(f"Accuracy:    {acc_cal - acc:+.4f}")
    print(f"Log loss:    {ll_cal - ll:+.4f}")
    print(f"Brier:       {bs_cal - bs:+.4f}")

    np.save('calibration_test_proba.npy', cal_proba_test)
    np.save('base_test_proba.npy', base_proba_test)
    print("\nSaved base and calibrated probabilities.")


if __name__ == '__main__':
    main()
