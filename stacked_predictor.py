"""
Stacked predictor: combines LR, Dixon-Coles, and base rates via a meta-model.
"""
import numpy as np
import joblib

from soccer_pipeline import FEATURE_COLS, rest_quality


class StackedPredictor:
    def __init__(self):
        self.lr_model = joblib.load('model.joblib')
        self.dc_model = joblib.load('poisson_model.joblib')
        self.meta_model = joblib.load('meta_model.joblib')
        self.base_rates = joblib.load('base_rates.joblib')
        self.feature_cols = FEATURE_COLS

    def _lr_probs(self, feature_dict):
        import pandas as pd
        X = pd.DataFrame([feature_dict], columns=self.feature_cols)
        return self.lr_model.predict_proba(X)[0]

    def _dc_probs(self, home_team, away_team):
        """Return DC probabilities or base rates on failure."""
        try:
            p = self.dc_model.predict(home_team, away_team)
            return np.array([p.home_win, p.draw, p.away_win])
        except Exception:
            return self.base_rates.copy()

    def predict(self, feature_dict, home_team, away_team):
        """Returns (p_home, p_draw, p_away) from the stacked model."""
        lr = self._lr_probs(feature_dict)
        dc = self._dc_probs(home_team, away_team)
        base = self.base_rates

        x_meta = np.concatenate([lr, dc, base]).reshape(1, -1)
        stacked = self.meta_model.predict_proba(x_meta)[0]
        return float(stacked[0]), float(stacked[1]), float(stacked[2])

    def predict_extra_markets(self, home_team, away_team):
        """
        Return extra market probabilities from the Dixon-Coles model.
        Returns dict or None on failure.

        These are informational. Historical testing showed no edge over
        the base rate on public data.
        """
        try:
            p = self.dc_model.predict(home_team, away_team)
            under, push, over = p.totals(2.5)
            return {
                'over_2_5': float(over),
                'under_2_5': float(under),
                'btts_yes': float(p.btts_yes),
                'btts_no': float(p.btts_no),
            }
        except Exception:
            return None
