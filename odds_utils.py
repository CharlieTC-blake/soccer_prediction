"""
Overround removal and Expected Value utilities.

Two methods for converting bookmaker odds into true probabilities:
  - Proportional Margin Method (simple, standard)
  - Shin's Method (corrects for favourite-longshot bias)

Both take decimal odds [home, draw, away] and return probabilities
that sum to exactly 1.0.
"""
import numpy as np


def remove_overround_proportional(odds):
    """
    Proportional Margin Method.

    probs_i = (1 / odds_i) / sum_j(1 / odds_j)

    Returns:
        (probs, overround)
    """
    odds = np.asarray(odds, dtype=float)
    if np.any(odds <= 1.0):
        raise ValueError("All odds must be greater than 1.0")
    raw = 1.0 / odds
    total = raw.sum()
    probs = raw / total
    return probs, float(total - 1.0)


def remove_overround_shin(odds, max_iter=200, tol=1e-10):
    """
    Shin's Method for overround removal.

    Bisection on z in [0, 0.5] to make sum(pi(z)) = 1.

    Reference: Shin, H. S. (1993). Economic Journal, 103(420), 1141-1153.
    """
    odds = np.asarray(odds, dtype=float)
    if np.any(odds <= 1.0):
        raise ValueError("All odds must be greater than 1.0")

    raw = 1.0 / odds
    total = raw.sum()

    def pi_of_z(z):
        if z <= 0:
            return raw / total
        if z >= 1:
            return raw / raw.sum()
        num = np.sqrt(z ** 2 + 4 * (1 - z) * (raw ** 2) / total) - z
        return num / (2 * (1 - z))

    lo, hi = 0.0, 0.5
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        s = pi_of_z(mid).sum()
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = mid
        else:
            hi = mid

    probs = pi_of_z((lo + hi) / 2.0)
    probs = probs / probs.sum()
    return probs, float(total - 1.0)


def compute_ev(model_probs, odds):
    """
    EV per £1 stake: EV_i = model_p_i * (odds_i - 1) - (1 - model_p_i)
    """
    model_probs = np.asarray(model_probs, dtype=float)
    odds = np.asarray(odds, dtype=float)
    if model_probs.shape != odds.shape:
        raise ValueError("model_probs and odds must have the same shape")
    return model_probs * (odds - 1.0) - (1.0 - model_probs)


def kelly_stake_fraction(model_p, odds, fraction=0.25, cap=0.05):
    """
    Fractional Kelly stake as a fraction of bankroll.
    Returns 0.0 if the Kelly stake is non-positive.
    """
    b = odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - model_p
    f_star = (b * model_p - q) / b
    if f_star <= 0:
        return 0.0
    return min(f_star * fraction, cap)