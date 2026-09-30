"""
Unit tests for odds_utils.py.
Run: python tests_odds_utils.py
"""
import numpy as np
from odds_utils import (
    remove_overround_proportional,
    remove_overround_shin,
    compute_ev,
    kelly_stake_fraction,
)


def assert_close(a, b, tol=1e-6):
    assert abs(a - b) < tol, f"{a} != {b}"


def test_proportional_sums_to_one():
    probs, overround = remove_overround_proportional([2.10, 3.40, 3.75])
    assert_close(probs.sum(), 1.0, tol=1e-9)
    assert overround > 0


def test_proportional_known_values():
    probs, _ = remove_overround_proportional([2.0, 4.0, 4.0])
    assert_close(probs[0], 0.5)
    assert_close(probs[1], 0.25)
    assert_close(probs[2], 0.25)


def test_shin_sums_to_one():
    probs, overround = remove_overround_shin([2.10, 3.40, 3.75])
    assert_close(probs.sum(), 1.0, tol=1e-6)
    assert overround > 0


def test_shin_versus_proportional():
    odds = [1.30, 5.50, 10.00]
    p_prop, _ = remove_overround_proportional(odds)
    p_shin, _ = remove_overround_shin(odds)
    assert p_shin[0] > p_prop[0]
    assert p_shin[2] < p_prop[2]


def test_ev_positive():
    model_p = np.array([0.55, 0.25, 0.20])
    odds = np.array([2.20, 3.40, 4.50])
    evs = compute_ev(model_p, odds)
    assert evs[0] > 0
    assert evs[1] < 0


def test_ev_zero():
    ev = compute_ev(np.array([0.5]), np.array([2.0]))[0]
    assert_close(ev, 0.0)


def test_kelly_positive():
    f = kelly_stake_fraction(0.55, 2.20, fraction=0.25, cap=0.05)
    assert_close(f, 0.04375, tol=1e-6)


def test_kelly_negative_returns_zero():
    f = kelly_stake_fraction(0.30, 2.00)
    assert f == 0.0


def test_kelly_cap():
    f = kelly_stake_fraction(0.9, 3.0, fraction=0.25, cap=0.05)
    assert f == 0.05


def test_invalid_odds_proportional():
    try:
        remove_overround_proportional([0.5, 3.0, 4.0])
        assert False
    except ValueError:
        pass


def test_invalid_odds_shin():
    try:
        remove_overround_shin([1.0, 3.0, 4.0])
        assert False
    except ValueError:
        pass


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f"✅ {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")