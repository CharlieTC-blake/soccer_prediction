"""
Match Analyser: generates ranked recommendations from the model's
1X2 predictions, enriched with bookmaker odds where available.

Currency: UGX (Ugandan Shilling) — configurable stake per bet.
"""
import pandas as pd
import numpy as np


DEFAULT_STAKE_UGX = 1000


def compute_double_chance_probs(p_home, p_draw, p_away):
    return {
        '1X': p_home + p_draw,
        '12': p_home + p_away,
        'X2': p_draw + p_away,
    }


def compute_ev_ugx(model_p, odds, stake_ugx):
    """
    Expected Value in UGX for a single bet.

    Returns None if odds are missing or invalid.
    """
    if odds is None or odds <= 1.0 or stake_ugx <= 0:
        return None
    return stake_ugx * (model_p * odds - 1.0)


def fair_odds(prob):
    """Fair decimal odds for a probability (no bookmaker margin)."""
    if prob <= 0:
        return float('inf')
    return 1.0 / prob


def generate_recommendations(home_team, away_team, p_home, p_draw, p_away,
                             odds_home=None, odds_draw=None, odds_away=None,
                             stake_ugx=DEFAULT_STAKE_UGX):
    """
    Generate three ranked recommendations from model probabilities.
    """
    outcomes = [
        {'label': f'{home_team} to win', 'short': 'H',
         'prob': p_home, 'odds': odds_home},
        {'label': 'Draw', 'short': 'D',
         'prob': p_draw, 'odds': odds_draw},
        {'label': f'{away_team} to win', 'short': 'A',
         'prob': p_away, 'odds': odds_away},
    ]
    outcomes_sorted = sorted(outcomes, key=lambda x: x['prob'], reverse=True)

    double_chances = compute_double_chance_probs(p_home, p_draw, p_away)
    dc_labels = {
        '1X': f'{home_team} or Draw',
        '12': f'{home_team} or {away_team}',
        'X2': f'Draw or {away_team}',
    }
    dc_list = [
        {'label': dc_labels[k], 'short': k, 'prob': v}
        for k, v in double_chances.items()
    ]
    dc_sorted = sorted(dc_list, key=lambda x: x['prob'], reverse=True)

    top_pick = outcomes_sorted[0].copy()
    top_pick['ev_ugx'] = compute_ev_ugx(top_pick['prob'], top_pick['odds'],
                                         stake_ugx)
    top_pick['fair_odds'] = fair_odds(top_pick['prob'])

    top_dc = dc_sorted[0].copy()
    top_dc['fair_odds'] = fair_odds(top_dc['prob'])

    avoid = outcomes_sorted[-1].copy()
    avoid['ev_ugx'] = compute_ev_ugx(avoid['prob'], avoid['odds'], stake_ugx)

    return {
        'stake_ugx': stake_ugx,
        'top_pick': top_pick,
        'top_double_chance': top_dc,
        'avoid': avoid,
        'all_outcomes': outcomes_sorted,
        'double_chances': dc_sorted,
    }


def format_recommendation_block(home_team, away_team, recommendations):
    r = recommendations
    stake = r['stake_ugx']
    lines = []
    lines.append(f"Match: {home_team} vs {away_team}")
    lines.append(f"Stake: UGX {stake:,}")
    lines.append("")
    lines.append("Model probabilities:")
    for o in r['all_outcomes']:
        lines.append(f"  {o['label']:35s} {o['prob']*100:5.1f}%")
    lines.append("")
    lines.append("Recommendations (ranked):")
    tp = r['top_pick']
    lines.append(f"  1. {tp['label']:35s} {tp['prob']*100:5.1f}%  "
                 f"Fair odds {tp['fair_odds']:.2f}")
    if tp['ev_ugx'] is not None:
        lines.append(f"       EV: UGX {tp['ev_ugx']:+,.0f}")
    tdc = r['top_double_chance']
    lines.append(f"  2. {tdc['label']:35s} {tdc['prob']*100:5.1f}%  "
                 f"Fair odds {tdc['fair_odds']:.2f}")
    av = r['avoid']
    lines.append(f"  Avoid: {av['label']:31s} {av['prob']*100:5.1f}%")
    return "\n".join(lines)


if __name__ == '__main__':
    result = generate_recommendations(
        "Arsenal", "Man City",
        p_home=0.32, p_draw=0.26, p_away=0.42,
        odds_home=3.10, odds_draw=3.40, odds_away=2.15,
        stake_ugx=1000,
    )
    print(format_recommendation_block("Arsenal", "Man City", result))
