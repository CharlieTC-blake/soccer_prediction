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
    if odds is None or odds <= 1.0 or stake_ugx <= 0:
        return None
    return stake_ugx * (model_p * odds - 1.0)


def fair_odds(prob):
    if prob <= 0:
        return float('inf')
    return 1.0 / prob


def generate_recommendations(home_team, away_team, p_home, p_draw, p_away,
                             odds_home=None, odds_draw=None, odds_away=None,
                             stake_ugx=DEFAULT_STAKE_UGX):
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


def rank_all_markets(p_home, p_draw, p_away, extra_markets,
                     home_team, away_team, odds_home=None,
                     odds_draw=None, odds_away=None, stake_ugx=1000):
    """
    Rank all available market outcomes by model confidence.

    Parameters:
      p_home, p_draw, p_away: 1X2 probabilities from the stacked model
      extra_markets: dict with keys 'over_2_5', 'under_2_5', 'btts_yes',
                     'btts_no' (or None if unavailable)
      home_team, away_team: display names
      odds_home, odds_draw, odds_away: optional bookmaker odds for 1X2
      stake_ugx: stake for EV computation

    Returns dict with 'singles', 'double_chances', 'all'.
    """
    def make_entry(label, prob, market, odds=None):
        entry = {
            'label': label,
            'prob': float(prob),
            'market': market,
            'fair_odds': (1.0 / prob) if prob > 0 else float('inf'),
            'odds': odds,
        }
        if odds is not None and odds > 1.0:
            entry['ev_ugx'] = stake_ugx * (prob * odds - 1.0)
        else:
            entry['ev_ugx'] = None
        return entry

    singles = [
        make_entry(f'{home_team} to win', p_home, '1X2', odds_home),
        make_entry('Draw', p_draw, '1X2', odds_draw),
        make_entry(f'{away_team} to win', p_away, '1X2', odds_away),
    ]

    if extra_markets:
        singles.append(make_entry('Over 2.5 goals',
                                   extra_markets['over_2_5'], 'O/U'))
        singles.append(make_entry('Under 2.5 goals',
                                   extra_markets['under_2_5'], 'O/U'))
        singles.append(make_entry('BTTS Yes',
                                   extra_markets['btts_yes'], 'BTTS'))
        singles.append(make_entry('BTTS No',
                                   extra_markets['btts_no'], 'BTTS'))

    singles.sort(key=lambda x: x['prob'], reverse=True)

    double_chances = [
        make_entry(f'{home_team} or Draw', p_home + p_draw, 'DC'),
        make_entry(f'{home_team} or {away_team}', p_home + p_away, 'DC'),
        make_entry(f'Draw or {away_team}', p_draw + p_away, 'DC'),
    ]
    double_chances.sort(key=lambda x: x['prob'], reverse=True)

    all_outcomes = singles + double_chances
    all_outcomes.sort(key=lambda x: x['prob'], reverse=True)

    return {
        'singles': singles,
        'double_chances': double_chances,
        'all': all_outcomes,
    }


def get_head_to_head(matches_df, home_team, away_team, n=5):
    mask = (
        ((matches_df['HomeTeam'] == home_team) & (matches_df['AwayTeam'] == away_team)) |
        ((matches_df['HomeTeam'] == away_team) & (matches_df['AwayTeam'] == home_team))
    )
    meetings = matches_df[mask].sort_values('Date', ascending=False).head(n)

    if len(meetings) == 0:
        return [], {}

    rows = []
    home_wins = 0
    away_wins = 0
    draws = 0
    total_goals = 0

    for _, m in meetings.iterrows():
        h, a = m['HomeTeam'], m['AwayTeam']
        hg, ag = int(m['FTHG']), int(m['FTAG'])
        total_goals += hg + ag

        if hg > ag:
            winner = h
        elif hg == ag:
            winner = None
        else:
            winner = a

        if winner == home_team:
            home_wins += 1
            label = f"{home_team} win"
        elif winner == away_team:
            away_wins += 1
            label = f"{away_team} win"
        else:
            draws += 1
            label = "Draw"

        rows.append({
            'date': m['Date'].strftime('%Y-%m-%d'),
            'home': h,
            'away': a,
            'home_goals': hg,
            'away_goals': ag,
            'result': label,
        })

    summary = {
        'matches': len(meetings),
        'home_wins': home_wins,
        'away_wins': away_wins,
        'draws': draws,
        'avg_total_goals': total_goals / len(meetings),
    }
    return rows, summary


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
