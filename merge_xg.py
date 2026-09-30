"""
Merge xG data from the Kaggle dataset into our existing season CSVs.

Reads:
    data/season-*.csv      (from football-data.co.uk)
    data/epl_matches_xg.csv (from Kaggle)

Writes:
    data/season-*.csv (overwritten with new xg columns)

Columns added:
    home_xg, away_xg
"""
import pandas as pd
import glob
import os

# Team name translation: Kaggle -> our data
KAGGLE_TO_OURS = {
    "Nottingham": "Nott'm Forest",
    "Sheffield": "Sheffield United",
    "Man City": "Man City",
    "Man United": "Man United",
    "Newcastle": "Newcastle",
    "West Ham": "West Ham",
    "West Brom": "West Brom",
    "Crystal Palace": "Crystal Palace",
    "Aston Villa": "Aston Villa",
    "Tottenham": "Tottenham",
    "Brighton": "Brighton",
    "Wolves": "Wolves",
    "Bournemouth": "Bournemouth",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Southampton": "Southampton",
    "Norwich": "Norwich",
    "Watford": "Watford",
    "Ipswich": "Ipswich",
    "Hull": "Hull",
    "Sunderland": "Sunderland",
    "Coventry": "Coventry",
    "Luton": "Luton",
    "Burnley": "Burnley",
    "Brentford": "Brentford",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Arsenal": "Arsenal",
    "Chelsea": "Chelsea",
    "Liverpool": "Liverpool",
}

def translate(team):
    return KAGGLE_TO_OURS.get(team, team)

def main():
    print("Loading Kaggle xG data...")
    xg = pd.read_csv('data/epl_matches_xg.csv')
    print(f"  Total rows: {len(xg)}")

    # Build a proper date column (Year-Month-Day -> datetime)
    xg['date'] = pd.to_datetime(
        dict(year=xg['Year'], month=xg['Month'], day=xg['Day']),
        errors='coerce'
    )

    # Translate team names
    xg['home_norm'] = xg['HomeTeam'].apply(translate)
    xg['away_norm'] = xg['AwayTeam'].apply(translate)

    # Keep only what we need
    xg_slim = xg[['date', 'home_norm', 'away_norm', 'XGHome', 'XGAway']].copy()
    xg_slim = xg_slim.rename(columns={
        'home_norm': 'HomeTeam',
        'away_norm': 'AwayTeam',
        'XGHome': 'home_xg',
        'XGAway': 'away_xg',
    })

    # Drop rows with NaN xG or date
    before = len(xg_slim)
    xg_slim = xg_slim.dropna(subset=['date', 'home_xg', 'away_xg'])
    print(f"  Rows with complete data: {len(xg_slim)} (dropped {before - len(xg_slim)})")

    # Process each season file
    for season_file in sorted(glob.glob('data/season-*.csv')):
        print(f"\nProcessing {season_file}...")
        df = pd.read_csv(season_file)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        # Merge on date + home + away
        merged = df.merge(
            xg_slim,
            left_on=['Date', 'HomeTeam', 'AwayTeam'],
            right_on=['date', 'HomeTeam', 'AwayTeam'],
            how='left'
        )

        # Drop the redundant date column from the merge
        if 'date' in merged.columns:
            merged = merged.drop(columns=['date'])

        # Report match rate
        matched = merged['home_xg'].notna().sum()
        total = len(merged)
        pct = 100 * matched / total if total > 0 else 0
        print(f"  Matched {matched}/{total} rows ({pct:.1f}%)")

        # Warn if many misses
        if pct < 80:
            print(f"  WARNING: only {pct:.1f}% matched. Check team name translations.")

        # Save back
        merged.to_csv(season_file, index=False)
        print(f"  Saved {season_file}")

    print("\nMerge complete.")

if __name__ == '__main__':
    main()
