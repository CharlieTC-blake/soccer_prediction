"""
Fetch Premier League xG data from Understat.
"""
import requests
import json
import pandas as pd
import time

SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
BASE_URL = "https://understat.com/league/EPL/{season}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

def extract_json_block(html, var_name):
    marker = f"var {var_name} = JSON.parse('"
    start = html.find(marker)
    if start == -1:
        return None
    start += len(marker)
    end = html.find("');", start)
    if end == -1:
        return None
    raw = html[start:end]
    try:
        decoded = raw.encode('utf-8').decode('unicode_escape')
    except Exception:
        decoded = raw
    try:
        return json.loads(decoded)
    except json.JSONDecodeError as e:
        print(f"    JSON decode error: {e}")
        return None

def fetch_season(season):
    url = BASE_URL.format(season=season)
    print(f"  GET {url}")
    r = requests.get(url, headers=HEADERS, timeout=30)
    print(f"    HTTP {r.status_code}, page size {len(r.text)} bytes")
    if r.status_code != 200:
        return []
    dates_data = extract_json_block(r.text, "datesData")
    if not dates_data:
        print(f"    Could not extract datesData")
        return []
    matches = []
    for m in dates_data:
        if not m.get('isResult'):
            continue
        try:
            matches.append({
                'season': season,
                'date': m['datetime'].split(' ')[0],
                'home_team': m['h']['title'],
                'away_team': m['a']['title'],
                'home_goals': int(m['goals']['h']),
                'away_goals': int(m['goals']['a']),
                'home_xg': float(m['xG']['h']),
                'away_xg': float(m['xG']['a']),
            })
        except Exception as e:
            print(f"    Skipping row: {e}")
    return matches

def main():
    all_matches = []
    for season in SEASONS:
        print(f"Fetching season {season}...")
        try:
            m = fetch_season(season)
            all_matches.extend(m)
            print(f"  -> {len(m)} matches")
        except Exception as e:
            print(f"  ERROR: {e}")
        time.sleep(2)
    if not all_matches:
        print("\nNo data fetched.")
        return
    df = pd.DataFrame(all_matches)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df.to_csv('data/xg_data.csv', index=False)
    print(f"\nSaved {len(df)} matches to data/xg_data.csv")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(df.head(3).to_string(index=False))

if __name__ == '__main__':
    main()
