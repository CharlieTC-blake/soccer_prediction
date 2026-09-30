# Premier League Match Prediction System

A complete machine learning pipeline and deployed web application for predicting Premier League match outcomes, built with Python, scikit-learn, and Streamlit.

**Live app:** https://soccerprediction-fww8kgdts3applt95rucyv.streamlit.app/

## What this project does

- Predicts Premier League match outcomes (Home / Draw / Away) using a stacked ensemble of three models.
- Provides a Match Analyser web app that shows model probabilities, head-to-head context, bookmaker odds inputs, and ranked betting recommendations.
- Includes a betting backtest that measures what the models would have earned against real Bet365 odds.

The project is honest about its limits: no model tested is profitable for betting. See the Honest Findings section below.

## Quick start

    git clone https://github.com/CharlieTC-blake/soccer_prediction.git
    cd soccer_prediction
    pip install -r requirements.txt
    python soccer_pipeline.py
    python train_all_models.py
    python -m streamlit run app.py

The app will open in your browser at http://localhost:8501.

## The models

The final production model is a stacked ensemble combining three sources:

| Component | Description |
|---|---|
| Logistic Regression | 8 engineered features: Elo, form, rest quality, goals, head-to-head, xG |
| Dixon-Coles Poisson | Goals-based model fitted on team attack/defense strengths |
| Base rates | Historical H/D/A frequencies from training data |
| Meta-model | Logistic regression combining the three above |

Performance on held-out test set (1,520 matches):

| Model | Accuracy | Log loss |
|---|---|---|
| Logistic Regression alone | 52.5% | 0.992 |
| Dixon-Coles alone | 49.7% | 1.016 |
| Stacked ensemble | 53.6% | 0.987 |

## Features

8 engineered features, all computed walk-forward (no data leakage):

- Elo rating difference (K=20, home advantage = 60)
- Recent form difference (last 5 matches)
- Non-linear rest quality difference (fatigue / match rust)
- Rolling goals scored / conceded difference (last 5)
- Head-to-head points (last 3 meetings)
- Rolling xG created / conceded difference (last 10)

Betting backtest:

- Evaluates against real Bet365 odds
- Uses fractional Kelly staking (quarter-Kelly, 5% cap)
- Sweeps EV thresholds from 0% to 10%

## The web app

The Streamlit app has two tabs.

**Fixtures by Date** - pick any date, see that day's fixtures with the model's predicted probabilities and accuracy tracking for finished matches.

**Match Analyser** - for each match shows model probabilities, head-to-head context, bookmaker odds inputs, ranked recommendations across 10 market outcomes, expected value in UGX, and fair odds after removing the bookmaker's margin.

## Honest findings

No model tested is profitable for betting. The best model loses about 9% of staked money over the test period, against Bet365's 5.46% average overround.

Eight modelling approaches were tested:

| Approach | Result |
|---|---|
| Logistic Regression (baseline) | ~53% accuracy |
| Extended features + sample weights | Small improvement |
| xG features | Small calibration improvement |
| XGBoost | Overfit - worse than LR |
| Isotonic calibration | Worse log loss |
| Over/Under 2.5 model | No signal |
| Dixon-Coles Poisson | Worse than LR alone |
| Stacked ensemble | Best model |

Why? Public data features (Elo, form, goals, xG) are already priced into bookmaker odds. To beat the margin, a model would need information the market does not have - injury news, tactical lineups, proprietary data.

The value of this project is the process, not the profit.

## Data

- Source: football-data.co.uk
- Coverage: 6 Premier League seasons (2020-21 to 2025-26)
- Matches: 2,280
- xG source: Kaggle comprehensive EPL dataset (100% merge coverage)

## Tech stack

- Python 3.13
- pandas, numpy - data manipulation
- scikit-learn - Logistic Regression
- penaltyblog - Dixon-Coles Poisson
- Streamlit - web application
- openfootball - fixtures JSON
- joblib - model serialisation

## Running the tests

    python tests_odds_utils.py

All 11 tests should pass - they verify overround removal and EV calculations.

## License

This project is for educational purposes. Not financial advice.

## Author

Charlie Blake - https://github.com/CharlieTC-blake
