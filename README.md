# World Cup 2026 Prediction Using Machine Learning


**Try the live app here:**  
[https://2026usaworldcuppredictor.streamlit.app/]


## Overview

This repository hosts a suite of machine learning models developed to forecast the outcomes of the FIFA World Cup 2026. The project leverages an extensive historical dataset comprising all international football match results spanning from 1876 to 2026, capturing over a century of international football across all confederations.

The core deliverable is a fully interactive Streamlit application that simulates the complete 48‑team, 12‑group tournament — from the group stage through the Round of 32, Round of 16, Quarter‑Finals, Semi‑Finals, and the Final — producing a predicted champion based on model probabilities and Monte Carlo simulation.



## What Was Built

Here is the updated section that explicitly mentions the model selection process, the evaluation of multiple models, and the justification for choosing Random Forest:



### 1. Machine Learning Model

Several machine learning models were trained and systematically evaluated using chronological cross‑validation (TimeSeriesSplit) to ensure a strict separation between past and future data and prevent look‑ahead bias. The candidate models included Random Forest, XGBoost, CatBoost, and HistGradientBoosting. Each model underwent hyperparameter optimisation using Optuna, with the same 8 engineered features and training set.

After extensive evaluation on the 2022 World Cup test set — the most representative benchmark for our prediction target — the Random Forest classifier consistently outperformed the other models in terms of predictive accuracy and stability. While some boosting models achieved slightly higher cross‑validation scores, they suffered from over‑regularisation and failed to generalise as effectively to the high‑stakes, tightly contested matches typical of a World Cup. Random Forest offered the best balance of performance, robustness, and reliable draw predictions, and was therefore selected as the final production model.

- **Algorithm:** Random Forest Classifier (saved as `best_rf_model.joblib`)
- **Task:** Multi‑class classification — predicting whether a match ends in a home win, away win, or draw (classes `0`, `1`, `2`)
- **Training data:** International match results from 1876 to 2026
- **Input features (8 variables):**


 `elo_diff` : ELO rating difference between home and away team 
 `elo_home_before` : ELO rating of the home team prior to the match 
 `elo_away_before` : ELO rating of the away team prior to the match 
 `home_recent_goal_avg` : Home team's average goal difference in recent matches 
 `away_recent_goal_avg` : Away team's average goal difference in recent matches 
 `h2h_win_pct` : Historical head‑to‑head win percentage for the home team 
 `h2h_goal_diff` : Average goal difference in head‑to‑head encounters 
 `h2h_match_count` : Number of historical head‑to‑head matches recorded 
### 2. Streamlit Simulation Application (`worldcup2026_predictor.py`)

An interactive web application with three main sections:

**Bracket Simulation Tab**  
Simulates the full 2026 FIFA World Cup tournament with the official 12‑group draw (Groups A through L, 48 teams). Applies the  2026 format: top 2 from each group qualify automatically; the 8 best third‑place finishers complete the 32‑team knockout bracket. Visualises every round as match cards showing both teams, win probabilities, and the predicted winner. Configurable random seed for reproducible simulations.

**Probability Analysis Tab**  
Runs up to 1,000 independent Monte Carlo simulations of the full tournament and produces ranked probability distributions for championship win probability, final appearance probability, and semi‑final appearance probability per team. All results displayed as proportional bar charts.

**Group Stage Tab**  
Full standings table for all 12 groups (points, goals for, goals against, goal difference). Qualifying teams are highlighted, and each group has an expandable fixture list showing scorelines and results.

**Sidebar — Head‑to‑Head Predictor**  
Select any two teams from the 48‑team pool and instantly get the model's predicted win/draw/loss probabilities for that fixture.








## Requirements

- Python 3.9 or higher
- The trained model file `best_rf_model.joblib` must be placed in the same directory as `worldcup2026_predictor.py`



## Installation and Setup

1. Clone the repository


git clone https://github.com/your-username/World_cup_prediction_using_ML_models.git
cd World_cup_prediction_using_ML_models


2. (Recommended) Create a virtual environment


python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows


3. Install dependencies


pip install -r requirements.txt


4. Run the application

streamlit run worldcup2026_predictor.py

The application will open automatically in your browser at `http://localhost:8501`.



## Dependencies


streamlit>=1.32.0
scikit-learn>=1.3.0
joblib>=1.3.0
numpy>=1.24.0
pandas>=2.0.0



## Dataset

The model was trained on historical international football results covering matches from 1876 to 2026, including friendlies, competitive fixtures, and major tournament matches across all FIFA confederations. Features were engineered to capture both long‑term team strength (ELO ratings) and short‑term form (recent goal averages and head‑to‑head records).


## Live Demo

You can try the application online without any installation:  
[https://2026usaworldcuppredictor.streamlit.app/]
