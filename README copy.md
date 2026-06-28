
#  2026 World Cup Predictor

Visual bracket simulator for the World Cup using your trained Random Forest model.

## Installation


pip install -r requirements.txt


## 🚀 Execution

1. Make sure `best_rf_model.joblib` is in the **same folder** as `mundial_predictor.py`.
2. Run:


streamlit run mundial_predictor.py


3. The app will automatically open at `http://localhost:8501`.

##  Features

- **Full bracket simulation**: Groups → Round of 16 → Quarter-finals → Semi-finals → Final
- **Statistical analysis**: Run N simulations and display champion, finalist, and semi-finalist probabilities
- **Manual prediction**: Compare any pair of teams in the sidebar
- **Group stage**: Standings table with goals and detailed match results
- **Dark design** with animations and outcome‑based colors

##  Model

The model uses these 8 features:
- `elo_diff` — ELO difference between teams
- `elo_home_before` / `elo_away_before` — ELO rating for each team
- `home_recent_goal_avg` / `away_recent_goal_avg` — Recent goal average
- `h2h_win_pct` — Win percentage in H2H history
- `h2h_goal_diff` — H2H goal difference
- `h2h_match_count` — Number of H2H matches

Predicts: `0=Draw`, `1=Home Win`, `2=Away Win`
```