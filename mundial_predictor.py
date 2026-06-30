"""
World Cup 2026 Predictor
------------------------
A Streamlit application for simulating the FIFA World Cup 2026 tournament
using a pre-trained Random Forest classifier.

Requirements:
    pip install streamlit scikit-learn joblib numpy pandas

Usage:
    Place best_rf_model.joblib in the same directory as this script, then run:
    streamlit run worldcup2026_predictor.py

Model features expected (in order):
    elo_diff, elo_home_before, elo_away_before,
    home_recent_goal_avg, away_recent_goal_avg,
    h2h_win_pct, h2h_goal_diff, h2h_match_count

Model output classes:
    0 = Draw, 1 = Home win, 2 = Away win
"""

import pickle
import random
import warnings
from pathlib import Path
from textwrap import dedent

import joblib
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# GLOBAL STYLES
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }

    .match-card {
        background: linear-gradient(135deg, #1c2128 0%, #21262d 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 6px 0;
        transition: transform 0.2s;
    }
    .match-card:hover { transform: scale(1.02); border-color: #58a6ff; }

    .team-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 0;
    }
    .win-pct {
        font-size: 12px;
        color: #8b949e;
        background: #21262d;
        border-radius: 6px;
        padding: 2px 8px;
    }
    .vs-badge {
        text-align: center;
        color: #8b949e;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
    }

    .phase-header {
        background: linear-gradient(90deg, #1f6feb 0%, #388bfd 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 16px;
        text-align: center;
        margin-bottom: 10px;
    }

    .champion-card {
        background: linear-gradient(135deg, #b8860b 0%, #ffd700 50%, #b8860b 100%);
        color: #000;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        font-size: 28px;
        font-weight: 900;
        margin: 20px auto;
        max-width: 420px;
        box-shadow: 0 0 40px #ffd70080;
    }

    .group-table {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .group-title {
        color: #58a6ff;
        font-weight: 700;
        font-size: 15px;
        border-bottom: 1px solid #30363d;
        padding-bottom: 6px;
        margin-bottom: 8px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        font-size: 15px;
        width: 100%;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #388bfd, #58a6ff); }

    div[data-testid="stSidebar"] { background-color: #161b22; }

    h1, h2, h3 { color: #e6edf3; }

    .stat-box {
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 13px;
    }
    .stat-label { color: #8b949e; font-size: 11px; }
    .stat-value { color: #3fb950; font-weight: 700; font-size: 16px; }

    /* ---------------------------------------------------------------- */
    /* SYMMETRICAL KNOCKOUT BRACKET (Custom Knockout tab)                */
    /* ---------------------------------------------------------------- */
    .bracket-scroll {
        overflow-x: auto;
        padding: 10px 4px 24px 4px;
    }
    .bracket-grid {
        display: flex;
        align-items: stretch;
        gap: 10px;
        min-width: max-content;
    }
    .bracket-col {
        display: flex;
        flex-direction: column;
        justify-content: space-around;
        min-width: 168px;
        max-width: 168px;
    }
    .bracket-col-title {
        text-align: center;
        font-size: 10.5px;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #8b949e;
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 5px 4px;
        margin-bottom: 10px;
    }
    .bracket-card {
        background: linear-gradient(135deg, #1c2128 0%, #21262d 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 8px 10px;
        margin: 8px 0;
    }
    .bracket-team-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 6px;
        padding: 3px 0;
        font-size: 12.5px;
    }
    .bracket-team-name { color: #c9d1d9; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .bracket-team-name.bracket-winner { color: #3fb950; font-weight: 700; }
    .bracket-pct {
        font-size: 11px;
        font-weight: 700;
        color: #8b949e;
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 5px;
        padding: 1px 6px;
        flex-shrink: 0;
    }
    .bracket-pct.bracket-winner { color: #3fb950; border-color: #2ea04344; }
    .bracket-center-col {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        gap: 16px;
        min-width: 178px;
        max-width: 178px;
    }
    .bracket-final-card {
        background: linear-gradient(135deg, #2d2410 0%, #3a2e0f 100%);
        border: 1px solid #b8860b;
    }
    .bracket-third-card { opacity: 0.92; }
    .bracket-sub-label {
        text-align: center;
        font-size: 9.5px;
        font-weight: 700;
        letter-spacing: 1px;
        color: #b8860b;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TEAM DATA  (ELO ratings and average goals per match)
# ---------------------------------------------------------------------------
TEAM_DATA = {
    # Group A
    "Mexico":          {"elo": 1912, "goals_avg": 1.78, "group": "A"},
    "South Korea":     {"elo": 1723, "goals_avg": 1.82, "group": "A"},
    "South Africa":    {"elo": 1575, "goals_avg": 1.36, "group": "A"},
    "Czech Republic":  {"elo": 1680, "goals_avg": 1.84, "group": "A"},
    
    # Group B
    "Canada":                {"elo": 1748, "goals_avg": 1.29, "group": "B"},
    "Switzerland":           {"elo": 1914, "goals_avg": 1.50, "group": "B"},
    "Qatar":                 {"elo": 1411, "goals_avg": 1.42, "group": "B"},
    "Bosnia and Herzegovina": {"elo": 1622, "goals_avg": 1.41, "group": "B"},
    
    # Group C
    "Brazil":          {"elo": 2009, "goals_avg": 2.16, "group": "C"},
    "Morocco":         {"elo": 1877, "goals_avg": 1.52, "group": "C"},
    "Scotland":        {"elo": 1745, "goals_avg": 1.72, "group": "C"},
    "Haiti":           {"elo": 1517, "goals_avg": 1.70, "group": "C"},
    
    # Group D
    "USA":             {"elo": 1781, "goals_avg": 1.52, "group": "D"},
    "Australia":       {"elo": 1800, "goals_avg": 2.02, "group": "D"},
    "Paraguay":        {"elo": 1815, "goals_avg": 1.27, "group": "D"},
    "Türkiye":         {"elo": 1852, "goals_avg": 1.42, "group": "D"},
    
    # Group E
    "Germany":         {"elo": 1916, "goals_avg": 2.26, "group": "E"},
    "Ecuador":         {"elo": 1902, "goals_avg": 1.20, "group": "E"},
    "Ivory Coast":     {"elo": 1743, "goals_avg": 1.63, "group": "E"},
    "Curacao":         {"elo": 1438, "goals_avg": 1.76, "group": "E"},
    
    # Group F
    "Netherlands":     {"elo": 1980, "goals_avg": 2.10, "group": "F"},
    "Japan":           {"elo": 1910, "goals_avg": 1.82, "group": "F"},
    "Tunisia":         {"elo": 1562, "goals_avg": 1.42, "group": "F"},
    "Sweden":          {"elo": 1742, "goals_avg": 1.99, "group": "F"},
    
    # Group G
    "Belgium":         {"elo": 1884, "goals_avg": 1.83, "group": "G"},
    "Iran":            {"elo": 1764, "goals_avg": 1.90, "group": "G"},
    "Egypt":           {"elo": 1742, "goals_avg": 1.75, "group": "G"},
    "New Zealand":     {"elo": 1534, "goals_avg": 1.74, "group": "G"},
    
    # Group H
    "Spain":           {"elo": 2144, "goals_avg": 2.04, "group": "H"},
    "Uruguay":         {"elo": 1841, "goals_avg": 1.58, "group": "H"},
    "Saudi Arabia":    {"elo": 1596, "goals_avg": 1.52, "group": "H"},
    "Cape Verde":      {"elo": 1622, "goals_avg": 1.07, "group": "H"},
    
    # Group I
    "France":          {"elo": 2123, "goals_avg": 1.82, "group": "I"},
    "Senegal":         {"elo": 1842, "goals_avg": 1.38, "group": "I"},
    "Norway":          {"elo": 1918, "goals_avg": 1.56, "group": "I"},
    "Iraq":            {"elo": 1561, "goals_avg": 1.57, "group": "I"},
    
    # Group J
    "Argentina":       {"elo": 2148, "goals_avg": 1.91, "group": "J"},
    "Austria":         {"elo": 1836, "goals_avg": 1.80, "group": "J"},
    "Algeria":         {"elo": 1785, "goals_avg": 1.53, "group": "J"},
    "Jordan":          {"elo": 1628, "goals_avg": 1.25, "group": "J"},
    
    # Group K
    "Portugal":                {"elo": 1990, "goals_avg": 1.76, "group": "K"},
    "Colombia":                {"elo": 2004, "goals_avg": 1.30, "group": "K"},
    "Uzbekistan":              {"elo": 1631, "goals_avg": 1.72, "group": "K"},
    "Democratic Republic of the Congo": {"elo": 1712, "goals_avg": 1.52, "group": "K"},
    
    # Group L
    "England":         {"elo": 2038, "goals_avg": 2.34, "group": "L"},
    "Croatia":         {"elo": 1905, "goals_avg": 1.74, "group": "L"},
    "Panama":          {"elo": 1658, "goals_avg": 1.24, "group": "L"},
    "Ghana":           {"elo": 1575, "goals_avg": 1.60, "group": "L"},
}

GROUPS = {
    "A": ["Mexico", "South Korea", "South Africa", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["USA", "Australia", "Paraguay", "Türkiye"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "Democratic Republic of the Congo"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}

TEAM_ABBR = {
    "Mexico": "MEX", "South Korea": "KOR", "South Africa": "RSA", "Czech Republic": "CZE",
    "Canada": "CAN", "Switzerland": "SUI", "Qatar": "QAT", "Bosnia and Herzegovina": "BIH",
    "Brazil": "BRA", "Morocco": "MAR", "Scotland": "SCO", "Haiti": "HAI",
    "USA": "USA", "Australia": "AUS", "Paraguay": "PAR", "Türkiye": "TUR",
    "Germany": "GER", "Ecuador": "ECU", "Ivory Coast": "CIV", "Curacao": "CUW",
    "Netherlands": "NED", "Japan": "JPN", "Tunisia": "TUN", "Sweden": "SWE",
    "Belgium": "BEL", "Iran": "IRN", "Egypt": "EGY", "New Zealand": "NZL",
    "Spain": "ESP", "Uruguay": "URU", "Saudi Arabia": "KSA", "Cape Verde": "CPV",
    "France": "FRA", "Senegal": "SEN", "Norway": "NOR", "Iraq": "IRQ",
    "Argentina": "ARG", "Austria": "AUT", "Algeria": "ALG", "Jordan": "JOR",
    "Portugal": "POR", "Colombia": "COL", "Uzbekistan": "UZB",
    "Democratic Republic of the Congo": "COD",
    "England": "ENG", "Croatia": "CRO", "Panama": "PAN", "Ghana": "GHA",
}


def team_display(name: str, use_abbr: bool = False) -> str:
    """Return the text-only label for a team: full name or 3-letter code."""
    if not use_abbr:
        return name
    return TEAM_ABBR.get(name, name[:3].upper())


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "best_rf_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None

@st.cache_resource
def load_h2h_data():
    try:
        with open('h2h_data.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

@st.cache_resource
def load_team_data():
    try:
        with open('team_data.pkl', 'rb') as f:
            data = pickle.load(f)
        for team in TEAM_DATA.keys():
            if team not in data:
                found = False
                for key in data.keys():
                    if team.lower() in key.lower() or key.lower() in team.lower():
                        data[team] = data.pop(key)
                        found = True
                        break
                if not found:
                    data[team] = {'elo': 1500.0, 'goals_avg': 0.0}
        return data
    except FileNotFoundError:
        return TEAM_DATA

h2h_dict = load_h2h_data()
model = load_model()
TEAM_DATA_DYNAMIC = load_team_data()


# ---------------------------------------------------------------------------
# PREDICTION LOGIC
# ---------------------------------------------------------------------------
def predict_match(home: str, away: str):
    """
    Return (prob_home_win, prob_draw, prob_away_win) for a given fixture.
    Falls back to an ELO-based estimate when the model file is unavailable.
    """
    # Defensive checks for custom bracket text placeholders (e.g. GM77)
    if home not in TEAM_DATA_DYNAMIC or away not in TEAM_DATA_DYNAMIC:
        return 0.5, 0.0, 0.5

    hd = TEAM_DATA_DYNAMIC[home]
    ad = TEAM_DATA_DYNAMIC[away]
    elo_diff = hd["elo"] - ad["elo"]
    key = tuple(sorted([home, away]))
    if key in h2h_dict:
        h2h_win_pct, h2h_goal_diff, h2h_count = h2h_dict[key]
    else:
        h2h_win_pct, h2h_goal_diff, h2h_count = 0.5, 0.0, 0
        
    features = np.array([[
        elo_diff,
        hd["elo"],
        ad["elo"],
        hd["goals_avg"],
        ad["goals_avg"],
        h2h_win_pct,
        h2h_goal_diff,
        h2h_count,
    ]])

    if model is not None:
        proba = model.predict_proba(features)[0]
        return float(proba[1]), float(proba[0]), float(proba[2])
    
    # ELO baseline fallback
    dr = 0.2
    ph = 1.0 / (10.0 ** (-elo_diff / 400.0) + 1.0)
    pa = 1.0 - ph
    return ph * (1 - dr), dr, pa * (1 - dr)


def simulate_match(home: str, away: str, allow_draw: bool = True):
    """
    Randomly sample a match outcome according to predicted probabilities.
    Returns (winner, prob_home, prob_draw, prob_away).
    In knockout rounds (allow_draw=False) the draw probability is redistributed.
    """
    ph, pd, pa = predict_match(home, away)
    if not allow_draw:
        total = ph + pa if (ph + pa) > 0 else 1.0
        ph, pa = ph / total, pa / total
        pd = 0.0

    roll = random.random()
    if roll < ph:
        return home, ph, pd, pa
    if roll < ph + pd:
        return "Draw", ph, pd, pa
    return away, ph, pd, pa

# ---------------------------------------------------------------------------
# CUSTOM KNOCKOUT BRACKET SIMULATION
# ---------------------------------------------------------------------------
KNOCKOUT_ROUNDS = [
    {"key": "r32", "label": "Round of 32", "slots": 32},
    {"key": "r16", "label": "Round of 16", "slots": 16},
    {"key": "qf",  "label": "Quarter-Finals", "slots": 8},
    {"key": "sf",  "label": "Semi-Finals", "slots": 4},
    {"key": "final", "label": "Final", "slots": 2},
]

ROUND_LABELS   = [r["label"] for r in KNOCKOUT_ROUNDS]
ROUND_BY_LABEL = {r["label"]: r for r in KNOCKOUT_ROUNDS}


def simulate_custom_bracket(starting_label: str, slot_teams: list) -> dict:
    start_index = next(
        i for i, r in enumerate(KNOCKOUT_ROUNDS) if r["label"] == starting_label
    )
    rounds_to_play = KNOCKOUT_ROUNDS[start_index:]

    bracket_results = {}
    current_teams = list(slot_teams)

    for round_info in rounds_to_play:
        round_label = round_info["label"]
        matches = []
        next_round_teams = []
        for i in range(0, len(current_teams), 2):
            home, away = current_teams[i], current_teams[i + 1]
            w, ph, pd, pa = simulate_match(home, away, allow_draw=False)
            matches.append({"home": home, "away": away, "winner": w, "ph": ph, "pa": pa})
            next_round_teams.append(w)
        bracket_results[round_label] = matches
        current_teams = next_round_teams

    if "Semi-Finals" in bracket_results:
        sf_matches = bracket_results["Semi-Finals"]
        if len(sf_matches) == 2:
            losers = [
                (m["away"] if m["winner"] == m["home"] else m["home"])
                for m in sf_matches
            ]
            w3, ph3, pd3, pa3 = simulate_match(losers[0], losers[1], allow_draw=False)
            bracket_results["Third Place"] = [{
                "home": losers[0], "away": losers[1],
                "winner": w3, "ph": ph3, "pa": pa3,
            }]

    bracket_results["champion"] = current_teams[0]
    return bracket_results


def get_bracket_display_data(starting_label: str, slot_teams: list, simulated_results: dict = None) -> dict:
    """
    Builds the visual tree matrix. If a user hasn't run the simulation yet,
    it returns placeholder elements matching the requested design layout.
    """
    if simulated_results is not None:
        return simulated_results
        
    start_index = next(i for i, r in enumerate(KNOCKOUT_ROUNDS) if r["label"] == starting_label)
    rounds_to_display = KNOCKOUT_ROUNDS[start_index:]
    
    display_bracket = {}
    current_round_teams = list(slot_teams)
    
    base_ids = {
        "Round of 32": 50,
        "Round of 16": 77,
        "Quarter-Finals": 89,
        "Semi-Finals": 97,
        "Final": 101,
        "Third Place": 103
    }
    
    for round_idx, round_info in enumerate(rounds_to_display):
        round_label = round_info["label"]
        slots_needed = round_info["slots"]
        n_matches = slots_needed // 2
        
        matches = []
        next_round_teams = []
        
        for i in range(n_matches):
            home = current_round_teams[i * 2]
            away = current_round_teams[i * 2 + 1]
            
            matches.append({
                "home": home,
                "away": away,
                "winner": None,
                "ph": None,
                "pa": None
            })
            
            if round_idx + 1 < len(rounds_to_display):
                next_round_label = rounds_to_display[round_idx + 1]["label"]
                next_round_base = base_ids.get(next_round_label, 100)
                next_round_teams.append(f"GM{next_round_base + i}")
            else:
                next_round_teams.append(f"GM{base_ids['Final'] + 1}")
                
        display_bracket[round_label] = matches
        current_round_teams = next_round_teams
        
    if "Semi-Finals" in display_bracket:
        display_bracket["Third Place"] = [{
            "home": f"GM{base_ids['Semi-Finals']}",
            "away": f"GM{base_ids['Semi-Finals'] + 1}",
            "winner": None,
            "ph": None,
            "pa": None
        }]
        
    return display_bracket

# ---------------------------------------------------------------------------
# FULL TOURNAMENT SIMULATION
# ---------------------------------------------------------------------------
def simulate_world_cup() -> dict:
    results = {
        "groups": {}, "r32": [], "r16": [], "qf": [], "sf": [], "final": [],
        "champion": None, "best_thirds": [],
    }

    group_standings: dict[str, list[str]] = {}
    all_team_stats = {}
    
    for grp, teams in GROUPS.items():
        standings = {t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0} for t in teams}
        matches = []
        pairs = [(teams[i], teams[j]) for i in range(len(teams)) for j in range(i + 1, len(teams))]

        for home, away in pairs:
            ph, pd, pa = predict_match(home, away)
            roll = random.random()
            if roll < ph:
                winner = home
                gh, ga = random.randint(1, 3), random.randint(0, 1)
            elif roll < ph + pd:
                winner = "Draw"
                g = random.randint(0, 2)
                gh = ga = g
            else:
                winner = away
                gh, ga = random.randint(0, 1), random.randint(1, 3)

            standings[home]["gf"] += gh
            standings[home]["ga"] += ga
            standings[away]["gf"] += ga
            standings[away]["ga"] += gh

            if winner == home: standings[home]["pts"] += 3
            elif winner == away: standings[away]["pts"] += 3
            else:
                standings[home]["pts"] += 1
                standings[away]["pts"] += 1

            for t in (home, away):
                standings[t]["gd"] = standings[t]["gf"] - standings[t]["ga"]

            matches.append({
                "home": home, "away": away, "gh": gh, "ga": ga, "winner": winner,
                "ph": ph, "pd": pd, "pa": pa,
            })

        sorted_teams = sorted(
            standings, key=lambda t: (standings[t]["pts"], standings[t]["gd"], standings[t]["gf"]), reverse=True,
        )
        results["groups"][grp] = {"standings": standings, "sorted": sorted_teams, "matches": matches}
        group_standings[grp] = sorted_teams
        for t in teams: all_team_stats[t] = standings[t]

    first_places = [group_standings[g][0] for g in GROUPS.keys()]
    second_places = [group_standings[g][1] for g in GROUPS.keys()]
    third_places = [group_standings[g][2] for g in GROUPS.keys()]

    sorted_thirds = sorted(
        third_places, key=lambda t: (all_team_stats[t]["pts"], all_team_stats[t]["gd"], all_team_stats[t]["gf"]), reverse=True
    )
    best_8_thirds = sorted_thirds[:8]
    results["best_thirds"] = best_8_thirds

    sorted_seconds = sorted(
        second_places, key=lambda t: (all_team_stats[t]["pts"], all_team_stats[t]["gd"], all_team_stats[t]["gf"]), reverse=True
    )
    seeded = first_places + sorted_seconds[:4]
    unseeded = sorted_seconds[4:] + best_8_thirds
    
    r32_pairs = list(zip(seeded, reversed(unseeded)))
    r32_winners = []
    for home, away in r32_pairs:
        w, ph, pd, pa = simulate_match(home, away, allow_draw=False)
        results["r32"].append({"home": home, "away": away, "winner": w, "ph": ph, "pa": pa})
        r32_winners.append(w)

    r16_pairs = [(r32_winners[i], r32_winners[i + 1]) for i in range(0, 16, 2)]
    r16_winners = []
    for home, away in r16_pairs:
        w, ph, pd, pa = simulate_match(home, away, allow_draw=False)
        results["r16"].append({"home": home, "away": away, "winner": w, "ph": ph, "pa": pa})
        r16_winners.append(w)

    qf_pairs = [(r16_winners[i], r16_winners[i + 1]) for i in range(0, 8, 2)]
    qf_winners = []
    for home, away in qf_pairs:
        w, ph, pd, pa = simulate_match(home, away, allow_draw=False)
        results["qf"].append({"home": home, "away": away, "winner": w, "ph": ph, "pa": pa})
        qf_winners.append(w)

    sf_pairs = [(qf_winners[0], qf_winners[1]), (qf_winners[2], qf_winners[3])]
    sf_winners = []
    for home, away in sf_pairs:
        w, ph, pd, pa = simulate_match(home, away, allow_draw=False)
        results["sf"].append({"home": home, "away": away, "winner": w, "ph": ph, "pa": pa})
        sf_winners.append(w)

    w, ph, pd, pa = simulate_match(sf_winners[0], sf_winners[1], allow_draw=False)
    results["final"].append({"home": sf_winners[0], "away": sf_winners[1], "winner": w, "ph": ph, "pa": pa})
    results["champion"] = w
    return results

# ---------------------------------------------------------------------------
# UI HELPER COMPONENTS
# ---------------------------------------------------------------------------
def render_phase_header(title: str) -> None:
    st.markdown(f'<div class="phase-header">{title}</div>', unsafe_allow_html=True)


def render_match_card(home: str, away: str, winner: str | None = None, ph: float | None = None, pa: float | None = None) -> None:
    home_won = winner == home
    away_won = winner == away
    home_style = "color:#3fb950; font-weight:700;" if home_won else "color:#e6edf3;"
    away_style = "color:#3fb950; font-weight:700;" if away_won else "color:#e6edf3;"
    home_badge = "  [W]" if home_won else ""
    away_badge = "  [W]" if away_won else ""
    ph_pct = int((ph or 0.5) * 100)
    pa_pct = int((pa or 0.5) * 100)

    st.markdown(dedent(f"""
    <div class="match-card">
        <div class="team-row">
            <span style="{home_style}">{home}{home_badge}</span>
            <span class="win-pct">{ph_pct}%</span>
        </div>
        <div class="vs-badge">VS</div>
        <div class="team-row">
            <span style="{away_style}">{away}{away_badge}</span>
            <span class="win-pct">{pa_pct}%</span>
        </div>
    </div>
    """), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SYMMETRICAL BRACKET RENDERER (Custom Knockout tab)
# ---------------------------------------------------------------------------
BRACKET_COLUMN_LABEL = {
    "Round of 32":    "Round of 32",
    "Round of 16":    "Round of 16",
    "Quarter-Finals": "Quarterfinals",
    "Semi-Finals":    "Semifinals",
    "Final":          "Final",
}


def _bracket_card_html(m: dict, use_abbr: bool, css_extra: str = "") -> str:
    home, away, winner = m["home"], m["away"], m["winner"]
    
    is_home_placeholder = str(home).startswith("GM")
    is_away_placeholder = str(away).startswith("GM")
    
    home_label = home if is_home_placeholder else team_display(home, use_abbr)
    away_label = away if is_away_placeholder else team_display(away, use_abbr)
    
    home_won = winner == home if (winner and not is_home_placeholder) else False
    away_won = winner == away if (winner and not is_away_placeholder) else False
    
    home_cls = "bracket-team-name bracket-winner" if home_won else "bracket-team-name"
    away_cls = "bracket-team-name bracket-winner" if away_won else "bracket-team-name"
    
    home_pct_cls = "bracket-pct bracket-winner" if home_won else "bracket-pct"
    away_pct_cls = "bracket-pct bracket-winner" if away_won else "bracket-pct"
    
    if "ph" in m and m["ph"] is not None and not is_home_placeholder:
        ph_pct = int(m["ph"] * 100)
        home_pct_html = f'<span class="{home_pct_cls}">{ph_pct}%</span>'
    else:
        home_pct_html = ''
        
    if "pa" in m and m["pa"] is not None and not is_away_placeholder:
        pa_pct = int(m["pa"] * 100)
        away_pct_html = f'<span class="{away_pct_cls}">{pa_pct}%</span>'
    else:
        away_pct_html = ''
        
    return dedent(f"""
    <div class="bracket-card {css_extra}">
        <div class="bracket-team-row">
            <span class="{home_cls}" title="{home if not is_home_placeholder else ''}">{home_label}</span>
            {home_pct_html}
        </div>
        <div class="bracket-team-row">
            <span class="{away_cls}" title="{away if not is_away_placeholder else ''}">{away_label}</span>
            {away_pct_html}
        </div>
    </div>
    """)


def render_symmetric_bracket(custom_result: dict, starting_round: str, use_abbr: bool = False, slot_selections: list = None) -> list:
    """
    Renders the symmetric bracket tree. If custom_result is None, it displays interactive
    selectors in the starting round using Streamlit columns. If custom_result is present, 
    it builds the raw HTML structure using the original CSS grid to ensure perfect horizontal symmetry.
    """
    played_rounds = [r["label"] for r in KNOCKOUT_ROUNDS if r["label"] in (custom_result or {})]
    
    if not custom_result:
        played_rounds = [starting_round]
        
    non_final_rounds = [r for r in played_rounds if r != "Final"]

    def split_half(label, is_result=True):
        if is_result:
            matches = custom_result[label]
            half = len(matches) // 2
            return matches[:half], matches[half:]
        else:
            n_slots = ROUND_BY_LABEL[starting_round]["slots"]
            mock_matches = []
            for i in range(0, n_slots, 2):
                mock_matches.append({
                    "home": slot_selections[i] if slot_selections else "TBD",
                    "away": slot_selections[i+1] if slot_selections else "TBD",
                    "winner": None, "ph": 0.5, "pa": 0.5
                })
            half = len(mock_matches) // 2
            return mock_matches[:half], mock_matches[half:]

    if custom_result:
        left_cols = [(label, split_half(label, True)[0]) for label in non_final_rounds]
        right_cols = [(label, split_half(label, True)[1]) for label in reversed(non_final_rounds)]
    else:
        left_cols = [(starting_round, split_half(starting_round, False)[0])]
        right_cols = [(starting_round, split_half(starting_round, False)[1])]

    all_teams = sorted(TEAM_DATA.keys())
    updated_slots = []

    # PRE-SIMULATION INTERACTIVE MODE (Uses native columns for selectors)
    if not custom_result:
        total_cols_count = len(left_cols) + 1 + len(right_cols)
        st_cols = st.columns(total_cols_count)
        current_col_idx = 0

        # Left Column Inputs
        for label, matches in left_cols:
            with st_cols[current_col_idx]:
                title = BRACKET_COLUMN_LABEL.get(label, label)
                st.markdown(f'<div class="bracket-col-title">{title}</div>', unsafe_allow_html=True)
                for m_idx, m in enumerate(matches):
                    st.markdown('<div class="bracket-card">', unsafe_allow_html=True)
                    t_a = st.selectbox("Home", all_teams, index=all_teams.index(m["home"]) if m["home"] in all_teams else 0, key=f"vis_slot_L_{m_idx}_A", label_visibility="collapsed")
                    st.markdown('<div style="text-align:center; font-size:10px; color:#8b949e; margin:2px 0;">VS</div>', unsafe_allow_html=True)
                    t_b = st.selectbox("Away", all_teams, index=all_teams.index(m["away"]) if m["away"] in all_teams else 1, key=f"vis_slot_L_{m_idx}_B", label_visibility="collapsed")
                    st.markdown('</div>', unsafe_allow_html=True)
                    updated_slots.extend([t_a, t_b])
            current_col_idx += 1

        # Center Column Placeholder
        with st_cols[current_col_idx]:
            st.markdown('<div class="bracket-col-title">Final</div>', unsafe_allow_html=True)
            st.markdown('<div class="bracket-card bracket-final-card" style="text-align:center; color:#8b949e; padding:20px 10px;">No qualified teams yet</div>', unsafe_allow_html=True)
        current_col_idx += 1

        # Right Column Inputs
        for label, matches in right_cols:
            with st_cols[current_col_idx]:
                title = BRACKET_COLUMN_LABEL.get(label, label)
                st.markdown(f'<div class="bracket-col-title">{title}</div>', unsafe_allow_html=True)
                for m_idx, m in enumerate(matches):
                    st.markdown('<div class="bracket-card">', unsafe_allow_html=True)
                    t_a = st.selectbox("Home", all_teams, index=all_teams.index(m["home"]) if m["home"] in all_teams else 2, key=f"vis_slot_R_{m_idx}_A", label_visibility="collapsed")
                    st.markdown('<div style="text-align:center; font-size:10px; color:#8b949e; margin:2px 0;">VS</div>', unsafe_allow_html=True)
                    t_b = st.selectbox("Away", all_teams, index=all_teams.index(m["away"]) if m["away"] in all_teams else 3, key=f"vis_slot_R_{m_idx}_B", label_visibility="collapsed")
                    st.markdown('</div>', unsafe_allow_html=True)
                    updated_slots.extend([t_a, t_b])
            current_col_idx += 1

        return updated_slots

    # POST-SIMULATION RESULTS MODE (Restores the exact original HTML tree and CSS symmetry)
    else:
        col_html_parts = []

        # Left Side Columns HTML
        for label, matches in left_cols:
            title = BRACKET_COLUMN_LABEL.get(label, label)
            cards = "".join(_bracket_card_html(m, use_abbr) for m in matches)
            col_html_parts.append(
                f'<div class="bracket-col"><div class="bracket-col-title">{title}</div>{cards}</div>'
            )

        # Center Column HTML (Final & Third Place)
        center_parts = []
        if "Final" in custom_result:
            final_card = _bracket_card_html(custom_result["Final"][0], use_abbr, "bracket-final-card")
            center_parts.append(
                f'<div><div class="bracket-col-title">Final</div>{final_card}</div>'
            )
        if "Third Place" in custom_result:
            third_card = _bracket_card_html(custom_result["Third Place"][0], use_abbr, "bracket-third-card")
            center_parts.append(
                f'<div><div class="bracket-sub-label">3rd Place Match</div>{third_card}</div>'
            )
        col_html_parts.append(f'<div class="bracket-center-col">{"".join(center_parts)}</div>')

        # Right Side Columns HTML
        for label, matches in right_cols:
            title = BRACKET_COLUMN_LABEL.get(label, label)
            cards = "".join(_bracket_card_html(m, use_abbr) for m in matches)
            col_html_parts.append(
                f'<div class="bracket-col"><div class="bracket-col-title">{title}</div>{cards}</div>'
            )

        # Wrap everything using your original CSS layout configuration classes
        grid_html = (
            '<div class="bracket-scroll"><div class="bracket-grid">'
            + "".join(col_html_parts)
            + "</div></div>"
        )
        st.markdown(grid_html, unsafe_allow_html=True)
        return []
# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Settings")

    model_status = "Loaded" if model else "Unavailable — using ELO fallback"
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-label">Prediction Model</div>
        <div class="stat-value">{model_status}</div>
    </div>
    """, unsafe_allow_html=True)

    if model:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-label">Algorithm</div>
            <div class="stat-value">Random Forest</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Input Features</div>
            <div class="stat-value">8 variables</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Simulation Parameters")
    n_sims = st.slider("Number of simulations", min_value=1, max_value=1000, value=100, step=10)
    seed = st.number_input("Random seed (0 = unseeded)", min_value=0, max_value=9999, value=0)

    st.markdown("---")
    st.markdown("### Head-to-Head Prediction")
    teams_list = sorted(TEAM_DATA.keys())
    home_team = st.selectbox("Home team", teams_list, index=teams_list.index("Spain"))
    away_team = st.selectbox("Away team", teams_list, index=teams_list.index("Brazil"))
    is_knockout = st.checkbox("Knockout match (no draws allowed)", value=False)

    if home_team != away_team:
        ph, pd, pa = predict_match(home_team, away_team)
        if is_knockout:
            total = ph + pa if (ph + pa) > 0 else 1.0
            ph_display = ph / total
            pa_display = pa / total
            pd_display = 0.0
            label_suffix = " (adjusted, no draws)"
        else:
            ph_display, pd_display, pa_display = ph, pd, pa
            label_suffix = ""
        st.markdown(f"""
        <div style="margin-top:10px;">
            <div class="stat-box">
                <div class="stat-label">{home_team} win probability{label_suffix}</div>
                <div class="stat-value">{ph_display * 100:.1f}%</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Draw probability{label_suffix}</div>
                <div class="stat-value">{pd_display * 100:.1f}%</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">{away_team} win probability{label_suffix}</div>
                <div class="stat-value">{pa_display * 100:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------------------------
st.markdown("# World Cup 2026 Predictor")
st.markdown("A statistical simulation of the FIFA World Cup 2026 tournament driven by a pre-trained Random Forest classifier.")

tab1, tab2, tab3, tab4 = st.tabs(["Bracket Simulation", "Probability Analysis", "Group Stage", "Custom Knockout Simulation"])

# =============================================================================
# TAB 1 — BRACKET SIMULATION
# =============================================================================
with tab1:
    col_btn, col_info = st.columns([2, 3])
    with col_btn: run_sim = st.button("Run Simulation", use_container_width=True)
    with col_info: st.markdown(f"<p style='color:#8b949e; padding-top:12px;'>Results are aggregated across {n_sims} independent simulation runs.</p>", unsafe_allow_html=True)

    if run_sim or "wc_results" in st.session_state:
        if run_sim:
            with st.spinner("Running tournament simulation..."):
                if seed > 0: random.seed(seed)
                result = simulate_world_cup()
                st.session_state["wc_results"] = result

                champion_counts = {t: 0 for t in TEAM_DATA}
                final_counts    = {t: 0 for t in TEAM_DATA}
                sf_counts       = {t: 0 for t in TEAM_DATA}

                champion_counts[result["champion"]] += 1
                for m in result["final"]:
                    final_counts[m["home"]] += 1; final_counts[m["away"]] += 1
                for m in result["sf"]:
                    sf_counts[m["home"]] += 1; sf_counts[m["away"]] += 1

                for _ in range(n_sims - 1):
                    r = simulate_world_cup()
                    champion_counts[r["champion"]] += 1
                    for m in r["final"]:
                        final_counts[m["home"]] += 1; final_counts[m["away"]] += 1
                    for m in r["sf"]:
                        sf_counts[m["home"]] += 1; sf_counts[m["away"]] += 1

                st.session_state["champ_counts"] = champion_counts
                st.session_state["final_counts"] = final_counts
                st.session_state["sf_counts"]    = sf_counts
                st.session_state["n_sims"]       = n_sims

        result = st.session_state["wc_results"]
        st.markdown(f'<div class="champion-card">WORLD CUP CHAMPION<br>{result["champion"]}</div>', unsafe_allow_html=True)

        render_phase_header("FINAL")
        _, col_center, _ = st.columns([1, 2, 1])
        with col_center: m = result["final"][0]; render_match_card(m["home"], m["away"], m["winner"], m["ph"], m["pa"])

        for phase, key in [("SEMI-FINALS", "sf"), ("QUARTER-FINALS", "qf"), ("ROUND OF 16", "r16"), ("ROUND OF 32", "r32")]:
            st.markdown("<br>", unsafe_allow_html=True)
            render_phase_header(phase)
            cols = st.columns(4 if len(result[key]) >= 4 else len(result[key]))
            for i, m in enumerate(result[key]):
                with cols[i % len(cols)]: render_match_card(m["home"], m["away"], m["winner"], m["ph"], m["pa"])
    else:
        st.markdown('<div style="text-align:center; padding:60px; color:#8b949e;"><div style="font-size:20px; font-weight:600;">Press <strong>Run Simulation</strong> to begin.</div></div>', unsafe_allow_html=True)

# =============================================================================
# TAB 2 — PROBABILITY ANALYSIS
# =============================================================================
with tab2:
    if "champ_counts" in st.session_state:
        cc = st.session_state["champ_counts"]; fc = st.session_state["final_counts"]; sc = st.session_state["sf_counts"]; ns = st.session_state["n_sims"]
        st.markdown(f"### Results across {ns} simulations")
        st.markdown("#### Championship Probability")
        for team, count in sorted(cc.items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = count / ns * 100
            st.markdown(f'<div style="display:flex; align-items:center; gap:12px; margin:6px 0;"><span style="width:160px; font-size:14px;">{team}</span><div style="flex:1; height:20px; background:#21262d; border-radius:4px; overflow:hidden;"><div style="width:{min(pct * 3, 100):.0f}%; height:100%; background:linear-gradient(90deg,#1f6feb,#3fb950);"></div></div><span style="width:50px; text-align:right; font-weight:700; color:#3fb950;">{pct:.1f}%</span></div>', unsafe_allow_html=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Final Appearance Probability")
            for team, count in sorted(fc.items(), key=lambda x: x[1], reverse=True)[:8]:
                pct = count / ns * 100
                st.markdown(f'<div style="display:flex; align-items:center; gap:8px; margin:4px 0; font-size:13px;"><span style="width:140px;">{team}</span><div style="flex:1; height:14px; background:#21262d; border-radius:3px; overflow:hidden;"><div style="width:{min(pct * 1.5, 100):.0f}%; height:100%; background:#388bfd;"></div></div><span style="width:44px; text-align:right; color:#58a6ff;">{pct:.0f}%</span></div>', unsafe_allow_html=True)
        with c2:
            st.markdown("#### Semi-Final Appearance Probability")
            for team, count in sorted(sc.items(), key=lambda x: x[1], reverse=True)[:8]:
                pct = count / ns * 100
                st.markdown(f'<div style="display:flex; align-items:center; gap:8px; margin:4px 0; font-size:13px;"><span style="width:140px;">{team}</span><div style="flex:1; height:14px; background:#21262d; border-radius:3px; overflow:hidden;"><div style="width:{min(pct, 100):.0f}%; height:100%; background:#f78166;"></div></div><span style="width:44px; text-align:right; color:#f78166;">{pct:.0f}%</span></div>', unsafe_allow_html=True)
    else:
        st.info("Run a simulation in the Bracket tab to populate this section.")

# =============================================================================
# TAB 3 — GROUP STAGE
# =============================================================================
with tab3:
    if "wc_results" in st.session_state:
        result = st.session_state["wc_results"]; st.markdown("### Group Stage Results")
        group_keys = list(GROUPS.keys())
        for row_start in range(0, len(group_keys), 2):
            cols = st.columns(2)
            for ci, grp in enumerate(group_keys[row_start:row_start + 2]):
                with cols[ci]:
                    gdata = result["groups"][grp]; st.markdown(f'<div class="group-title">Group {grp}</div>', unsafe_allow_html=True)
                    rows_html = ""
                    for rank, team in enumerate(gdata["sorted"]):
                        s = gdata["standings"][team]; qualified = (rank < 2) or (team in result["best_thirds"])
                        rows_html += f'<tr style="{"background:#0d3526;" if qualified else ""}"><td style="padding:5px 2px;color:#8b949e;">{rank + 1}</td><td style="padding:5px 6px;">{team}{" *" if qualified else ""}</td><td style="padding:5px;text-align:center;font-weight:700;color:#3fb950;">{s["pts"]}</td><td style="padding:5px;text-align:center;">{s["gf"]}</td><td style="padding:5px;text-align:center;">{s["ga"]}</td><td style="padding:5px;text-align:center;color:{"#3fb950" if s["gd"]>=0 else "#f85149"};">{"++" if s["gd"]>0 else ""}{s["gd"]}</td></tr>'
                    st.markdown('<div class="group-table"><table style="width:100%;font-size:12px;border-collapse:collapse;"><tr style="color:#8b949e;border-bottom:1px solid #30363d;"><th style="text-align:left;padding:4px 2px;">#</th><th style="text-align:left;padding:4px 6px;">Team</th><th style="padding:4px;">Pts</th><th style="padding:4px;">GF</th><th style="padding:4px;">GA</th><th style="padding:4px;">GD</th></tr>'+rows_html+'</table></div>', unsafe_allow_html=True)
    else:
        st.info("Run a simulation in the Bracket tab to populate this section.")

# =============================================================================
# TAB 4 — CUSTOM KNOCKOUT SIMULATION
# =============================================================================
# TAB 4: CUSTOM KNOCKOUT SIMULATION (Interactive Tree Layout)
# TAB 4: CUSTOM KNOCKOUT SIMULATION (Interactive Tree Layout)
with tab4:
    st.markdown("### Custom Knockout Simulation")
    st.markdown(
        "Select teams directly inside the dropdown menus on the starting columns of the bracket layout below. "
        "The tournament will automatically simulate the remaining rounds up to the Grand Final."
    )

    all_teams = sorted(TEAM_DATA.keys())

    starting_round = st.selectbox(
        "Tournament starting round",
        ROUND_LABELS,
        index=0,
        help="The interactive bracket structure will adapt dynamically based on your chosen starting round.",
    )
    
    n_slots = ROUND_BY_LABEL[starting_round]["slots"]
    
    default_teams = (all_teams * ((n_slots // len(all_teams)) + 1))[:n_slots]

    if "temp_slots" not in st.session_state or len(st.session_state["temp_slots"]) != n_slots:
        st.session_state["temp_slots"] = default_teams

    st.markdown("<br>", unsafe_allow_html=True)
    
    c_seed, c_style = st.columns([2, 2])
    with c_seed:
        custom_seed = st.number_input("Random Seed (0 = unseeded simulation)", min_value=0, max_value=9999, value=0, key="custom_bracket_seed_input")
    with c_style:
        display_mode = st.radio("Display Label Mode (Simulation Results Only)", ["Full names", "3-letter ISO Codes"], index=0, horizontal=True)
        use_abbr = display_mode.startswith("3-letter")

    st.markdown("<br>", unsafe_allow_html=True)

    slot_selections = []
    
    if "custom_result" not in st.session_state:
        st.markdown("#### 1. Setup your match fixtures on the tree diagram:")
        slot_selections = render_symmetric_bracket(
            custom_result=None,
            starting_round=starting_round,
            use_abbr=use_abbr,
            slot_selections=st.session_state["temp_slots"]
        )
        st.session_state["temp_slots"] = slot_selections
        
        st.markdown("<br>", unsafe_allow_html=True)
        run_custom = st.button("🔥 Simulate Complete Tournament Tree", use_container_width=True)
        
        if run_custom:
            duplicates = [t for t in set(slot_selections) if slot_selections.count(t) > 1]
            if duplicates:
                st.error(f"⚠️ Duplicate teams detected: {', '.join(sorted(set(duplicates)))}. Every single dropdown slot must have a unique country.")
            else:
                if custom_seed > 0:
                    random.seed(custom_seed)
                custom_result = simulate_custom_bracket(starting_round, slot_selections)
                st.session_state["custom_result"] = custom_result
                st.session_state["custom_starting_round"] = starting_round
                st.rerun()

    else:
        custom_result = st.session_state["custom_result"]
        champ = custom_result["champion"]

        st.markdown(f"""
        <div class="champion-card">
            🏆 BRACKET WINNER CHAMPION<br>
            {champ}
        </div>
        """, unsafe_allow_html=True)

        if "Third Place" in custom_result:
            tp = custom_result["Third Place"][0]
            st.markdown(f"<p style='text-align:center; color:#b8860b; font-weight:700; font-size:14px; margin-top:-8px;'>🥉 3rd Place Winner: {tp['winner']}</p>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        render_symmetric_bracket(
            custom_result=custom_result,
            starting_round=st.session_state.get("custom_starting_round", starting_round),
            use_abbr=use_abbr
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset Bracket and Design a New Tree"):
            del st.session_state["custom_result"]
            st.rerun()
# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; color:#484f58; font-size:12px; margin-top:40px; padding:20px; border-top:1px solid #21262d;">
    World Cup 2026 Predictor &nbsp;|&nbsp; Random Forest Classifier &nbsp;|&nbsp; Features: ELO differential, recent goal average, head-to-head record
</div>
""", unsafe_allow_html=True)