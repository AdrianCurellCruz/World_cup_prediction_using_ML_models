
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

import random
import warnings
from pathlib import Path

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
# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "best_rf_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None


model = load_model()

# ---------------------------------------------------------------------------
# PREDICTION LOGIC
# ---------------------------------------------------------------------------
def predict_match(
    home: str,
    away: str,
    h2h_win_pct: float = 0.5,
    h2h_goal_diff: float = 0.0,
    h2h_count: int = 5,
):
    """
    Return (prob_home_win, prob_draw, prob_away_win) for a given fixture.
    Falls back to an ELO-based estimate when the model file is unavailable.
    """
    hd = TEAM_DATA[home]
    ad = TEAM_DATA[away]
    elo_diff = hd["elo"] - ad["elo"]

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
        # classes order: 0=draw, 1=home_win, 2=away_win
        return float(proba[1]), float(proba[0]), float(proba[2])

    # ELO-based fallback
    base = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
    draw_prob = 0.25
    return base * (1 - draw_prob), draw_prob, (1 - base) * (1 - draw_prob)


def simulate_match(home: str, away: str, allow_draw: bool = True):
    """
    Randomly sample a match outcome according to predicted probabilities.
    Returns (winner, prob_home, prob_draw, prob_away).
    In knockout rounds (allow_draw=False) the draw probability is redistributed.
    """
    ph, pd, pa = predict_match(home, away)
    if not allow_draw:
        total = ph + pa
        ph, pa = ph / total, pa / total
        pd = 0.0

    roll = random.random()
    if roll < ph:
        return home, ph, pd, pa
    if roll < ph + pd:
        return "Draw", ph, pd, pa
    return away, ph, pd, pa

# ---------------------------------------------------------------------------
# FULL TOURNAMENT SIMULATION
# ---------------------------------------------------------------------------
def simulate_world_cup() -> dict:
    """
    Simulate a complete World Cup tournament from group stage through final.
    Returns a dictionary containing all results and the predicted champion.
    """
    results = {
        "groups": {},
        "r32": [],
        "r16": [],
        "qf": [],
        "sf": [],
        "final": [],
        "champion": None,
        "best_thirds": [],
    }

    # -- GROUP STAGE --
    group_standings: dict[str, list[str]] = {}
    all_team_stats = {}
    
    for grp, teams in GROUPS.items():
        standings = {t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0} for t in teams}
        matches = []
        pairs = [
            (teams[i], teams[j])
            for i in range(len(teams))
            for j in range(i + 1, len(teams))
        ]

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

            if winner == home:
                standings[home]["pts"] += 3
            elif winner == away:
                standings[away]["pts"] += 3
            else:
                standings[home]["pts"] += 1
                standings[away]["pts"] += 1

            for t in (home, away):
                standings[t]["gd"] = standings[t]["gf"] - standings[t]["ga"]

            matches.append({
                "home": home, "away": away,
                "gh": gh, "ga": ga,
                "winner": winner,
                "ph": ph, "pd": pd, "pa": pa,
            })

        sorted_teams = sorted(
            standings,
            key=lambda t: (standings[t]["pts"], standings[t]["gd"], standings[t]["gf"]),
            reverse=True,
        )
        results["groups"][grp] = {
            "standings": standings,
            "sorted": sorted_teams,
            "matches": matches,
        }
        group_standings[grp] = sorted_teams
        for t in teams:
            all_team_stats[t] = standings[t]

    # -- QUALIFICATION LOGIC --
    first_places = [group_standings[g][0] for g in GROUPS.keys()]
    second_places = [group_standings[g][1] for g in GROUPS.keys()]
    third_places = [group_standings[g][2] for g in GROUPS.keys()]

    # Sort thirds to find best 8 across all groups
    sorted_thirds = sorted(
        third_places,
        key=lambda t: (all_team_stats[t]["pts"], all_team_stats[t]["gd"], all_team_stats[t]["gf"]),
        reverse=True
    )
    best_8_thirds = sorted_thirds[:8]
    results["best_thirds"] = best_8_thirds

    # -- ROUND OF 32 --
    # Seeded: 12 group winners + 4 best 2nd places
    sorted_seconds = sorted(
        second_places,
        key=lambda t: (all_team_stats[t]["pts"], all_team_stats[t]["gd"], all_team_stats[t]["gf"]),
        reverse=True
    )
    seeded = first_places + sorted_seconds[:4]
    unseeded = sorted_seconds[4:] + best_8_thirds
    
    # Round of 32 Matches setup (1v32, 2v31...) simplified seeding alignment
    r32_pairs = list(zip(seeded, reversed(unseeded)))
    
    r32_winners = []
    for home, away in r32_pairs:
        w, ph, pd, pa = simulate_match(home, away, allow_draw=False)
        results["r32"].append({"home": home, "away": away, "winner": w, "ph": ph, "pa": pa})
        r32_winners.append(w)

    # -- ROUND OF 16 --
    r16_pairs = [(r32_winners[i], r32_winners[i + 1]) for i in range(0, 16, 2)]
    r16_winners = []
    for home, away in r16_pairs:
        w, ph, pd, pa = simulate_match(home, away, allow_draw=False)
        results["r16"].append({"home": home, "away": away, "winner": w, "ph": ph, "pa": pa})
        r16_winners.append(w)

    # -- QUARTER-FINALS --
    qf_pairs = [(r16_winners[i], r16_winners[i + 1]) for i in range(0, 8, 2)]
    qf_winners = []
    for home, away in qf_pairs:
        w, ph, pd, pa = simulate_match(home, away, allow_draw=False)
        results["qf"].append({"home": home, "away": away, "winner": w, "ph": ph, "pa": pa})
        qf_winners.append(w)

    # -- SEMI-FINALS --
    sf_pairs = [(qf_winners[0], qf_winners[1]), (qf_winners[2], qf_winners[3])]
    sf_winners = []
    for home, away in sf_pairs:
        w, ph, pd, pa = simulate_match(home, away, allow_draw=False)
        results["sf"].append({"home": home, "away": away, "winner": w, "ph": ph, "pa": pa})
        sf_winners.append(w)

    # -- FINAL --
    w, ph, pd, pa = simulate_match(sf_winners[0], sf_winners[1], allow_draw=False)
    results["final"].append({
        "home": sf_winners[0], "away": sf_winners[1],
        "winner": w, "ph": ph, "pa": pa,
    })
    results["champion"] = w
    return results

# ---------------------------------------------------------------------------
# UI HELPER COMPONENTS
# ---------------------------------------------------------------------------
def render_phase_header(title: str) -> None:
    st.markdown(f'<div class="phase-header">{title}</div>', unsafe_allow_html=True)


def render_match_card(
    home: str,
    away: str,
    winner: str | None = None,
    ph: float | None = None,
    pa: float | None = None,
) -> None:
    home_won = winner == home
    away_won = winner == away
    home_style = "color:#3fb950; font-weight:700;" if home_won else "color:#e6edf3;"
    away_style = "color:#3fb950; font-weight:700;" if away_won else "color:#e6edf3;"
    home_badge = "  [W]" if home_won else ""
    away_badge = "  [W]" if away_won else ""
    ph_pct = int((ph or 0.5) * 100)
    pa_pct = int((pa or 0.5) * 100)

    st.markdown(f"""
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
    """, unsafe_allow_html=True)

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

    if home_team != away_team:
        ph, pd, pa = predict_match(home_team, away_team)
        st.markdown(f"""
        <div style="margin-top:10px;">
            <div class="stat-box">
                <div class="stat-label">{home_team} win probability</div>
                <div class="stat-value">{ph * 100:.1f}%</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Draw probability</div>
                <div class="stat-value">{pd * 100:.1f}%</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">{away_team} win probability</div>
                <div class="stat-value">{pa * 100:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------------------------
st.markdown("# World Cup 2026 Predictor")
st.markdown(
    "A statistical simulation of the FIFA World Cup 2026 tournament "
    "driven by a pre-trained Random Forest classifier."
)

tab1, tab2, tab3 = st.tabs(["Bracket Simulation", "Probability Analysis", "Group Stage"])

# =============================================================================
# TAB 1 — BRACKET
# =============================================================================
with tab1:
    col_btn, col_info = st.columns([2, 3])
    with col_btn:
        run_sim = st.button("Run Simulation", use_container_width=True)
    with col_info:
        st.markdown(
            f"<p style='color:#8b949e; padding-top:12px;'>"
            f"Results are aggregated across {n_sims} independent simulation runs."
            f"</p>",
            unsafe_allow_html=True,
        )

    if run_sim or "wc_results" in st.session_state:
        if run_sim:
            with st.spinner("Running tournament simulation..."):
                if seed > 0:
                    random.seed(seed)

                # Single simulation for the visual bracket
                result = simulate_world_cup()
                st.session_state["wc_results"] = result

                # Aggregate statistics over n_sims runs
                champion_counts = {t: 0 for t in TEAM_DATA}
                final_counts    = {t: 0 for t in TEAM_DATA}
                sf_counts       = {t: 0 for t in TEAM_DATA}

                for _ in range(n_sims):
                    r = simulate_world_cup()
                    champion_counts[r["champion"]] += 1
                    for m in r["final"]:
                        final_counts[m["home"]] += 1
                        final_counts[m["away"]] += 1
                    for m in r["sf"]:
                        sf_counts[m["home"]] += 1
                        sf_counts[m["away"]] += 1

                st.session_state["champ_counts"] = champion_counts
                st.session_state["final_counts"] = final_counts
                st.session_state["sf_counts"]    = sf_counts
                st.session_state["n_sims"]       = n_sims

        result = st.session_state["wc_results"]
        champ  = result["champion"]

        # Champion banner
        st.markdown(f"""
        <div class="champion-card">
            WORLD CUP CHAMPION<br>
            {champ}
        </div>
        """, unsafe_allow_html=True)

        # Final
        render_phase_header("FINAL")
        _, col_center, _ = st.columns([1, 2, 1])
        with col_center:
            m = result["final"][0]
            render_match_card(m["home"], m["away"], m["winner"], m["ph"], m["pa"])

        # Semi-finals
        st.markdown("<br>", unsafe_allow_html=True)
        render_phase_header("SEMI-FINALS")
        cols = st.columns(2)
        for i, m in enumerate(result["sf"]):
            with cols[i]:
                render_match_card(m["home"], m["away"], m["winner"], m["ph"], m["pa"])

        # Quarter-finals
        st.markdown("<br>", unsafe_allow_html=True)
        render_phase_header("QUARTER-FINALS")
        cols = st.columns(4)
        for i, m in enumerate(result["qf"]):
            with cols[i]:
                render_match_card(m["home"], m["away"], m["winner"], m["ph"], m["pa"])

        # Round of 16
        st.markdown("<br>", unsafe_allow_html=True)
        render_phase_header("ROUND OF 16")
        cols = st.columns(4)
        for i, m in enumerate(result["r16"]):
            with cols[i % 4]:
                render_match_card(m["home"], m["away"], m["winner"], m["ph"], m["pa"])
                
        # Round of 32
        st.markdown("<br>", unsafe_allow_html=True)
        render_phase_header("ROUND OF 32")
        cols = st.columns(4)
        for i, m in enumerate(result["r32"]):
            with cols[i % 4]:
                render_match_card(m["home"], m["away"], m["winner"], m["ph"], m["pa"])

    else:
        st.markdown("""
        <div style="text-align:center; padding:60px; color:#8b949e;">
            <div style="font-size:20px; margin-top:16px; font-weight:600;">
                Press <strong>Run Simulation</strong> to begin.
            </div>
            <div style="font-size:14px; margin-top:8px;">
                The model will evaluate all 104 tournament fixtures.
            </div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# TAB 2 — PROBABILITY ANALYSIS
# =============================================================================
with tab2:
    if "champ_counts" in st.session_state:
        cc = st.session_state["champ_counts"]
        fc = st.session_state["final_counts"]
        sc = st.session_state["sf_counts"]
        ns = st.session_state["n_sims"]

        st.markdown(f"### Results across {ns} simulations")

        # Championship probability — top 10
        top10 = sorted(cc.items(), key=lambda x: x[1], reverse=True)[:10]
        st.markdown("#### Championship Probability")
        for team, count in top10:
            pct = count / ns * 100
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:12px; margin:6px 0;">
                <span style="width:160px; font-size:14px;">{team}</span>
                <div style="flex:1; height:20px; background:#21262d; border-radius:4px; overflow:hidden;">
                    <div style="width:{min(pct * 3, 100):.0f}%; height:100%;
                                background:linear-gradient(90deg,#1f6feb,#3fb950);"></div>
                </div>
                <span style="width:50px; text-align:right; font-weight:700;
                             color:#3fb950;">{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Final Appearance Probability")
            for team, count in sorted(fc.items(), key=lambda x: x[1], reverse=True)[:8]:
                pct = count / ns * 100
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:8px; margin:4px 0; font-size:13px;">
                    <span style="width:140px;">{team}</span>
                    <div style="flex:1; height:14px; background:#21262d; border-radius:3px; overflow:hidden;">
                        <div style="width:{min(pct * 1.5, 100):.0f}%; height:100%; background:#388bfd;"></div>
                    </div>
                    <span style="width:44px; text-align:right; color:#58a6ff;">{pct:.0f}%</span>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("#### Semi-Final Appearance Probability")
            for team, count in sorted(sc.items(), key=lambda x: x[1], reverse=True)[:8]:
                pct = count / ns * 100
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:8px; margin:4px 0; font-size:13px;">
                    <span style="width:140px;">{team}</span>
                    <div style="flex:1; height:14px; background:#21262d; border-radius:3px; overflow:hidden;">
                        <div style="width:{min(pct, 100):.0f}%; height:100%; background:#f78166;"></div>
                    </div>
                    <span style="width:44px; text-align:right; color:#f78166;">{pct:.0f}%</span>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.info("Run a simulation in the Bracket tab to populate this section.")

# =============================================================================
# TAB 3 — GROUP STAGE
# =============================================================================
with tab3:
    if "wc_results" in st.session_state:
        result = st.session_state["wc_results"]
        st.markdown("### Group Stage Results")

        group_keys = list(GROUPS.keys())
        for row_start in range(0, len(group_keys), 2):
            cols = st.columns(2)
            for ci, grp in enumerate(group_keys[row_start:row_start + 2]):
                with cols[ci]:
                    gdata = result["groups"][grp]
                    st.markdown(f'<div class="group-title">Group {grp}</div>', unsafe_allow_html=True)

                    rows_html = ""
                    for rank, team in enumerate(gdata["sorted"]):
                        s = gdata["standings"][team]
                        pts = s["pts"]
                        gf  = s["gf"]
                        ga  = s["ga"]
                        gd  = s["gd"]
                        
                        # Qualification check: Top 2 + best 8 thirds
                        qualified      = (rank < 2) or (team in result["best_thirds"])
                        
                        row_bg         = "background:#0d3526;" if qualified else ""
                        qualifier_mark = " *" if qualified else ""
                        gd_sign        = "+" if gd > 0 else ""
                        gd_color       = "#3fb950" if gd >= 0 else "#f85149"
                        rows_html += (
                            f'<tr style="{row_bg}">'
                            f'<td style="padding:5px 2px;color:#8b949e;">{rank + 1}</td>'
                            f'<td style="padding:5px 6px;">{team}{qualifier_mark}</td>'
                            f'<td style="padding:5px;text-align:center;font-weight:700;color:#3fb950;">{pts}</td>'
                            f'<td style="padding:5px;text-align:center;">{gf}</td>'
                            f'<td style="padding:5px;text-align:center;">{ga}</td>'
                            f'<td style="padding:5px;text-align:center;color:{gd_color};">{gd_sign}{gd}</td>'
                            f'</tr>'
                        )

                    standings_html = (
                        '<table style="width:100%;font-size:12px;border-collapse:collapse;">'
                        '<tr style="color:#8b949e;border-bottom:1px solid #30363d;">'
                        '<th style="text-align:left;padding:4px 2px;">#</th>'
                        '<th style="text-align:left;padding:4px 6px;">Team</th>'
                        '<th style="padding:4px;">Pts</th>'
                        '<th style="padding:4px;">GF</th>'
                        '<th style="padding:4px;">GA</th>'
                        '<th style="padding:4px;">GD</th>'
                        '</tr>'
                        + rows_html +
                        '</table>'
                        '<p style="font-size:11px;color:#8b949e;margin:6px 0 0;">* qualified (top 2 or best 3rd)</p>'
                    )
                    st.markdown(
                        '<div class="group-table">' + standings_html + '</div>',
                        unsafe_allow_html=True,
                    )

                    with st.expander(f"Group {grp} — all fixtures"):
                        for m in gdata["matches"]:
                            w            = m["winner"]
                            home_name    = m["home"]
                            away_name    = m["away"]
                            gh_score     = m["gh"]
                            ga_score     = m["ga"]
                            result_label = "H" if w == home_name else ("D" if w == "Draw" else "A")
                            st.markdown(
                                f'<div style="font-size:12px;padding:5px 0;border-bottom:1px solid #21262d;">'
                                f'{home_name} <strong>{gh_score} - {ga_score}</strong> {away_name}'
                                f'<span style="color:#8b949e;margin-left:8px;">[{result_label}]</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
    else:
        st.info("Run a simulation in the Bracket tab to populate this section.")

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; color:#484f58; font-size:12px; margin-top:40px; padding:20px;
            border-top:1px solid #21262d;">
    World Cup 2026 Predictor &nbsp;|&nbsp; Random Forest Classifier &nbsp;|&nbsp;
    Features: ELO differential, recent goal average, head-to-head record
</div>
""", unsafe_allow_html=True)
