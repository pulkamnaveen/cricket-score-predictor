import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Cricket Score Predictor",
    page_icon="🏏",
    layout="centered"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    /* Warm, clean background */
    .stApp {
        background: #1b2838;
    }

    /* Header */
    .hero-banner {
        background: #276749;
        border-radius: 10px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .hero-banner h1 {
        color: #fff;
        font-size: 1.9rem;
        margin: 0;
        font-weight: 600;
    }
    .hero-banner p {
        color: #c6e4d0;
        font-size: 0.95rem;
        margin: 0.4rem 0 0 0;
        font-weight: 400;
    }

    /* Section headings */
    .section-title {
        color: #81c9b4;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    /* Prediction result card */
    .prediction-box {
        background: #276749;
        border-radius: 10px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    .prediction-label {
        color: #c6e4d0;
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
    }
    .prediction-score {
        color: #fff;
        font-size: 3rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }

    /* Stats underneath */
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    .stat-item {
        text-align: center;
        min-width: 60px;
    }
    .stat-value {
        color: #fff;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .stat-label {
        color: #a3d5c1;
        font-size: 0.72rem;
    }

    /* Match info strip */
    .match-info {
        background: #223344;
        border-radius: 8px;
        padding: 0.7rem 1.2rem;
        text-align: center;
        margin-bottom: 0.8rem;
    }
    .match-vs {
        color: #81c9b4;
        font-size: 1rem;
        font-weight: 500;
    }

    /* Predict button */
    div.stButton > button {
        background: #d4a03c;
        color: #1b2838;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 2rem;
        width: 100%;
        transition: background 0.2s;
    }
    div.stButton > button:hover {
        background: #e2b44e;
    }

    /* Input labels */
    .stSelectbox label, .stNumberInput label {
        color: #ccc !important;
        font-weight: 400 !important;
    }

    /* Score range bar */
    .range-box {
        background: #223344;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        text-align: center;
        margin-top: 0.8rem;
    }
    .range-text {
        color: #fff;
        font-size: 1.3rem;
        font-weight: 600;
    }
    .range-sub {
        color: #a3d5c1;
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }

    /* Win probability */
    .win-prob-container {
        background: #223344;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-top: 0.8rem;
    }
    .win-prob-title {
        color: #a3d5c1;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .win-bar-bg {
        background: #c0392b;
        border-radius: 6px;
        height: 32px;
        width: 100%;
        overflow: hidden;
        position: relative;
    }
    .win-bar-fill {
        background: #276749;
        height: 100%;
        border-radius: 6px 0 0 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: width 0.5s;
    }
    .win-bar-labels {
        display: flex;
        justify-content: space-between;
        margin-top: 0.4rem;
    }
    .win-team {
        color: #ccc;
        font-size: 0.78rem;
    }
    .win-pct {
        font-weight: 600;
    }

    /* What-if simulator */
    .whatif-title {
        color: #81c9b4;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .whatif-result {
        background: #223344;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    .whatif-score {
        color: #fff;
        font-size: 1.8rem;
        font-weight: 600;
    }
    .whatif-diff {
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }
    .whatif-up { color: #2ecc71; }
    .whatif-down { color: #e74c3c; }
    .whatif-same { color: #95a5a6; }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #ccc;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Hero Banner ----------------
st.markdown("""
<div class="hero-banner">
    <h1>🏏 Cricket Score Predictor</h1>
    <p>Predict the final T20 score based on the current match situation</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Helpers for overs ----------------

def next_ball(ov):
    o = int(ov)
    b = int(round((ov - o) * 10))
    b += 1
    if b > 6:
        o += 1
        b = 1
    return float(f"{o}.{b}")


def prev_ball(ov):
    o = int(ov)
    b = int(round((ov - o) * 10))
    b -= 1
    if b < 1:
        o -= 1
        b = 6
    if o < 0:
        return 0.1
    return float(f"{o}.{b}")


def balls_from_overs(ov):
    o = int(ov)
    b = int(round((ov - o) * 10))
    return o * 6 + b


# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    with open("xgb_pipeline.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    pipe = load_model()
except Exception as e:
    st.error("⚠️ Model load failed")
    st.exception(e)
    st.stop()

# ---------------- Data ----------------
teams = ['Pakistan','India','New Zealand','Sri Lanka','South Africa','West Indies',
         'England','Bangladesh','Zimbabwe','Australia','Ireland','Hong Kong',
         'Netherlands','United Arab Emirates','Malaysia','Nigeria','Uganda',
         'Bahrain','Scotland','Nepal']

cities = ['Dubai','Wellington','Cape Town','Kuala Lumpur','Hamilton','Christchurch',
          "St George's",'Pallekele','Edinburgh','Melbourne','Chattogram','Harare',
          'Dhaka','Bangalore','Colombo','Basseterre','Cardiff','Johannesburg',
          'Kigali City','Rajkot','Sydney','Napier','Auckland','Kandy','Canberra',
          'Manchester','Bridgetown','Adelaide','Perth','Mumbai','Hambantota',
          'Al Amarat','Hobart','Abu Dhabi','Chester-le-Street','Trinidad','Guyana',
          'Nottingham','Sharjah','Tarouba','Southampton','St Lucia','Mirpur',
          'Glasgow','London','Lahore','Brisbane','Mount Maunganui','Barbados',
          'Karachi','Kolkata','Sylhet','Delhi','Chandigarh','Rawalpindi','Birmingham',
          'Centurion','Gros Islet','Kingston','Mong Kok','Dambulla','Dharamsala',
          'The Hague','Ahmedabad','Kirtipur','Lauderhill','Kathmandu','Kingstown',
          'Pune','Dublin','Nagpur','Durban','Bristol','Entebbe','Providence',
          'Chittagong','Belfast','Rotterdam','Bulawayo','Visakhapatnam']

# ==================== MATCH SETUP ====================
st.markdown('<div class="section-title">🏟️ Match Setup</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("🏏 Batting Team", sorted(teams))
with col2:
    bowling_team = st.selectbox("⚾ Bowling Team", sorted(teams))

city = st.selectbox("📍 Venue City", sorted(cities))

# Show match-up bar
if batting_team != bowling_team:
    st.markdown(
        f'<div class="match-info"><span class="match-vs">{batting_team}  &nbsp;vs&nbsp;  {bowling_team}  •  {city}</span></div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ==================== MATCH SITUATION ====================
st.markdown('<div class="section-title">📊 Current Match Situation</div>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input("Runs Scored", min_value=0, max_value=500, value=100)

with col4:
    # Generate only valid cricket over values: 0.1–0.6, 1.1–1.6, ..., 19.1–19.6
    valid_overs = [float(f"{o}.{b}") for o in range(20) for b in range(1, 7)]
    overs_completed = st.selectbox(
        "Overs Completed",
        options=valid_overs,
        index=5,  # default 0.6
        format_func=lambda x: f"{x:.1f}"
    )

with col5:
    wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=10, value=2)

col6, col7 = st.columns(2)
with col6:
    last_five_runs = st.number_input(
        "Runs in Last 5 Overs",
        min_value=0,
        max_value=200,
        value=30
    )
with col7:
    innings = st.radio("Innings", ["1st Innings", "2nd Innings"], horizontal=True)

target_score = 0
if innings == "2nd Innings":
    target_score = st.number_input(
        "Target Score to Chase",
        min_value=1,
        max_value=500,
        value=180
    )

st.markdown("---")

# ==================== PREDICTION ====================

# Helper to predict with custom params
def predict_score(bat, bowl, ct, score, b_left, w_left, run_rate, l5):
    df = pd.DataFrame({
        "batting_team": [bat], "bowling_team": [bowl], "city": [ct],
        "curr_score": [score], "balls_left": [b_left],
        "wickets_left": [w_left], "crr": [run_rate], "last_five": [l5]
    })
    return int(round(pipe.predict(df)[0]))


if st.button("🎯  Predict Final Score"):

    if batting_team == bowling_team:
        st.warning("⚠️ Batting and bowling teams must be different.")
        st.stop()

    balls_bowled = balls_from_overs(overs_completed)
    balls_left = 120 - balls_bowled
    if balls_left < 0:
        balls_left = 0

    wickets_left = 10 - wickets_lost

    if balls_bowled == 0:
        crr = 0
    else:
        crr = current_score / (balls_bowled / 6)

    try:
        predicted_score = predict_score(
            batting_team, bowling_team, city,
            current_score, balls_left, wickets_left, crr, last_five_runs
        )

        # Save to session state so What-If works outside button
        st.session_state.prediction_done = True
        st.session_state.predicted_score = predicted_score
        st.session_state.balls_left = balls_left
        st.session_state.wickets_left = wickets_left
        st.session_state.crr = crr
        st.session_state.target_score = target_score
        st.session_state.batting_team = batting_team
        st.session_state.bowling_team = bowling_team
        st.session_state.city = city
        st.session_state.current_score = current_score
        st.session_state.last_five_runs = last_five_runs

    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)


# ==================== DISPLAY RESULTS ====================
if st.session_state.get("prediction_done"):

    predicted_score = st.session_state.predicted_score
    balls_left = st.session_state.balls_left
    wickets_left = st.session_state.wickets_left
    crr = st.session_state.crr
    target_score = st.session_state.target_score
    bat_team = st.session_state.batting_team
    bowl_team = st.session_state.bowling_team
    s_city = st.session_state.city
    s_score = st.session_state.current_score
    s_l5 = st.session_state.last_five_runs

    runs_remaining = predicted_score - s_score
    overs_left_str = f"{balls_left // 6}.{balls_left % 6}"

    # --- Score Range ---
    variations = []
    for l5_adj in [-5, 0, 5]:
        for w_adj in [-1, 0, 1]:
            adj_l5 = max(0, s_l5 + l5_adj)
            adj_wl = max(1, min(10, wickets_left + w_adj))
            v = predict_score(
                bat_team, bowl_team, s_city,
                s_score, balls_left, adj_wl, crr, adj_l5
            )
            variations.append(v)
    score_low = min(variations)
    score_high = max(variations)

    # Big prediction display
    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-label">Predicted Final Score</div>
        <div class="prediction-score">{predicted_score}</div>
        <div class="stats-row">
            <div class="stat-item">
                <div class="stat-value">{s_score}</div>
                <div class="stat-label">Current</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{runs_remaining}</div>
                <div class="stat-label">Runs to Add</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{crr:.1f}</div>
                <div class="stat-label">Run Rate</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{overs_left_str}</div>
                <div class="stat-label">Overs Left</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{wickets_left}</div>
                <div class="stat-label">Wickets Left</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Score range
    st.markdown(f"""
    <div class="range-box">
        <div class="range-text">{score_low} — {score_high}</div>
        <div class="range-sub">Estimated score range based on match variables</div>
    </div>
    """, unsafe_allow_html=True)

    # --- Win Probability (only for 2nd innings) ---
    if target_score > 0:
        runs_needed = target_score - s_score
        overs_remaining = balls_left / 6
        rrr = runs_needed / overs_remaining if overs_remaining > 0 else 99.0

        # Sigmoid-based probability for smoother, more realistic values
        # x > 0 means batting team is ahead (predicted > target)
        score_diff = predicted_score - target_score

        # Scale factor: how sensitive the probability is
        # More overs left = less certain, fewer wickets = less certain
        certainty = 0.5 + (1 - balls_left / 120) * 0.5  # 0.5 early to 1.0 late
        wicket_factor = wickets_left / 10  # 1.0 with all wickets, 0.1 with 1

        k = 0.08 * certainty  # steepness of sigmoid
        raw_prob = 1 / (1 + np.exp(-k * score_diff))

        # Blend with wicket factor (more wickets = more confident)
        win_prob = raw_prob * wicket_factor + 0.5 * (1 - wicket_factor)

        # Adjust for required run rate pressure
        if overs_remaining > 0:
            rr_pressure = rrr - crr  # how much harder they need to score
            if rr_pressure > 6:
                win_prob *= 0.7
            elif rr_pressure > 3:
                win_prob *= 0.85
            elif rr_pressure < -3:
                win_prob = min(0.95, win_prob * 1.15)

        # Clamp between 5% and 95%
        win_prob = int(round(max(5, min(95, win_prob * 100))))

        lose_prob = 100 - win_prob
        st.markdown(f"""
        <div class="win-prob-container">
            <div class="win-prob-title">Win Probability</div>
            <div class="win-bar-bg">
                <div class="win-bar-fill" style="width:{win_prob}%"></div>
            </div>
            <div class="win-bar-labels">
                <span class="win-team">{bat_team} <span class="win-pct">{win_prob}%</span></span>
                <span class="win-team"><span class="win-pct">{lose_prob}%</span> {bowl_team}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="range-box">
            <div class="range-text">RRR: {rrr:.2f}</div>
            <div class="range-sub">Required run rate to reach target of {target_score}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ==================== WHAT-IF SIMULATOR ====================
    st.markdown('<div class="whatif-title">🔮 What-If Simulator</div>', unsafe_allow_html=True)
    st.caption("Adjust these sliders to see how the predicted score changes")

    wi_col1, wi_col2 = st.columns(2)
    with wi_col1:
        wi_extra_wickets = st.slider(
            "Extra wickets falling",
            min_value=0,
            max_value=max(0, wickets_left - 1),
            value=0
        )
    with wi_col2:
        wi_rr_change = st.slider(
            "Run rate change",
            min_value=-4.0,
            max_value=4.0,
            value=0.0,
            step=0.5
        )

    wi_col3, wi_col4 = st.columns(2)
    with wi_col3:
        wi_l5_change = st.slider(
            "Last 5 overs runs change",
            min_value=-20,
            max_value=20,
            value=0,
            step=5
        )
    with wi_col4:
        wi_score_change = st.slider(
            "Current score adjustment",
            min_value=-30,
            max_value=30,
            value=0,
            step=5
        )

    # Calculate what-if prediction
    wi_wickets_left = max(1, wickets_left - wi_extra_wickets)
    wi_crr = max(0, crr + wi_rr_change)
    wi_l5 = max(0, s_l5 + wi_l5_change)
    wi_score = max(0, s_score + wi_score_change)

    wi_predicted = predict_score(
        bat_team, bowl_team, s_city,
        wi_score, balls_left, wi_wickets_left, wi_crr, wi_l5
    )
    wi_diff = wi_predicted - predicted_score

    if wi_diff > 0:
        diff_class = "whatif-up"
        diff_text = f"▲ {wi_diff} more than base prediction"
    elif wi_diff < 0:
        diff_class = "whatif-down"
        diff_text = f"▼ {abs(wi_diff)} less than base prediction"
    else:
        diff_class = "whatif-same"
        diff_text = "Same as base prediction"

    st.markdown(f"""
    <div class="whatif-result">
        <div class="whatif-score">{wi_predicted}</div>
        <div class="whatif-diff {diff_class}">{diff_text}</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:rgba(255,255,255,0.3); font-size:0.8rem;">'
    'Built with Streamlit & XGBoost  •  T20 Cricket Score Predictor'
    '</p>',
    unsafe_allow_html=True
)