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
    /* Soft blue background */
    .stApp {
        background: #eef2f7;
    }

    /* Hide streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* ---- Banner ---- */
    .hero {
        background: #2d6a4f;
        border-radius: 14px;
        padding: 2.2rem 1.5rem 1.8rem;
        text-align: center;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -30px; right: -30px;
        width: 120px; height: 120px;
        background: rgba(255,255,255,0.06);
        border-radius: 50%;
    }
    .hero h1 {
        color: #fff;
        font-size: 1.75rem;
        margin: 0;
        font-weight: 700;
        position: relative;
    }
    .hero .tagline {
        color: #b7dbca;
        font-size: 0.88rem;
        margin-top: 0.4rem;
        font-weight: 400;
        position: relative;
    }

    /* ---- Section headers ---- */
    .sec-head {
        color: #2d6a4f;
        font-size: 0.82rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin-bottom: 0.5rem;
    }

    /* ---- Match-up strip ---- */
    .matchup {
        background: #fff;
        border: 2px solid #2d6a4f;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        text-align: center;
        margin: 0.6rem 0 0.4rem;
    }
    .matchup-text {
        color: #2d6a4f;
        font-size: 0.95rem;
        font-weight: 600;
    }

    /* ---- Result card ---- */
    .result-card {
        background: #2d6a4f;
        border-radius: 14px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-label {
        color: #b7dbca;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .result-score {
        color: #fff;
        font-size: 3.2rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 0.3rem 0 0.1rem;
    }

    /* ---- Stat chips ---- */
    .chips {
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    .chip {
        background: rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 0.35rem 0.9rem;
        text-align: center;
    }
    .chip-val {
        color: #fff;
        font-size: 1rem;
        font-weight: 600;
    }
    .chip-lbl {
        color: #b7dbca;
        font-size: 0.65rem;
        margin-top: 1px;
    }

    /* ---- Range pill ---- */
    .range-pill {
        background: #fff;
        border: 1px solid #e0d8cc;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        text-align: center;
        margin-bottom: 0.8rem;
    }
    .range-num {
        color: #2d6a4f;
        font-size: 1.2rem;
        font-weight: 700;
    }
    .range-hint {
        color: #8a7e6b;
        font-size: 0.75rem;
        margin-top: 0.15rem;
    }

    /* ---- Win probability ---- */
    .winprob {
        background: #fff;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .winprob-head {
        color: #555;
        font-size: 0.78rem;
        font-weight: 500;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .bar-track {
        background: #e8d5d5;
        border-radius: 8px;
        height: 28px;
        width: 100%;
        overflow: hidden;
    }
    .bar-fill {
        background: #2d6a4f;
        height: 100%;
        border-radius: 8px 0 0 8px;
        transition: width 0.4s ease;
    }
    .bar-labels {
        display: flex;
        justify-content: space-between;
        margin-top: 0.35rem;
        font-size: 0.78rem;
    }
    .bar-left { color: #2d6a4f; font-weight: 600; }
    .bar-right { color: #c0392b; font-weight: 600; }

    /* ---- RRR pill ---- */
    .rrr-pill {
        background: #fff;
        border: 1px solid #e0d8cc;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        text-align: center;
        margin-bottom: 0.8rem;
    }
    .rrr-val {
        color: #c0392b;
        font-size: 1.1rem;
        font-weight: 700;
    }
    .rrr-hint {
        color: #8a7e6b;
        font-size: 0.72rem;
    }

    /* ---- What-If ---- */
    .whatif-box {
        background: #fff;
        border-radius: 10px;
        padding: 1.2rem 1.3rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        margin-top: 0.5rem;
    }
    .whatif-num {
        color: #2d6a4f;
        font-size: 2rem;
        font-weight: 700;
    }
    .whatif-msg { font-size: 0.82rem; margin-top: 0.2rem; }
    .msg-up { color: #27ae60; }
    .msg-down { color: #c0392b; }
    .msg-same { color: #8a7e6b; }

    /* ---- Button ---- */
    div.stButton > button {
        background: #2d6a4f;
        color: #fff;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        width: 100%;
        cursor: pointer;
        transition: background 0.2s, transform 0.1s;
    }
    div.stButton > button:hover {
        background: #3a8563;
        transform: translateY(-1px);
    }
    div.stButton > button:active {
        transform: translateY(0);
    }

    /* ---- Form labels ---- */
    .stSelectbox label, .stNumberInput label, .stRadio label, .stSlider label {
        color: #555 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }

    /* ---- Dividers ---- */
    hr { border-color: #e0d8cc !important; }

    /* ---- Footer ---- */
    .footer-text {
        text-align: center;
        color: #b5a99a;
        font-size: 0.75rem;
        padding: 0.5rem 0 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== BANNER ====================
st.markdown("""
<div class="hero">
    <h1>🏏 Cricket Score Predictor</h1>
    <p class="tagline">Predict the final T20 score based on the current match situation</p>
</div>
""", unsafe_allow_html=True)


# ==================== HELPERS ====================
def balls_from_overs(ov):
    o = int(ov)
    b = int(round((ov - o) * 10))
    return o * 6 + b


# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    with open("xgb_pipeline.pkl", "rb") as f:
        return pickle.load(f)

try:
    pipe = load_model()
except Exception as e:
    st.error("Could not load the prediction model.")
    st.exception(e)
    st.stop()


# ==================== DATA ====================
teams = sorted([
    'Pakistan','India','New Zealand','Sri Lanka','South Africa','West Indies',
    'England','Bangladesh','Zimbabwe','Australia','Ireland','Hong Kong',
    'Netherlands','United Arab Emirates','Malaysia','Nigeria','Uganda',
    'Bahrain','Scotland','Nepal'
])

cities = sorted([
    'Dubai','Wellington','Cape Town','Kuala Lumpur','Hamilton','Christchurch',
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
    'Chittagong','Belfast','Rotterdam','Bulawayo','Visakhapatnam'
])


# ==================== MATCH SETUP ====================
st.markdown('<div class="sec-head">🏟️ Pick the teams & venue</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Batting Team", teams)
with col2:
    bowling_team = st.selectbox("Bowling Team", teams)

city = st.selectbox("Venue City", cities)

if batting_team != bowling_team:
    st.markdown(
        f'<div class="matchup"><span class="matchup-text">{batting_team}  vs  {bowling_team}  •  {city}</span></div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ==================== MATCH SITUATION ====================
st.markdown('<div class="sec-head">📊 Enter the current match situation</div>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input("Runs Scored", min_value=0, max_value=500, value=100)

with col4:
    valid_overs = [float(f"{o}.{b}") for o in range(20) for b in range(1, 7)]
    overs_completed = st.selectbox(
        "Overs Completed", options=valid_overs, index=5,
        format_func=lambda x: f"{x:.1f}"
    )

with col5:
    wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=10, value=2)

col6, col7 = st.columns(2)
with col6:
    last_five_runs = st.number_input("Runs in Last 5 Overs", min_value=0, max_value=200, value=30)
with col7:
    innings = st.radio("Innings", ["1st Innings", "2nd Innings"], horizontal=True)

target_score = 0
if innings == "2nd Innings":
    target_score = st.number_input("Target Score to Chase", min_value=1, max_value=500, value=180)

st.markdown("---")


# ==================== PREDICT HELPER ====================
def predict_score(bat, bowl, ct, score, b_left, w_left, run_rate, l5):
    df = pd.DataFrame({
        "batting_team": [bat], "bowling_team": [bowl], "city": [ct],
        "curr_score": [score], "balls_left": [b_left],
        "wickets_left": [w_left], "crr": [run_rate], "last_five": [l5]
    })
    return int(round(pipe.predict(df)[0]))


# ==================== PREDICT BUTTON ====================
if st.button("Predict Final Score"):

    if batting_team == bowling_team:
        st.warning("Batting and bowling teams must be different.")
        st.stop()

    balls_bowled = balls_from_overs(overs_completed)
    balls_left = max(0, 120 - balls_bowled)
    wickets_left = 10 - wickets_lost
    crr = current_score / (balls_bowled / 6) if balls_bowled > 0 else 0

    try:
        predicted_score = predict_score(
            batting_team, bowling_team, city,
            current_score, balls_left, wickets_left, crr, last_five_runs
        )
        st.session_state.update({
            "done": True,
            "pred": predicted_score,
            "bl": balls_left, "wl": wickets_left, "crr": crr,
            "tgt": target_score,
            "bat": batting_team, "bowl": bowling_team,
            "ct": city, "sc": current_score, "l5": last_five_runs
        })
    except Exception as e:
        st.error("Prediction failed — please check your inputs.")
        st.exception(e)


# ==================== RESULTS ====================
if st.session_state.get("done"):

    predicted_score = st.session_state["pred"]
    balls_left = st.session_state["bl"]
    wickets_left = st.session_state["wl"]
    crr = st.session_state["crr"]
    target_score = st.session_state["tgt"]
    bat_team = st.session_state["bat"]
    bowl_team = st.session_state["bowl"]
    s_city = st.session_state["ct"]
    s_score = st.session_state["sc"]
    s_l5 = st.session_state["l5"]

    runs_to_add = predicted_score - s_score
    overs_left_str = f"{balls_left // 6}.{balls_left % 6}"

    # Score range
    variations = []
    for l5a in [-5, 0, 5]:
        for wa in [-1, 0, 1]:
            v = predict_score(
                bat_team, bowl_team, s_city, s_score, balls_left,
                max(1, min(10, wickets_left + wa)), crr, max(0, s_l5 + l5a)
            )
            variations.append(v)
    lo, hi = min(variations), max(variations)

    # --- Main prediction card ---
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Predicted Final Score</div>
        <div class="result-score">{predicted_score}</div>
        <div class="chips">
            <div class="chip"><div class="chip-val">{s_score}</div><div class="chip-lbl">Current</div></div>
            <div class="chip"><div class="chip-val">{runs_to_add}</div><div class="chip-lbl">To Add</div></div>
            <div class="chip"><div class="chip-val">{crr:.1f}</div><div class="chip-lbl">Run Rate</div></div>
            <div class="chip"><div class="chip-val">{overs_left_str}</div><div class="chip-lbl">Overs Left</div></div>
            <div class="chip"><div class="chip-val">{wickets_left}</div><div class="chip-lbl">Wickets</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Score range ---
    st.markdown(f"""
    <div class="range-pill">
        <div class="range-num">{lo}  —  {hi}</div>
        <div class="range-hint">Estimated score range considering match variables</div>
    </div>
    """, unsafe_allow_html=True)

    # --- Win probability (2nd innings only) ---
    if target_score > 0:
        runs_needed = target_score - s_score
        overs_rem = balls_left / 6
        rrr = runs_needed / overs_rem if overs_rem > 0 else 99.0

        score_diff = predicted_score - target_score
        certainty = 0.5 + (1 - balls_left / 120) * 0.5
        wicket_factor = wickets_left / 10
        k = 0.08 * certainty
        raw_prob = 1 / (1 + np.exp(-k * score_diff))
        win_prob = raw_prob * wicket_factor + 0.5 * (1 - wicket_factor)

        if overs_rem > 0:
            rr_pressure = rrr - crr
            if rr_pressure > 6:
                win_prob *= 0.7
            elif rr_pressure > 3:
                win_prob *= 0.85
            elif rr_pressure < -3:
                win_prob = min(0.95, win_prob * 1.15)

        wp = int(round(max(5, min(95, win_prob * 100))))
        lp = 100 - wp

        st.markdown(f"""
        <div class="winprob">
            <div class="winprob-head">Win Probability</div>
            <div class="bar-track"><div class="bar-fill" style="width:{wp}%"></div></div>
            <div class="bar-labels">
                <span class="bar-left">{bat_team} {wp}%</span>
                <span class="bar-right">{lp}% {bowl_team}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="rrr-pill">
            <div class="rrr-val">RRR: {rrr:.2f}</div>
            <div class="rrr-hint">Required run rate to chase {target_score}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- What-if simulator (fragment so sliders don't rerun full page) ---
    @st.fragment
    def whatif_simulator():
        st.markdown('<div class="sec-head">🔮 What-If Simulator</div>', unsafe_allow_html=True)
        st.caption("Move the sliders to explore different scenarios")

        wc1, wc2 = st.columns(2)
        with wc1:
            wi_wk = st.slider("Extra wickets falling", 0, max(1, wickets_left - 1), 0, key="wi_wk")
        with wc2:
            wi_rr = st.slider("Run rate change", -4.0, 4.0, 0.0, 0.5, key="wi_rr")

        wc3, wc4 = st.columns(2)
        with wc3:
            wi_l5 = st.slider("Last 5 overs runs change", -20, 20, 0, 5, key="wi_l5")
        with wc4:
            wi_sc = st.slider("Current score adjustment", -30, 30, 0, 5, key="wi_sc")

        wi_pred = predict_score(
            bat_team, bowl_team, s_city,
            max(0, s_score + wi_sc), balls_left,
            max(1, wickets_left - wi_wk),
            max(0.1, crr + wi_rr),
            max(0, s_l5 + wi_l5)
        )
        diff = wi_pred - predicted_score

        if diff > 0:
            cls, msg = "msg-up", f"▲ {diff} runs more than base prediction"
        elif diff < 0:
            cls, msg = "msg-down", f"▼ {abs(diff)} runs less than base prediction"
        else:
            cls, msg = "msg-same", "Same as base prediction"

        st.markdown(f"""
        <div class="whatif-box">
            <div class="whatif-num">{wi_pred}</div>
            <div class="whatif-msg {cls}">{msg}</div>
        </div>
        """, unsafe_allow_html=True)

    whatif_simulator()

# ==================== FOOTER ====================
st.markdown("---")
st.markdown('<div class="footer-text">Built with Streamlit & XGBoost • T20 Cricket Score Predictor</div>', unsafe_allow_html=True)
