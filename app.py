import streamlit as st
import pickle
import pandas as pd

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

last_five_runs = st.number_input(
    "Runs in Last 5 Overs",
    min_value=0,
    max_value=200,
    value=30
)

st.markdown("---")

# ==================== PREDICTION ====================

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

    input_df = pd.DataFrame({
        "batting_team": [batting_team],
        "bowling_team": [bowling_team],
        "city": [city],
        "curr_score": [current_score],
        "balls_left": [balls_left],
        "wickets_left": [wickets_left],
        "crr": [crr],
        "last_five": [last_five_runs]
    })

    try:
        prediction = pipe.predict(input_df)[0]
        predicted_score = int(round(prediction))
        runs_remaining = predicted_score - current_score
        overs_left = f"{balls_left // 6}.{balls_left % 6}"

        # Big prediction display
        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-label">Predicted Final Score</div>
            <div class="prediction-score">{predicted_score}</div>
            <div class="stats-row">
                <div class="stat-item">
                    <div class="stat-value">{current_score}</div>
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
                    <div class="stat-value">{overs_left}</div>
                    <div class="stat-label">Overs Left</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{wickets_left}</div>
                    <div class="stat-label">Wickets Left</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:rgba(255,255,255,0.3); font-size:0.8rem;">'
    'Built with Streamlit & XGBoost  •  T20 Cricket Score Predictor'
    '</p>',
    unsafe_allow_html=True
)