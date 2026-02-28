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
    /* Overall background */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #1a2a4a 50%, #0d2137 100%);
    }

    /* Header banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e6f3e 0%, #2d8f4e 50%, #1a5c32 100%);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .hero-banner h1 {
        color: #ffffff;
        font-size: 2.2rem;
        margin: 0;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.8);
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }

    /* Section cards */
    .section-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    .section-title {
        color: #4ecdc4;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.8rem;
    }

    /* Prediction result */
    .prediction-box {
        background: linear-gradient(135deg, #1e6f3e 0%, #2d8f4e 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(46,143,78,0.3);
        border: 1px solid rgba(255,255,255,0.15);
        animation: fadeIn 0.5s ease-in;
    }
    .prediction-label {
        color: rgba(255,255,255,0.8);
        font-size: 1rem;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .prediction-score {
        color: #ffffff;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0.2rem 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }

    /* Stats row */
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1rem;
    }
    .stat-item {
        text-align: center;
    }
    .stat-value {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 700;
    }
    .stat-label {
        color: rgba(255,255,255,0.6);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Match info bar */
    .match-info {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 1rem;
    }
    .match-vs {
        color: #4ecdc4;
        font-size: 1.1rem;
        font-weight: 600;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Style the predict button */
    div.stButton > button {
        background: linear-gradient(135deg, #e8a838 0%, #f0c040 100%);
        color: #1a1a2e;
        font-weight: 700;
        font-size: 1.1rem;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        width: 100%;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(232,168,56,0.3);
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #f0c040 0%, #e8a838 100%);
        box-shadow: 0 6px 20px rgba(232,168,56,0.5);
        transform: translateY(-1px);
    }

    /* Selectbox / input labels */
    .stSelectbox label, .stNumberInput label {
        color: rgba(255,255,255,0.85) !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Hero Banner ----------------
st.markdown("""
<div class="hero-banner">
    <h1>🏏 Cricket Score Predictor</h1>
    <p>T20 match score prediction powered by XGBoost ML</p>
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