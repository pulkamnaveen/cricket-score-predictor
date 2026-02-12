import streamlit as st
import pickle
import pandas as pd

# ---------------- Page config ----------------
st.set_page_config(page_title="Cricket Score Predictor", layout="centered")
st.title("Cricket Score Predictor")

# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    with open("xgb_pipeline.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    pipe = load_model()
    st.success("Model loaded successfully")

except Exception as e:
    st.error("Model load failed")
    st.exception(e)
    st.stop()

# ---------------- Data ----------------
teams = ['Pakistan',
 'India',
 'New Zealand',
 'Sri Lanka',
 'South Africa',
 'West Indies',
 'England',
 'Bangladesh',
 'Zimbabwe',
 'Australia',
 'Ireland',
 'Hong Kong',
 'Netherlands',
 'United Arab Emirates',
 'Malaysia',
 'Nigeria',
 'Uganda',
 'Bahrain',
 'Scotland',
 'Nepal']

cities = ['Dubai', 'Wellington', 'Cape Town', 'Kuala Lumpur', 'Hamilton', 'Christchurch', "St George's",
 'Pallekele', 'Edinburgh', 'Melbourne', 'Chattogram', 'Harare', 'Dhaka', 'Bangalore', 'Colombo',
 'Basseterre', 'Cardiff', 'Johannesburg', 'Kigali City', 'Rajkot', 'Sydney', 'Napier', 'Auckland',
 'Kandy', 'Canberra', 'Manchester', 'Bridgetown', 'Adelaide', 'Perth', 'Mumbai', 'Hambantota',
 'Al Amarat', 'Hobart', 'Abu Dhabi', 'Chester-le-Street', 'Trinidad', 'Guyana', 'Nottingham',
 'Sharjah', 'Tarouba', 'Southampton', 'St Lucia', 'Mirpur', 'Glasgow', 'London', 'Lahore',
 'Brisbane', 'Mount Maunganui', 'Barbados', 'Karachi', 'Kolkata', 'Sylhet', 'Delhi', 'Chandigarh',
 'Rawalpindi', 'Birmingham', 'Centurion', 'Gros Islet', 'Kingston', 'Mong Kok', 'Dambulla',
 'Dharamsala', 'The Hague', 'Ahmedabad', 'Kirtipur', 'Lauderhill', 'Kathmandu', 'Kingstown',
 'Pune', 'Dublin', 'Nagpur', 'Durban', 'Bristol', 'Entebbe', 'Providence', 'Chittagong',
 'Belfast', 'Rotterdam', 'Bulawayo', 'Visakhapatnam']

# ---------------- UI ----------------
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Batting Team", sorted(teams))
with col2:
    bowling_team = st.selectbox("Bowling Team", sorted(teams))

city = st.selectbox("City", sorted(cities))

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input(
        "Current Score", min_value=0, max_value=500, value=100
    )

with col4:
    overs_completed_raw = st.number_input(
        "Overs Completed",
        min_value=0.0,
        max_value=20.6,
        value=0.1,
        step=0.1
    )

    # ---- convert to valid cricket over format ----
    o = int(overs_completed_raw)
    b = int(round((overs_completed_raw - o) * 10))

    if overs_completed_raw != 0 and b == 0:
        b = 1

    if b > 6:
        o += 1
        b = 1

    overs_completed = float(f"{o}.{b}")

with col5:
    wickets_lost = st.number_input(
        "Wickets Lost", min_value=0, max_value=10, value=2
    )

last_five_runs = st.number_input(
    "Runs in last 5 overs (cumulative)",
    min_value=0,
    max_value=200,
    value=30
)

# ---------------- Prediction ----------------
if st.button("Predict Final Score"):

    if batting_team == bowling_team:
        st.warning("Batting and bowling teams must be different.")
        st.stop()

    # ---- convert overs to balls correctly ----
    completed_overs = int(overs_completed)
    completed_balls = int(round((overs_completed - completed_overs) * 10))

    balls_bowled = completed_overs * 6 + completed_balls
    balls_left = 120 - balls_bowled

    if balls_left < 0:
        balls_left = 0

    wickets_left = 10 - wickets_lost

    if overs_completed == 0:
        crr = 0
    else:
        crr = current_score / overs_completed

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
        st.success(f"Predicted Final Score: {int(round(prediction))}")
        st.info(f"Overs used for calculation: {overs_completed}")
    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)