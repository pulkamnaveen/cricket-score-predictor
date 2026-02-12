import streamlit as st
import pickle
import pandas as pd

# ---------------- Page config ----------------
st.set_page_config(page_title="Cricket Score Predictor", layout="centered")
st.title("Cricket Score Predictor")

# ---------------- Overs helpers ----------------
def add_ball(overs):
    o = int(overs)
    b = int(round((overs - o) * 10))

    b += 1
    if b == 6:
        o += 1
        b = 0

    return float(f"{o}.{b}")


def remove_ball(overs):
    o = int(overs)
    b = int(round((overs - o) * 10))

    b -= 1
    if b < 0:
        o -= 1
        b = 5

    if o < 0:
        return 0.0

    return float(f"{o}.{b}")


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

cities = ['Dubai', 'Wellington', 'Cape Town', 'Kuala Lumpur', 'Hamilton', 'Christchurch',
 "St George's", 'Pallekele', 'Edinburgh', 'Melbourne', 'Chattogram', 'Harare', 'Dhaka',
 'Bangalore', 'Colombo', 'Basseterre', 'Cardiff', 'Johannesburg', 'Kigali City', 'Rajkot',
 'Sydney', 'Napier', 'Auckland', 'Kandy', 'Canberra', 'Manchester', 'Bridgetown', 'Adelaide',
 'Perth', 'Mumbai', 'Hambantota', 'Al Amarat', 'Hobart', 'Abu Dhabi', 'Chester-le-Street',
 'Trinidad', 'Guyana', 'Nottingham', 'Sharjah', 'Tarouba', 'Southampton', 'St Lucia',
 'Mirpur', 'Glasgow', 'London', 'Lahore', 'Brisbane', 'Mount Maunganui', 'Barbados',
 'Karachi', 'Kolkata', 'Sylhet', 'Delhi', 'Chandigarh', 'Rawalpindi', 'Birmingham',
 'Centurion', 'Gros Islet', 'Kingston', 'Mong Kok', 'Dambulla', 'Dharamsala', 'The Hague',
 'Ahmedabad', 'Kirtipur', 'Lauderhill', 'Kathmandu', 'Kingstown', 'Pune', 'Dublin',
 'Nagpur', 'Durban', 'Bristol', 'Entebbe', 'Providence', 'Chittagong', 'Belfast',
 'Rotterdam', 'Bulawayo', 'Visakhapatnam']

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

    if "overs_completed" not in st.session_state:
        st.session_state.overs_completed = 10.0

    st.markdown("Overs Completed")

    c1, c2, c3 = st.columns([1, 2, 1])

    with c1:
        if st.button("−", key="overs_minus"):
            st.session_state.overs_completed = remove_ball(
                st.session_state.overs_completed
            )

    with c2:
        st.write(f"**{st.session_state.overs_completed:.1f}**")

    with c3:
        if st.button("+", key="overs_plus"):
            st.session_state.overs_completed = add_ball(
                st.session_state.overs_completed
            )

    overs_completed = st.session_state.overs_completed


with col5:
    wickets_lost = st.number_input(
        "Wickets Lost", min_value=0, max_value=10, value=2
    )

# User enters cumulative last 5 overs
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

    # ---- feature engineering (same as training) ----

    balls_bowled = int(overs_completed) * 6 + int(
        round((overs_completed - int(overs_completed)) * 10)
    )

    balls_left = 120 - balls_bowled
    wickets_left = 10 - wickets_lost

    if balls_left < 0:
        balls_left = 0

    if overs_completed == 0:
        crr = 0
    else:
        crr = current_score / overs_completed

    # ---- input dataframe (exact columns expected by model) ----
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
        st.info(f"Last 5 overs (cumulative runs entered): {last_five_runs}")
    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)