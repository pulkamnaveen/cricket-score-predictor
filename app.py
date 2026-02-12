import streamlit as st
import pickle
import pandas as pd

# ---------------- Page config ----------------
st.set_page_config(page_title="Cricket Score Predictor", layout="centered")
st.title("Cricket Score Predictor")

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
    st.success("Model loaded successfully")
except Exception as e:
    st.error("Model load failed")
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

# ---------------- UI ----------------

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Batting Team", sorted(teams))
with col2:
    bowling_team = st.selectbox("Bowling Team", sorted(teams))

city = st.selectbox("City", sorted(cities))

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input("Current Score", min_value=0, max_value=500, value=100)

with col4:
    if "overs" not in st.session_state:
        st.session_state.overs = 0.1

    st.markdown("Overs Completed")

    # Match the number_input widget's own +/- button layout
    c1, c2, c3 = st.columns([1, 2, 1])

    with c1:
        minus_clicked = st.button("−", key="ov_minus", use_container_width=True)
    with c2:
        # Display value in a box styled to match number_input
        st.markdown(
            f"""<div style="
                display:flex;align-items:center;justify-content:center;
                height:38px;
                background:rgb(38,39,48);
                border:1px solid rgba(250,250,250,0.2);
                border-radius:0.5rem;
                color:white;
                font-size:1rem;
            ">{st.session_state.overs:.1f}</div>""",
            unsafe_allow_html=True
        )
    with c3:
        plus_clicked = st.button("+", key="ov_plus", use_container_width=True)

    # Update state AFTER rendering so the new value shows on next rerun
    if minus_clicked and st.session_state.overs > 0.1:
        st.session_state.overs = prev_ball(st.session_state.overs)
        st.rerun()

    if plus_clicked and st.session_state.overs < 19.6:
        st.session_state.overs = next_ball(st.session_state.overs)
        st.rerun()

    overs_completed = st.session_state.overs

with col5:
    wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=10, value=2)

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
        st.success(f"Predicted Final Score: {int(round(prediction))}")
    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)