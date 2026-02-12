# 🏏 Cricket Score Predictor – Machine Learning Project

A web-based machine learning application that predicts the final score of a T20 cricket match based on the current match situation.  
The app is built using Python, Scikit-Learn, XGBoost and Streamlit.

---

## 🚀 Live Demo

https://cricket-score-predictor-jrhbjenslymm256qucr6v.streamlit.app

---

## 📌 Project Overview

This project predicts the final score of a batting team in a T20 match using the following real-time match inputs:

- Batting team
- Bowling team
- City
- Current score
- Overs completed
- Wickets lost
- Runs scored in the last 5 overs

The model learns patterns from historical international T20 match data and estimates the final score.

---

## 🧠 Machine Learning Approach

- Historical T20 match data used
- Feature engineering:
  - balls left
  - current run rate
  - wickets remaining
- Categorical features handled using a pipeline
- Regression model trained using XGBoost
- Final pipeline saved and reused in the Streamlit app

---

## 🧾 Input Features

| Feature | Description |
|------|------------|
| batting_team | Batting team |
| bowling_team | Bowling team |
| city | Match city |
| curr_score | Current score |
| balls_left | Balls remaining |
| wickets_left | Wickets remaining |
| crr | Current run rate |
| last_five | Runs in last 5 overs |

---

## 🖥️ Web Application

Users can select teams, city and enter match situation to predict the final score.

The overs input follows correct cricket logic:

0.1 → 0.2 → … → 0.6 → 1.1 → 1.2 → …

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Streamlit

---

## 📂 Project Structure

cricket-score-predictor/
│
├── app.py
├── xgb_pipeline.pkl
├── requirements.txt
├── runtime.txt
└── README.md

---

## ⚙️ Run Locally

```bash
git clone https://github.com/pulkamnaveen/cricket-score-predictor.git
cd cricket-score-predictor
pip install -r requirements.txt
streamlit run app.py


👤 Author
Pulkam Naveen
https://github.com/pulkamnaveen