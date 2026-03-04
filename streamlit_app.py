import os
import requests
import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# App setup
st.set_page_config(page_title="Deutsche Bahn Delay Project", layout="wide")
st.title("🚆 Deutsche Bahn Train Departure Delay Predictor")

# Load data
DATA_URL = "https://github.com/hz3396/DBDelayPredict/releases/download/v1.0/db_sample.csv"
DATA_FILE = "db_sample.csv"

if not os.path.exists(DATA_FILE):
    st.write("Downloading dataset (first time only)...")
    r = requests.get(DATA_URL, timeout=180)
    r.raise_for_status()
    with open(DATA_FILE, "wb") as f:
        f.write(r.content)

raw = pd.read_csv(
    DATA_FILE,
    na_values=["None", "none", "NULL", "null", "NaN", "nan", ""],
)

# Rename columns
raw = raw.rename(columns={
    "arrival_delay_m": "arrival_delay_time",
    "departure_delay_m": "departure_delay_time",
    "arrival_delay_flag": "arrival_delay_severe_or_not",
    "category": "station_category",
})

# Cleaning
def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw2 = raw_df.copy()

    raw2["arrival_plan"] = pd.to_datetime(raw2["arrival_plan"], errors="coerce")
    raw2["departure_plan"] = pd.to_datetime(raw2["departure_plan"], errors="coerce")

    raw2["hour"] = raw2["arrival_plan"].dt.hour
    raw2["day_of_week"] = raw2["arrival_plan"].dt.dayofweek

    raw2["planned_dwell_time"] = (raw2["departure_plan"] - raw2["arrival_plan"]).dt.total_seconds() / 60

    raw2["is_peak"] = raw2["hour"].isin([7,8,9,16,17,18]).astype(int)
    raw2["arrival_delay_severe_or_not"] = (raw2["arrival_delay_time"] > 6).astype(int)

    cols = [
        "departure_delay_time",
        "arrival_delay_time",
        "planned_dwell_time",
        "station_category",
        "hour",
        "line",
        "day_of_week",
        "is_peak",
        "arrival_delay_severe_or_not",
        "station",
        "state",
        "city",
        "info",
    ]

    cols_existing = [c for c in cols if c in raw2.columns]
    df = raw2[cols_existing].copy()

    numeric_cols = [
        "departure_delay_time",
        "arrival_delay_time",
        "planned_dwell_time",
        "station_category",
        "hour",
        "line",
        "day_of_week",
        "is_peak",
        "arrival_delay_severe_or_not",
    ]

    numeric_cols = [c for c in numeric_cols if c in df.columns]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=numeric_cols)

    df = df[(df["departure_delay_time"] >= 0) & (df["departure_delay_time"] <= 180)]
    df = df[(df["arrival_delay_time"] >= 0) & (df["arrival_delay_time"] <= 180)]
    df = df[(df["planned_dwell_time"] >= 0) & (df["planned_dwell_time"] <= 60)]
    df = df[(df["station_category"] >= 1) & (df["station_category"] <= 7)]
    df = df[(df["hour"] >= 0) & (df["hour"] <= 23)]
    df = df[(df["day_of_week"] >= 0) & (df["day_of_week"] <= 6)]

    text_cols = ["station","state","city","info"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()

    return df


df = clean_data(raw)

# Sidebar
page = st.sidebar.radio("**Page Selection**",
                        ["01 Introduction","02 Data Visualization","03 Prediction"])

# Page 01
if page == "01 Introduction":

    st.image("Weixin Image_2026-03-02_181855_878.jpg", width=1500)

    st.header("Project Overview")

    st.write("""
This project predicts **departure_delay_time** using a Linear Regression model.
""")

    st.subheader("Dataset Information")

    rows = st.slider("Rows",5,100,20)

    st.dataframe(raw.head(rows))


# Page 02
elif page == "02 Data Visualization":

    st.image("02.jpg", width=1500)

    st.subheader("1) Departure Delay Distribution")

    fig = plt.figure(figsize=(7,4))
    plt.hist(df["departure_delay_time"],bins=50,range=(1,20))
    plt.xlim(1,20)
    st.pyplot(fig)

    st.subheader("2) Arrival vs Departure Delay")

    fig = plt.figure(figsize=(7,5))

    delay_df = df[(df["arrival_delay_time"]>0)&(df["departure_delay_time"]>0)]

    sample = delay_df.sample(min(3000,len(delay_df)),random_state=42)

    plt.scatter(sample["departure_delay_time"],sample["arrival_delay_time"],alpha=0.3,s=10)

    st.pyplot(fig)

    st.subheader("3) On Time vs Delayed")

    on_time = (df["departure_delay_time"]==0).sum()
    delay = (df["departure_delay_time"]>0).sum()

    fig,ax = plt.subplots(figsize=(5,5))
    ax.pie([on_time,delay],labels=["on_time","delay"],autopct="%1.1f%%")
    st.pyplot(fig)

    st.subheader("6) Planned Dwell vs Departure Delay")

    fig = plt.figure(figsize=(7,4))
    plt.scatter(df["planned_dwell_time"],df["departure_delay_time"],alpha=0.3,s=8)
    st.pyplot(fig)

    st.subheader("7) Correlation Heatmap")

    fig = plt.figure(figsize=(8,5))
    corr = df.select_dtypes(include="number").corr()
    sns.heatmap(corr,annot=True,fmt=".2f")
    st.pyplot(fig)


# Page 03
else:

    st.subheader("Train Linear Regression Model")

    st.image("03.jpg", width=1500)

    feature_cols = [
        "arrival_delay_time",
        "planned_dwell_time",
        "station_category",
        "hour",
        "line",
        "day_of_week",
        "is_peak",
        "arrival_delay_severe_or_not",
    ]

    X = df[feature_cols]
    y = df["departure_delay_time"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    model = LinearRegression()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)

    c1,c2 = st.columns(2)
    c1.metric("MAE (minutes)",f"{mae:.2f}")
    c2.metric("R²",f"{r2:.3f}")

    st.subheader("Actual vs Predicted")

    fig = plt.figure(figsize=(10,6))
    plt.scatter(y_pred,y_test,s=10,alpha=0.3)

    mn=min(y_pred.min(),y_test.min())
    mx=max(y_pred.max(),y_test.max())

    plt.plot([mn,mx],[mn,mx],"r--")

    st.pyplot(fig)

    st.subheader("Try your own inputs")

    col1,col2=st.columns(2)

    with col1:
        is_peak=int(st.toggle("Is Peak Hour"))

    with col2:
        arrival_delay_severe_or_not=int(st.toggle("Arrival Delay Severe"))

    arrival_delay_time = st.slider("arrival_delay_time",0.0,180.0,5.0)

    planned_dwell_time = st.slider("planned_dwell_time",0.0,60.0,2.0)

    station_category = st.slider("station_category",1,7,3)

    hour = st.slider("hour",0,23,8)

    day_of_week = st.slider("day_of_week",0,6,1)

    line = st.slider("line",0,10,0)

    new_X = pd.DataFrame([{
        "arrival_delay_time":arrival_delay_time,
        "planned_dwell_time":planned_dwell_time,
        "station_category":station_category,
        "hour":hour,
        "line":line,
        "day_of_week":day_of_week,
        "is_peak":is_peak,
        "arrival_delay_severe_or_not":arrival_delay_severe_or_not
    }])

    new_X = new_X[feature_cols]

    pred = model.predict(new_X)[0]

    st.write(f"Predicted departure_delay_time: **{pred:.1f} minutes**")
