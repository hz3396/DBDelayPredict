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
st.write("Intro project: predict **departure_delay_m** with **Linear Regression**.")

# Load data
DATA_URL = "https://github.com/hz3396/DBDelayPredict/releases/download/v1.0/db_sample.csv"
DATA_FILE = "db_sample.csv"

if not os.path.exists(DATA_FILE):
    st.write("Downloading dataset (first time only)...")
    r = requests.get(DATA_URL)
    with open(DATA_FILE, "wb") as f:
        f.write(r.content)

raw = pd.read_csv(DATA_FILE)

# Build simple features
# Convert time columns
raw["arrival_plan"] = pd.to_datetime(raw["arrival_plan"], errors="coerce")
raw["departure_plan"] = pd.to_datetime(raw["departure_plan"], errors="coerce")

# Time features
raw["hour"] = raw["arrival_plan"].dt.hour
raw["day_of_week"] = raw["arrival_plan"].dt.dayofweek

# planned dwell time in minutes
raw["planned_dwell_m"] = (raw["departure_plan"] - raw["arrival_plan"]).dt.total_seconds() / 60

# simple flags
raw["is_peak"] = raw["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
raw["arrival_delay_flag"] = (raw["arrival_delay_m"] > 6).astype(int)

# Keep only needed columns (target + 8 features)
cols = [
    "departure_delay_m",     # target
    "arrival_delay_m",
    "planned_dwell_m",
    "category",
    "hour",
    "line",
    "day_of_week",
    "is_peak",
    "arrival_delay_flag",
]
df = raw[cols].copy()

# Convert to numeric (easy)
for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop missing
df = df.dropna()

# Simple filtering (keeps plots/model stable)
df = df[(df["departure_delay_m"] >= 0) & (df["departure_delay_m"] <= 180)]
df = df[(df["arrival_delay_m"] >= 0) & (df["arrival_delay_m"] <= 180)]
df = df[(df["planned_dwell_m"] >= 0) & (df["planned_dwell_m"] <= 60)]
df = df[(df["category"] >= 1) & (df["category"] <= 7)]
df = df[(df["hour"] >= 0) & (df["hour"] <= 23)]
df = df[(df["day_of_week"] >= 0) & (df["day_of_week"] <= 6)]

# Sidebar navigation
st.sidebar.header("Controls")
page = st.sidebar.radio("Select Page", ["01 Introduction", "02 Data Visualization", "03 Prediction"])

# Page 01: Introduction
if page == "01 Introduction":
    st.header("Project Overview")
    st.write(
        """
This project predicts **departure_delay_m** (departure delay in minutes)
using a Linear Regression model.

A train first arrives at a station and then departs.
After arrival, we already know operational and time-related information.
We use that information to predict how late the train will depart.
        """
    )

    # Variables explanation
    st.subheader("Variables Used")

    st.write("**Target (y):**")
    st.write("- departure_delay_m (minutes)")

    st.write("**Features (X): 8 variables**")
    st.write("""
    - arrival_delay_m  
    - planned_dwell_m  
    - category  
    - hour  
    - line  
    - day_of_week  
    - is_peak  
    - arrival_delay_flag  
    """)

    # Dataset info
    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total rows (raw)", f"{len(raw):,}")
    col2.metric("Rows used (cleaned)", f"{len(df):,}")
    col3.metric("Number of features", "8")

    # Missing values
    st.subheader("Missing Values (after cleaning)")

    missing = df.isna().sum()
    st.dataframe(missing)

    # Target statistics
    st.subheader("Target Variable Summary")
    st.write("Summary statistics for departure_delay_m:")
    st.dataframe(df["departure_delay_m"].describe())

    # Raw data sample
    st.subheader("Raw Dataset Sample (first 20 rows)")
    st.dataframe(raw.head(20))

    # Modeling dataset sample
    st.subheader("Modeling Dataset Sample (first 20 rows)")
    st.dataframe(df.head(20))

# Page 02: Data Visualization
elif page == "02 Data Visualization":
    st.subheader("1) Departure delay distribution")
    fig = plt.figure()
    plt.hist(df["departure_delay_m"], bins=50)
    plt.xlabel("departure_delay_m (minutes)")
    plt.ylabel("count")
    st.pyplot(fig)

    st.subheader("2) Arrival delay vs Departure delay")
    fig = plt.figure()
    plt.scatter(df["arrival_delay_m"], df["departure_delay_m"], s=8, alpha=0.3)
    plt.xlabel("arrival_delay_m (minutes)")
    plt.ylabel("departure_delay_m (minutes)")
    st.pyplot(fig)

    st.subheader("3) Planned dwell time vs Departure delay")
    fig = plt.figure()
    plt.scatter(df["planned_dwell_m"], df["departure_delay_m"], s=8, alpha=0.3)
    plt.xlabel("planned_dwell_m (minutes)")
    plt.ylabel("departure_delay_m (minutes)")
    st.pyplot(fig)

    st.subheader("4) Correlation heatmap (target + features)")
    fig = plt.figure(figsize=(10, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f")
    st.pyplot(fig)

# Page 03: Prediction
else:
    st.subheader("Train Linear Regression model")

    feature_cols = [
        "arrival_delay_m",
        "planned_dwell_m",
        "category",
        "hour",
        "line",
        "day_of_week",
        "is_peak",
        "arrival_delay_flag",
    ]

    X = df[feature_cols]
    y = df["departure_delay_m"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    c1, c2 = st.columns(2)
    c1.metric("MAE (minutes)", f"{mae:.2f}")
    c2.metric("R²", f"{r2:.3f}")

    st.subheader("Coefficients (how each feature affects prediction)")
    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": model.coef_})
    st.dataframe(coef_df)

    st.subheader("Actual vs Predicted")
    max_points = st.slider("Max points to plot", 2000, 50000, 20000, step=2000)

    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)

    if len(y_test_arr) > max_points:
        idx = np.random.choice(len(y_test_arr), size=max_points, replace=False)
        y_plot = y_test_arr[idx]
        p_plot = y_pred_arr[idx]
    else:
        y_plot = y_test_arr
        p_plot = y_pred_arr

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(p_plot, y_plot, s=10, alpha=0.3)
    mn = min(p_plot.min(), y_plot.min())
    mx = max(p_plot.max(), y_plot.max())
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=3)
    plt.title("Actual vs Predicted (departure_delay_m)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Try your own inputs")
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        arrival_delay = st.number_input("arrival_delay_m (0-180)", 0.0, 180.0, 5.0)
    with col2:
        dwell = st.number_input("planned_dwell_m (0-60)", 0.0, 60.0, 2.0)
    with col3:
        category = st.number_input("category (1-7)", 1, 7, 3)
    with col4:
        hour = st.number_input("hour (0-23)", 0, 23, 8)

    col5, col6 = st.columns(2)
    with col5:
        line = st.number_input("line", 0, 5000, 1)
    with col6:
        day_of_week = st.number_input("day_of_week (0=Mon ... 6=Sun)", 0, 6, 1)

    col7, col8 = st.columns(2)
    with col7:
        is_peak = st.number_input("is_peak (0/1)", 0, 1, 0)
    with col8:
        arrival_delay_flag = st.number_input("arrival_delay_flag (0/1)", 0, 1, 0)

    new_X = pd.DataFrame([{
        "arrival_delay_m": arrival_delay,
        "planned_dwell_m": dwell,
        "category": category,
        "hour": hour,
        "line": line,
        "day_of_week": day_of_week,
        "is_peak": is_peak,
        "arrival_delay_flag": arrival_delay_flag
    }])

    pred_one = model.predict(new_X)[0]
    st.write(f"Predicted departure_delay_m: **{pred_one:.1f} minutes**")
