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

# 1) Basic settings
st.set_page_config(page_title="Deutsche Bahn Delay Project", layout="wide")
st.title("🚆 Deutsche Bahn Departure Delay Predictor")
st.write("We predict **departure delay (minutes)** using a simple **Linear Regression** model.")

# 2) Load data 
DATA_URL = "https://github.com/hz3396/DBDelayPredict/releases/download/v1.0/db_sample.csv"
DATA_FILE = "db_sample.csv"

def load_data():
    # download once
    if not os.path.exists(DATA_FILE):
        st.write("Downloading default dataset (first time only)...")
        r = requests.get(DATA_URL)
        with open(DATA_FILE, "wb") as f:
            f.write(r.content)

    df = pd.read_csv(DATA_FILE)
    return df

raw = load_data()

# 3) Build features for regression
# Target: departure_delay_m
# Features:
#  - arrival_delay_m
#  - planned_dwell_m = (departure_plan - arrival_plan) in minutes
#  - category
#  - hour from arrival_plan

# Convert time columns
raw["arrival_plan"] = pd.to_datetime(raw["arrival_plan"], errors="coerce")
raw["departure_plan"] = pd.to_datetime(raw["departure_plan"], errors="coerce")

# Features
raw["hour"] = raw["arrival_plan"].dt.hour
raw["planned_dwell_m"] = (raw["departure_plan"] - raw["arrival_plan"]).dt.total_seconds() / 60

# Keep needed columns
df = raw[["departure_delay_m", "arrival_delay_m", "planned_dwell_m", "category", "hour"]].copy()

# Convert to numeric (safe)
for col in ["departure_delay_m", "arrival_delay_m", "planned_dwell_m", "category", "hour"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop missing
df = df.dropna()

# Simple filtering (keeps plots/model stable)
df = df[(df["departure_delay_m"] >= 0) & (df["departure_delay_m"] <= 180)]
df = df[(df["arrival_delay_m"] >= 0) & (df["arrival_delay_m"] <= 180)]
df = df[(df["planned_dwell_m"] >= 0) & (df["planned_dwell_m"] <= 60)]
df = df[(df["category"] >= 1) & (df["category"] <= 7)]
df = df[(df["hour"] >= 0) & (df["hour"] <= 23)]

# 4) Sidebar navigation
st.sidebar.header("Page Selection")
page = st.sidebar.radio(page = st.sidebar.radio(label="Select Page", options=["01 Introduction", "02 Data Visualization", "03 Prediction"]))

# Page 01: Introduction
if page == "01 Introduction":
    st.subheader("Business idea (easy story)")
    st.write(
        """
A train **arrives first** and then **departs**.  
After it arrives, we already know:
- how late it arrived (`arrival_delay_m`)
- how long it is planned to stop (`planned_dwell_m`)
- what type of station it is (`category`)
- what time of day it is (`hour`)

So we predict the **departure delay** (`departure_delay_m`).
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in raw data", f"{len(raw):,}")
    c2.metric("Rows used (after cleaning)", f"{len(df):,}")
    c3.metric("Columns used", "5 columns")

    st.subheader("Dataset preview (raw)")
    st.dataframe(raw.head(20))

    st.subheader("Modeling table preview (cleaned)")
    st.dataframe(df.head(20))

    st.subheader("Variables used in Linear Regression")
    st.write("**Target (y):** departure_delay_m")
    st.write("**Features (X):** arrival_delay_m, planned_dwell_m, category, hour")

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

    st.subheader("4) Departure delay by station category")
    fig = plt.figure()
    cats = sorted(df["category"].unique())
    data = [df[df["category"] == c]["departure_delay_m"] for c in cats]
    plt.boxplot(data, labels=cats)
    plt.xlabel("category (1=hub ... 7=small station)")
    plt.ylabel("departure_delay_m (minutes)")
    st.pyplot(fig)

    st.subheader("5) Correlation heatmap")
    fig = plt.figure()
    corr = df[["departure_delay_m", "arrival_delay_m", "planned_dwell_m", "category", "hour"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f")
    st.pyplot(fig)

# Page 03: Prediction (Linear Regression)
else:
    st.subheader("Train Linear Regression model")

    X = df[["arrival_delay_m", "planned_dwell_m", "category", "hour"]]
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

    st.subheader("Model coefficients")
    coef_df = pd.DataFrame({
        "feature": ["arrival_delay_m", "planned_dwell_m", "category", "hour"],
        "coefficient": model.coef_
    })
    st.dataframe(coef_df)

    st.subheader("Actual vs Predicted (required plot)")
    # Plot a sample for speed (too many points can be slow)
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
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=3)  # y = x line
    plt.title("Actual vs Predicted")
    plt.xlabel("Predicted departure_delay_m")
    plt.ylabel("Actual departure_delay_m")
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

    new_X = pd.DataFrame([{
        "arrival_delay_m": arrival_delay,
        "planned_dwell_m": dwell,
        "category": category,
        "hour": hour
    }])

    pred_one = model.predict(new_X)[0]
    st.write(f"✅ Predicted departure_delay_m: **{pred_one:.1f} minutes**")
