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
    r = requests.get(DATA_URL)
    with open(DATA_FILE, "wb") as f:
        f.write(r.content)

raw = pd.read_csv(DATA_FILE)

def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw2 = raw_df.copy()

    # Convert time columns
    raw2["arrival_plan"] = pd.to_datetime(raw2["arrival_plan"], errors="coerce")
    raw2["departure_plan"] = pd.to_datetime(raw2["departure_plan"], errors="coerce")

    # Time features
    raw2["hour"] = raw2["arrival_plan"].dt.hour
    raw2["day_of_week"] = raw2["arrival_plan"].dt.dayofweek

    # planned dwell time in minutes
    raw2["planned_dwell_m"] = (raw2["departure_plan"] - raw2["arrival_plan"]).dt.total_seconds() / 60

    # simple flags
    raw2["is_peak"] = raw2["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    raw2["arrival_delay_flag"] = (raw2["arrival_delay_m"] > 6).astype(int)

    # Keep target + ALL future-use columns
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
        "station",
        "state",
        "city",
        "info",
    ]

    # Only keep columns that actually exist (prevents KeyError if a column is missing)
    cols_existing = [c for c in cols if c in raw2.columns]
    df = raw2[cols_existing].copy()

    # Convert ONLY numeric columns
    numeric_cols = [
        "departure_delay_m",
        "arrival_delay_m",
        "planned_dwell_m",
        "category",
        "hour",
        "line",
        "day_of_week",
        "is_peak",
        "arrival_delay_flag",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing critical numeric columns (so model/plots don’t crash)
    df = df.dropna(subset=numeric_cols)

    # Basic filtering on numeric columns only
    df = df[(df["departure_delay_m"] >= 0) & (df["departure_delay_m"] <= 180)]
    df = df[(df["arrival_delay_m"] >= 0) & (df["arrival_delay_m"] <= 180)]
    df = df[(df["planned_dwell_m"] >= 0) & (df["planned_dwell_m"] <= 60)]
    df = df[(df["category"] >= 1) & (df["category"] <= 7)]
    df = df[(df["hour"] >= 0) & (df["hour"] <= 23)]
    df = df[(df["day_of_week"] >= 0) & (df["day_of_week"] <= 6)]

    # Clean text columns (optional but helpful)
    text_cols = ["station", "state", "city", "info"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()

    return df
    
df = clean_data(raw)
# Sidebar navigation
st.sidebar.header("Controls")
page = st.sidebar.radio("Select Page", ["01 Introduction", "02 Data Visualization", "03 Prediction"])

# Page 01: Introduction
st.image("Weixin Image_2026-03-02_181855_878.jpg", width=1500)
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

    # Dataset info
    st.subheader("Dataset Information")

    # 📊 DATA PREVIEW
    # ------------------------------
    st.markdown("##### Data Preview")

    rows = st.slider("Select a number of rows to display", 5, 100, 20)
    view = st.radio("View from top, bottom, or randomized", ["Head", "Tail", "Random"])

    rows = st.slider("Select a number of rows to display", 5, 100, 20)
    view = st.radio("View from top, bottom, or randomized", ["Head", "Tail", "Random"])
    
    if view == "Head":
        st.dataframe(raw.head(rows), use_container_width=True)
    elif view == "Tail":
        st.dataframe(raw.tail(rows), use_container_width=True)
    elif view == "Random":
        st.dataframe(raw.sample(n=rows), use_container_width=True)
    
    st.caption(f"This RAW data frame has {raw.shape[0]} rows and {raw.shape[1]} columns.")
    

    # ------------------------------
    # ❗ MISSING VALUES
    # ------------------------------
    st.markdown("##### Missing values")

    missing = df.isnull().sum()
    total_cells = df.shape[0] * df.shape[1]
    total_missing = missing.sum()
    missing_pct = (total_missing / total_cells) * 100

    st.write(missing)
    st.markdown(f"**Percentage of total missing values:** {missing_pct:.1f} %")

    if missing_pct < 1:
        st.success("✅ Missing values are extremely low. The data is safe to use.")
    elif 1 <= missing_pct <= 5:
        st.warning("⚠️ You have missing data. Please double check before proceeding.")
    else:
        st.error("🚨 You have a higher percentage of missing data. Please double check.")

    # ------------------------------
    # 📈 SUMMARY STATISTICS
    # ------------------------------
    st.markdown("##### 📈 Summary Statistics")
    st.dataframe(df.describe())
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total rows (raw)", f"{len(raw):,}")
    col2.metric("Rows used (cleaned)", f"{len(df):,}")
    col3.metric("Number of features", "8")

    # Missing values
    st.subheader("Missing Values (after cleaning)")

    missing = df.isna().sum()
    st.dataframe(missing)


# Page 02: Data Visualization
elif page == "02 Data Visualization":
    st.subheader("1) Departure delay distribution")
    fig = plt.figure(figsize=(7, 4))
    plt.hist(df["departure_delay_m"], bins=50)
    plt.xlabel("departure_delay_m (minutes)")
    plt.ylabel("count")
    st.pyplot(fig, use_container_width=False)
    
    st.subheader("2) Arrival delay vs Departure delay")
    fig = plt.figure(figsize=(7, 4))
    plt.scatter(df["arrival_delay_m"], df["departure_delay_m"], s=8, alpha=0.3)
    plt.xlabel("arrival_delay_m (minutes)")
    plt.ylabel("departure_delay_m (minutes)")
    st.pyplot(fig, use_container_width=False)
    
    st.subheader("3) Planned dwell time vs Departure delay")
    fig = plt.figure(figsize=(7, 4))
    plt.scatter(df["planned_dwell_m"], df["departure_delay_m"], s=8, alpha=0.3)
    plt.xlabel("planned_dwell_m (minutes)")
    plt.ylabel("departure_delay_m (minutes)")
    st.pyplot(fig, use_container_width=False)
    
    st.subheader("4) Correlation heatmap (target + features)")
    fig = plt.figure(figsize=(8, 5))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f")
    st.pyplot(fig, use_container_width=False)

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
