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

# Load data (GitHub Release)
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

# Cleaning
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

    # Keep target + future-use columns
    cols = [
        "departure_delay_m",  # target
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

    cols_existing = [c for c in cols if c in raw2.columns]
    df = raw2[cols_existing].copy()

    # Convert numeric columns
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

    # Drop rows missing critical numeric columns
    df = df.dropna(subset=numeric_cols)

    # Basic filtering
    df = df[(df["departure_delay_m"] >= 0) & (df["departure_delay_m"] <= 180)]
    df = df[(df["arrival_delay_m"] >= 0) & (df["arrival_delay_m"] <= 180)]
    df = df[(df["planned_dwell_m"] >= 0) & (df["planned_dwell_m"] <= 60)]
    df = df[(df["category"] >= 1) & (df["category"] <= 7)]
    df = df[(df["hour"] >= 0) & (df["hour"] <= 23)]
    df = df[(df["day_of_week"] >= 0) & (df["day_of_week"] <= 6)]

    # Clean text columns
    text_cols = ["station", "state", "city", "info"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()

    return df

df = clean_data(raw)

# Sidebar navigation
st.sidebar.header("Controls")
page = st.sidebar.radio("Select Page", ["01 Introduction", "02 Data Visualization", "03 Prediction"])

# Page 01: Introduction
if page == "01 Introduction":
    st.image("Weixin Image_2026-03-02_181855_878.jpg", width=1500)
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

    st.subheader("Dataset Information")

    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display", 5, 100, 20)
    view = st.radio("View from top, bottom, or randomized", ["Head", "Tail", "Random"])

    if view == "Head":
        st.dataframe(raw.head(rows), width="stretch")
    elif view == "Tail":
        st.dataframe(raw.tail(rows), width="stretch")
    else:
        st.dataframe(raw.sample(n=rows, random_state=42), width="stretch")

    st.caption(f"This RAW data frame has {raw.shape[0]} rows and {raw.shape[1]} columns.")

    st.markdown("##### Missing values (raw data)")
    missing_nan = raw.isnull().sum()

    obj_cols = raw.select_dtypes(include="object").columns
    missing_blank = pd.Series(0, index=raw.columns)
    for c in obj_cols:
        missing_blank[c] = raw[c].astype(str).str.strip().eq("").sum()

    missing_total = missing_nan + missing_blank
    total_cells = raw.shape[0] * raw.shape[1]
    total_missing = int(missing_total.sum())
    missing_pct = (total_missing / total_cells) * 100

    st.write("NaN missing values by column:")
    st.dataframe(missing_nan)

    st.write("Blank-string missing values by column (text columns):")
    st.dataframe(missing_blank[missing_blank > 0])

    st.write("Total missing (NaN + blanks) by column:")
    st.dataframe(missing_total[missing_total > 0])

    st.markdown(f"**Percentage of total missing values (NaN + blanks):** {missing_pct:.1f} %")

    if missing_pct < 1:
        st.success("✅ Missing values are extremely low in the raw data.")
    elif 1 <= missing_pct <= 5:
        st.warning("⚠️ Some missing data exists in the raw dataset.")
    else:
        st.error("🚨 High percentage of missing data in the raw dataset.")

    st.markdown("##### 📈 Summary Statistics")
    st.dataframe(df.describe())

    col1, col2 = st.columns(2)
    col1.metric("Total rows (raw)", f"{len(raw):,}")
    col2.metric("Rows used (cleaned)", f"{len(df):,}")

    st.subheader("Missing Values (after cleaning)")
    st.dataframe(df.isna().sum())


elif page == "02 Data Visualization":
    st.image("02.jpg", width=1500)


    # Chart 1: Histogram of departure delays, filtered to 1 to 20 minutes
    st.subheader("1) Departure Delay Distribution")
    st.markdown("This histogram shows how many trains fall into each delay range between 1 and 20 minutes. The x axis represents the delay in minutes, and the y axis shows the count of trains. You can see that most delayed trains only have a very short delay, and longer delays are much less common.")
    fig = plt.figure(figsize=(7, 4))
    plt.hist(df["departure_delay_m"], bins=50, range=(1, 20))
    plt.xlim(1, 20)
    plt.xlabel("departure_delay_m (minutes)")
    plt.ylabel("count")
    st.pyplot(fig)

    # Chart 2: Scatter plot comparing arrival delay and departure delay
    st.subheader("2) Arrival Delay vs Departure Delay")
    st.markdown("Each dot represents a train. The x axis is the departure delay and the y axis is the arrival delay. If the dots form a line going upward, it means trains that leave late also tend to arrive late. You can clearly see a positive correlation here.")
    fig = plt.figure(figsize=(7, 5))
    delay_df = df[(df["arrival_delay_m"] > 0) & (df["departure_delay_m"] > 0)]
    sample_size = min(3000, len(delay_df))
    sample = delay_df.sample(sample_size, random_state=42)
    plt.scatter(sample["departure_delay_m"], sample["arrival_delay_m"], alpha=0.3, s=10)
    plt.xlabel("Departure Delay (min)")
    plt.ylabel("Arrival Delay (min)")
    st.pyplot(fig)

    # Chart 3: Pie chart showing the ratio of on time vs delayed departures
    st.subheader("3) On Time vs Delayed Departures")
    st.markdown("This pie chart shows the percentage of trains that departed on time versus those that were delayed. The green portion is on time and the red portion is delayed. You can quickly tell that the vast majority of trains do depart on time.")
    on_time_count = (df["departure_delay_m"] == 0).sum()
    delay_count = (df["departure_delay_m"] > 0).sum()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie([on_time_count, delay_count], labels=["on_time", "delay"], autopct="%1.1f%%", colors=["#4CAF50", "#F44336"])
    st.pyplot(fig)

    # Chart 4: Bar chart ranking states by delay rate percentage
    st.subheader("4) Departure Delay Rate by State")
    st.markdown("Each bar represents a German state, and the length shows its departure delay rate as a percentage. States at the top have the highest delay rates. This helps you compare which regions are more likely to experience train delays.")
    state_delay = df.groupby("state")["departure_delay_m"].apply(lambda x: (x > 0).mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=state_delay.values, y=state_delay.index, palette="Reds_r", ax=ax)
    ax.set_xlabel("Delay Rate (%)")
    ax.set_ylabel("State")
    st.pyplot(fig)

    # Chart 5: Bar chart showing train volume per state
    st.subheader("5) Number of Trains by State")
    st.markdown("This chart shows how many trains operate in each German state. States with longer bars have more train traffic. Keep in mind that states with higher train volume might naturally have more total delays simply because there are more trains running.")
    state_counts = df["state"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=state_counts.values, y=state_counts.index, palette="viridis", ax=ax)
    ax.set_xlabel("Number of Trains")
    ax.set_ylabel("State")
    st.pyplot(fig)

    # Chart 6: Scatter plot comparing planned dwell time and departure delay
    st.subheader("6) Planned Dwell Time vs Departure Delay")
    st.markdown("Each dot represents a train. The x axis is the planned dwell time at the station and the y axis is the departure delay. This helps us see whether trains with shorter or longer planned stops tend to have more delays.")
    fig = plt.figure(figsize=(7, 4))
    plt.scatter(df["planned_dwell_m"], df["departure_delay_m"], s=8, alpha=0.3)
    plt.xlabel("planned_dwell_m (minutes)")
    plt.ylabel("departure_delay_m (minutes)")
    st.pyplot(fig, use_container_width=False)

    # Chart 7: Correlation heatmap of all numeric variables
    st.subheader("7) Correlation Heatmap (Target + Features)")
    st.markdown("This heatmap shows the correlation between all numeric variables. Values close to 1 or negative 1 mean a strong relationship, while values close to 0 mean almost no relationship. The color makes it easy to spot which variables are most related to each other.")
    fig = plt.figure(figsize=(8, 5))
    corr = df.select_dtypes(include="number").corr()
    sns.heatmap(corr, annot=True, fmt=".2f")
    st.pyplot(fig, use_container_width=False)


# =========================
# Page 03: Prediction (Route 2, simplified & stable)
# =========================
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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
    max_points = st.slider(
        "Max points to plot", 2000, 50000, 20000, step=2000, key="max_points_plot"
    )

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
             arrival_delay = st.slider("arrival_delay_m (0-180)",min_value=0.0,max_value=180.0,value=5.0,key="inp_arrival_delay")
        with col2:
            dwell = st.slider("planned_dwell_m (0-60)",min_value=0.0,max_value=60.0,value=2.0,key="inp_dwell")
        with col3:
            category = st.slider("category (1-7)",min_value=1,max_value=7,value=3,key="inp_category")
        with col4:
            hour = st.slider("hour (0-23)",min_value=0,max_value=23,value=8,key="inp_hour")
    
        col5, col6 = st.columns(2)
        with col5:
            line = st.number_input("line", 0, 5000, 1, key="inp_line")
        with col6:
            day_of_week = st.slider("day_of_week (0=Mon ... 6=Sun)",min_value=0,max_value=6,value=1,key="inp_day_of_week")
    
        col7, col8 = st.columns(2)
        with col7:
            is_peak = int(st.toggle("Is Peak Hour?", key="inp_is_peak"))
    
        with col8:
            arrival_delay_flag = int(st.toggle("Arrival Delay Flag?", key="inp_arrival_delay_flag"))
        
    "arrival_delay_m": arrival_delay,
        "planned_dwell_m": dwell,
        "category": category,
        "hour": hour,
        "line": line,
        "day_of_week": day_of_week,
        "is_peak": is_peak,
        "arrival_delay_flag": arrival_delay_flag,
    }])

    pred_one = model.predict(new_X)[0]
    st.write(f"Predicted departure_delay_m: **{pred_one:.1f} minutes**")
