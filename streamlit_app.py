import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("dark_background")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# App setup
st.set_page_config(page_title="Deutsche Bahn Delay Project", layout="wide")
st.title("🚄 DB Train Departure Delay Predictor")
st.write("Haochen Zhang, William Zheng, Tianlai Zhang")

#Theme
# Dark style
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0b0f1a 0%, #0e1117 100%);
}
header {
    background: transparent !important;
}
h1, h2, h3, h4 {
    color: white;
    font-weight: 700;
}
p, span, label, div {
    color: #d1d5db;
}
[data-testid="stSidebar"] {
    background: #0f172a;
}

[data-testid="stSidebar"] * {
    color: white;
}
[data-testid="stDataFrame"] {
    background: #111827;
    border-radius: 10px;
}
[data-testid="stImage"],
[data-testid="stPlotlyChart"],
[data-testid="stTable"] {
    background: #111827;
    border-radius: 12px;
    padding: 10px;
}
[data-testid="stMetric"] {
    background: #111827;
    padding: 15px;
    border-radius: 10px;
}
[data-testid="stMetricValue"] {
    color: #60a5fa;
    font-weight: bold;
}
.stSlider {
    color: white;
}
[data-testid="stDataFrame"] div {
    color: white;
}
button[kind="secondary"] {
    background: #1f2937;
    border-radius: 8px;
}

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #374151;
    border-radius: 5px;
}

</style>
""", unsafe_allow_html=True)
st.write("""Haochen Zhang, William Zheng, Tianlai Zhang""")

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

# ✅ Rename columns (ONLY variable name changes)
raw = raw.rename(
    columns={
        "arrival_delay_m": "arrival_delay_time",
        "departure_delay_m": "departure_delay_time",
        "arrival_delay_flag": "arrival_delay_severe_or_not",
        "category": "station_category",
    }
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
    raw2["arrival_delay_severe_or_not"] = (raw2["arrival_delay_time"] > 6).astype(int)

    # Encode line (string like '1', 'X85F') into integer
    le_line = LabelEncoder()
    raw2["line"] = le_line.fit_transform(raw2["line"].astype(str))

    # Keep target + future-use columns (including lat/long for map)
    cols = [
        "departure_delay_time",  # target
        "arrival_delay_time",
        "planned_dwell_m",
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
        "lat",
        "long",
    ]

    cols_existing = [c for c in cols if c in raw2.columns]
    df = raw2[cols_existing].copy()

    # Convert numeric columns
    numeric_cols = [
        "departure_delay_time",
        "arrival_delay_time",
        "planned_dwell_m",
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

    # Drop rows missing critical numeric columns
    df = df.dropna(subset=numeric_cols)

    # Basic filtering
    df = df[(df["departure_delay_time"] >= 0) & (df["departure_delay_time"] <= 180)]
    df = df[(df["arrival_delay_time"] >= 0) & (df["arrival_delay_time"] <= 180)]
    df = df[(df["planned_dwell_m"] >= 0) & (df["planned_dwell_m"] <= 60)]
    df = df[(df["station_category"] >= 1) & (df["station_category"] <= 7)]
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
page = st.sidebar.radio("**Page Selection**", ["01 Introduction", "02 Data Visualization", "03 Prediction"])

# Page 01: Introduction
if page == "01 Introduction":
    try:
        st.image("Weixin Image_2026-03-02_181855_878.jpg", width=1500)
    except Exception:
        pass
    st.header("Project Overview")
    st.write(
        """
Deutsche Bahn (DB) is a state-owned railway company in Germany, responsible for operating most of the trains within the country. Due to insufficient government investment, aging infrastructure, poor scheduling, and the sharing of tracks among different types of trains, the on-time performance of Deutsche Bahn trains has been far lower than that of other major railway companies in Europe recently, such as SNCF (France), SBB (Switzerland), NS (Netherlands), and SNCB (Belgium), etc.

We obtained all the train operation data from Deutsche Bahn company from July 2024 to July 2025 (approximately 1 million entries), and randomly selected 200,000 of them for analysis.

When a train arrives at a certain station, we can obtain relevant operational and time-related information. Subsequently, we used this information to predict how late the train would depart from this station.

This project will predict **departure_delay_time** (departure delay in minutes) using a Linear Regression model.
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

    st.markdown("##### Missing Values (Raw Data)")
    missing_nan = raw.isnull().sum()

    obj_cols = raw.select_dtypes(include="object").columns
    missing_blank = pd.Series(0, index=raw.columns)
    for c in obj_cols:
        missing_blank[c] = raw[c].astype(str).str.strip().eq("").sum()

    missing_total = missing_nan + missing_blank
    total_cells = raw.shape[0] * raw.shape[1]
    total_missing = int(missing_total.sum())
    missing_pct = (total_missing / total_cells) * 100

    st.write("Missing Values by Column:")
    st.dataframe(missing_nan)

    st.markdown("##### 📈 Summary Statistics")
    st.dataframe(df.describe())

    col1, col2 = st.columns(2)
    col1.metric("Total rows (raw)", f"{len(raw):,}")
    col2.metric("Rows used (cleaned)", f"{len(df):,}")

    st.markdown("##### Missing Values (After Cleaning)")
    st.dataframe(df.isna().sum())


elif page == "02 Data Visualization":
    try:
        st.image("02.jpg", width=1500)
    except Exception:
        pass

    # ── KPI cards at the top ──
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Trains", f"{len(df):,}")
    k2.metric("Avg Delay", f"{df['departure_delay_time'].mean():.2f} min")
    k3.metric("On-Time Rate", f"{(df['departure_delay_time'] == 0).mean()*100:.1f}%")
    k4.metric("Delay Rate", f"{(df['departure_delay_time'] > 0).mean()*100:.1f}%")
    st.markdown("---")

    # Chart 1: Histogram of departure delays, filtered to 1 to 20 minutes
    st.subheader("1) Departure Delay Distribution")
    st.markdown("This histogram shows how many trains fall into each delay bucket between 1 and 20 minutes. Short delays are the most common. The x axis shows delay in minutes and the y axis shows the count of trains.")
    fig = plt.figure(figsize=(7, 4))
    plt.hist(df["departure_delay_time"], bins=50, range=(1, 20))
    plt.xlim(1, 20)
    plt.xlabel("departure_delay_time (minutes)")
    plt.ylabel("count")
    st.pyplot(fig)

    # Chart 2: Scatter plot comparing arrival delay and departure delay
    st.subheader("2) Arrival Delay vs Departure Delay")
    st.markdown("Each dot represents one train. If a train arrives late it almost always departs late too. The tight upward cluster confirms a strong positive relationship between arrival delay and departure delay.")
    fig = plt.figure(figsize=(7, 5))
    delay_df = df[(df["arrival_delay_time"] > 0) & (df["departure_delay_time"] > 0)]
    sample_size = min(3000, len(delay_df))
    sample = delay_df.sample(sample_size, random_state=42)
    plt.scatter(sample["departure_delay_time"], sample["arrival_delay_time"], alpha=0.3, s=10, color=T["accent"])
    plt.xlabel("Departure Delay (min)")
    plt.ylabel("Arrival Delay (min)")
    st.pyplot(fig)

    # Chart 3: Pie chart showing the ratio of on time vs delayed departures
    st.subheader("3) On Time vs Delayed Departures")
    st.markdown("Green = on time, red = delayed. Despite DB's reputation the majority of departures still happen on schedule, though the delayed slice represents a very large number of trains in absolute terms.")
    on_time_count = (df["departure_delay_time"] == 0).sum()
    delay_count = (df["departure_delay_time"] > 0).sum()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie([on_time_count, delay_count], labels=["on_time", "delay"], autopct="%1.1f%%", colors=[T["bar_good"], T["bar_bad"]])
    st.pyplot(fig)

    # Chart 4: Bar chart ranking states by delay rate percentage
    st.subheader("4) Departure Delay Rate by State")
    st.markdown("Each bar shows the departure delay rate for one German state. States at the top have the worst punctuality. Comparing this to Chart 5 shows that high train volume does not always mean the highest delay rate.")
    state_delay = df.groupby("state")["departure_delay_time"].apply(lambda x: (x > 0).mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=state_delay.values, y=state_delay.index, palette="Reds_r", ax=ax)
    ax.set_xlabel("Delay Rate (%)")
    ax.set_ylabel("State")
    st.pyplot(fig)

    # Chart 5: Bar chart showing train volume per state
    st.subheader("5) Number of Trains by State")
    st.markdown("This chart shows how many trains operate in each German state. States with more traffic naturally have more scheduling pressure, which can lead to more opportunities for cascading delays.")
    state_counts = df["state"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=state_counts.values, y=state_counts.index, palette="viridis", ax=ax)
    ax.set_xlabel("Number of Trains")
    ax.set_ylabel("State")
    st.pyplot(fig)

    # Chart 6: Scatter plot comparing planned dwell time and departure delay
    st.subheader("6) Planned Dwell Time vs Departure Delay")
    st.markdown("Trains with longer planned dwell times at a station tend to have smaller departure delays. A generous stop duration gives slack to recover. Trains with very short stops have almost no buffer at all.")
    fig = plt.figure(figsize=(7, 4))
    plt.scatter(df["planned_dwell_m"], df["departure_delay_time"], s=8, alpha=0.3, color=T["accent"])
    plt.xlabel("planned_dwell_m (minutes)")
    plt.ylabel("departure_delay_time (minutes)")
    st.pyplot(fig, use_container_width=False)

    # Chart 7: Correlation heatmap of all numeric variables
    st.subheader("7) Correlation Heatmap (Target + Features)")
    st.markdown("Values close to +1 or -1 mean a strong relationship and values close to 0 mean almost none. The strongest signal is between arrival_delay_time and departure_delay_time at 0.98, which confirms the data is internally consistent.")
    fig = plt.figure(figsize=(8, 5))
    corr = df.select_dtypes(include="number").corr()
    sns.heatmap(corr, annot=True, fmt=".2f")
    st.pyplot(fig, use_container_width=False)

    st.markdown("---")

    # Chart 8: Delay Hotspot Map
    st.subheader("8) Station Delay Hotspot Map")
    st.markdown("Each dot is a station. Red = high average delay, green = mostly on time. The size of the dot also scales with delay severity. Hover over any dot to see the station name, city, state and average delay in minutes.")
    if "lat" in df.columns and "long" in df.columns:
        station_map = df.dropna(subset=["lat", "long"]).groupby(
            ["station", "lat", "long", "city", "state"]
        ).agg(
            avg_delay=("departure_delay_time", "mean"),
            total_trains=("departure_delay_time", "count"),
        ).reset_index()
        station_map = station_map[station_map["total_trains"] >= 20].copy()
        station_map["avg_delay"] = station_map["avg_delay"].round(2)
        max_d = station_map["avg_delay"].max()
        station_map["r"] = (station_map["avg_delay"] / max_d * 255).clip(0, 255).astype(int)
        station_map["g"] = (255 - station_map["r"]).astype(int)
        station_map["b"] = 50
        station_map["radius"] = (station_map["avg_delay"] * 350).clip(300, 6000)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=station_map,
            get_position="[long, lat]",
            get_radius="radius",
            get_fill_color="[r, g, b, 190]",
            pickable=True,
        )
        view_state = pdk.ViewState(latitude=51.3, longitude=10.5, zoom=5.5)
        tooltip = {
            "html": "<b>{station}</b><br/>City: {city}<br/>State: {state}<br/>Avg Delay: {avg_delay} min<br/>Trains: {total_trains}",
            "style": {"backgroundColor": "#1e2130", "color": "white", "fontSize": "13px"},
        }
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style=T["map_style"]))
    else:
        st.info("Location data not available in the cleaned dataset.")

    # Chart 9: Average delay by hour of day
    st.subheader("9) Avg Delay by Hour of Day")
    st.markdown("This line chart shows the average departure delay at each hour of the day. Delays are lowest around 4 to 5 AM when networks are quiet and peak around 5 to 6 PM during evening rush hour.")
    hourly = df.groupby("hour")["departure_delay_time"].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(hourly.index, hourly.values, alpha=0.25, color=T["accent"])
    ax.plot(hourly.index, hourly.values, color=T["accent"], linewidth=2.5, marker="o", markersize=5)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Avg Departure Delay (min)")
    ax.set_xticks(range(0, 24))
    st.pyplot(fig)

    # Chart 10: Average delay by day of week
    st.subheader("10) Avg Delay by Day of Week")
    st.markdown("Red bars are above the weekly average and green bars are below it. Weekend days tend to have fewer delays. ")
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_delay = df.groupby("day_of_week")["departure_delay_time"].mean()
    avg_d = day_delay.mean()
    colors_d = [T["bar_bad"] if v > avg_d else T["bar_good"] for v in day_delay.values]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(day_delay.index, day_delay.values, color=colors_d, edgecolor="#37415144", linewidth=0.5)
    ax.axhline(avg_d, color=T["accent"], linestyle="--", alpha=0.7, linewidth=1.2, label=f"Weekly avg: {avg_d:.2f} min")
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_labels)
    ax.set_ylabel("Avg Departure Delay (min)")
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")

    # Station Delay Lookup
    st.subheader("11) Station Delay Lookup")
    st.markdown("Select a station from the dropdown to see its delay statistics compared to the overall network average. You can also see how the average delay changes by hour for that specific station.")
    if "station" in df.columns:
        all_stations = sorted(df["station"].dropna().unique())
        selected_station = st.selectbox("Select a station", all_stations)
        sdf = df[df["station"] == selected_station]
        if len(sdf) > 0:
            s_avg = sdf["departure_delay_time"].mean()
            s_rate = (sdf["departure_delay_time"] > 0).mean() * 100
            overall_avg = df["departure_delay_time"].mean()
            s_city = sdf["city"].iloc[0] if "city" in sdf.columns else "N/A"
            s_state = sdf["state"].iloc[0] if "state" in sdf.columns else "N/A"
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Avg Delay", f"{s_avg:.2f} min", f"{s_avg - overall_avg:+.2f} vs network avg")
            sc2.metric("Delay Rate", f"{s_rate:.1f}%")
            sc3.metric("Total Trains", f"{len(sdf):,}")
            sc4.metric("City", s_city)
            if len(sdf) >= 10:
                hourly_s = sdf.groupby("hour")["departure_delay_time"].mean()
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.fill_between(hourly_s.index, hourly_s.values, alpha=0.25, color=T["accent2"])
                ax.plot(hourly_s.index, hourly_s.values, color=T["accent2"], linewidth=2, marker="o", markersize=4)
                ax.set_xlabel("Hour of Day")
                ax.set_ylabel("Avg Delay (min)")
                ax.set_title(f"Delay by Hour at {selected_station}, {s_state}")
                ax.set_xticks(range(0, 24))
                st.pyplot(fig)


# =========================
# Page 03: Prediction (Route 2, simplified & stable)
# =========================
else:
    st.subheader("Train Linear Regression Model")
    try:
        st.image("03.jpg", width=1500)
    except Exception:
        pass

    feature_cols = [
        "arrival_delay_time",
        "planned_dwell_m",
        "station_category",
        "hour",
        "line",
        "day_of_week",
        "is_peak",
        "arrival_delay_severe_or_not",
    ]

    X = df[feature_cols]
    y = df["departure_delay_time"]

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
    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": model.coef_}).sort_values("coefficient", key=abs, ascending=False)
    st.dataframe(coef_df)

    # Feature importance bar chart
    st.subheader(" Feature Importance")
    st.markdown("The longer the bar, the more that feature influences the predicted departure delay. The highlighted bar is the single most important feature. arrival_delay_time dominates because if a train arrives late it will almost certainly leave late too.")
    importance = pd.Series(np.abs(model.coef_), index=feature_cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = [T["accent2"] if i == len(importance) - 1 else T["accent"] for i in range(len(importance))]
    ax.barh(importance.index, importance.values, color=bar_colors, edgecolor="#1f293744", linewidth=0.5)
    ax.set_xlabel("Absolute Coefficient")
    st.pyplot(fig)

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
    plt.scatter(p_plot, y_plot, s=10, alpha=0.3, color=T["accent"])
    mn = min(p_plot.min(), y_plot.min())
    mx = max(p_plot.max(), y_plot.max())
    plt.plot([mn, mx], [mn, mx], color=T["accent2"], linestyle="--", linewidth=3)
    plt.title("Actual vs Predicted (departure_delay_time)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Try your own inputs")

    # Binary toggles (0/1)
    colT1, colT2 = st.columns(2)
    with colT1:
        is_peak = int(st.toggle("Is Peak Hour?", key="inp_is_peak"))
    with colT2:
        arrival_delay_severe_or_not = int(st.toggle("Arrival Delay Severe?", key="inp_arrival_delay_severe_or_not"))

    # Row 1
    col1, col2, col3 = st.columns(3)
    with col1:
        arrival_delay_time = st.slider("arrival_delay_time (0-180)", 0.0, 180.0, 5.0, key="inp_arrival_delay_time")
    with col2:
        dwell = st.slider("planned_dwell_m (0-60)", 0.0, 60.0, 2.0, key="inp_dwell")
    with col3:
        station_category = st.slider("station_category (1-7)", 1, 7, 3, key="inp_station_category")

    # Row 2
    col4, col5, col6 = st.columns(3)
    with col4:
        hour = st.slider("hour (0-23)", 0, 23, 8, key="inp_hour")
    with col5:
        day_of_week = st.slider("day_of_week (0=Mon ... 6=Sun)", 0, 6, 1, key="inp_day_of_week")
    with col6:
        line = st.slider("line (encoded)", 0, 287, 0, key="inp_line")

    # Build input row with ALL features
    new_X = pd.DataFrame([{
        "arrival_delay_time": arrival_delay_time,
        "planned_dwell_m": dwell,
        "station_category": station_category,
        "hour": hour,
        "line": line,
        "day_of_week": day_of_week,
        "is_peak": is_peak,
        "arrival_delay_severe_or_not": arrival_delay_severe_or_not,
    }])
    new_X = new_X[feature_cols]

    pred_one = model.predict(new_X)[0]
    st.success(f"Predicted departure delay: **{pred_one:.1f} minutes**")
