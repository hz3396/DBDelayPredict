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

st.set_page_config(page_title="DB Departure Delay Predictor", layout="wide")
st.title("🚆 DB Departure Delay Predictor (Linear Regression)")
st.caption("Predict departure delay (minutes) using arrival-related info at the same station.")

# Default dataset (GitHub Release)
DATA_URL = "https://github.com/hz3396/DBDelayPredict/releases/download/v1.0/db_sample.csv"
DATA_FILE = "db_sample.csv"

# We will predict departure_delay_m
TARGET_COL = "departure_delay_m"

# Features: arrival_delay_m + planned_dwell_m + category + hour
FEATURE_COLS = ["arrival_delay_m", "planned_dwell_m", "category", "hour"]

REQUIRED_COLS = [
    "arrival_plan",
    "departure_plan",
    "arrival_delay_m",
    "departure_delay_m",
    "category",
]

# =============================
# UI HELPERS
# =============================
def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div style="padding:14px;border-radius:14px;border:1px solid #e6e6e6;">
            <div style="font-size:13px;color:#666;">{label}</div>
            <div style="font-size:22px;font-weight:700;margin-top:6px;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================
# DATA FUNCTIONS
# =============================
@st.cache_data
def download_default_dataset() -> str:
    """Download default CSV from GitHub Release (only if not already present)."""
    if not os.path.exists(DATA_FILE):
        with st.spinner("Downloading default dataset from GitHub Release (first time only)..."):
            r = requests.get(DATA_URL, timeout=180)
            r.raise_for_status()
            with open(DATA_FILE, "wb") as f:
                f.write(r.content)
    return DATA_FILE


@st.cache_data
def load_default_data() -> pd.DataFrame:
    path = download_default_dataset()
    return pd.read_csv(path)


def build_model_table(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build modeling table:
    y = departure_delay_m
    X = arrival_delay_m, planned_dwell_m, category, hour

    planned_dwell_m = (departure_plan - arrival_plan) in minutes
    hour extracted from arrival_plan

    Minimal cleaning to keep model stable.
    """
    df = raw.copy()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse time columns
    df["arrival_plan"] = pd.to_datetime(df["arrival_plan"], errors="coerce")
    df["departure_plan"] = pd.to_datetime(df["departure_plan"], errors="coerce")

    # hour from arrival_plan (could also use departure_plan; keep it simple)
    df["hour"] = df["arrival_plan"].dt.hour

    # planned dwell time in minutes
    df["planned_dwell_m"] = (df["departure_plan"] - df["arrival_plan"]).dt.total_seconds() / 60.0

    # Convert to numeric
    for c in ["arrival_delay_m", "departure_delay_m", "category", "hour", "planned_dwell_m"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep only needed columns
    df = df[[TARGET_COL, "arrival_delay_m", "planned_dwell_m", "category", "hour"]].dropna()

    # Clip to reasonable ranges (helps visuals and avoids extreme outliers)
    df = df[df["arrival_delay_m"].between(0, 180)]
    df = df[df["departure_delay_m"].between(0, 180)]
    df = df[df["category"].between(1, 7)]
    df = df[df["hour"].between(0, 23)]

    # dwell time can be weird if data issues; keep a reasonable band
    df = df[df["planned_dwell_m"].between(0, 60)]

    return df


def train_lr(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    return model, mae, r2, y_test, pred


# =============================
# SIDEBAR 
# =============================
st.sidebar.header("Page Selection")
page = st.sidebar.radio(
    "Select Page",
    ["01 Introduction", "02 Data Visualization", "03 Prediction"],
    index=0
)

# =============================
# LOAD + PREP DATA
# =============================
try:
    raw = load_default_data()
except Exception as e:
    st.error("Failed to download/load the default dataset.")
    st.code(str(e))
    st.stop()

try:
    df = build_model_table(raw)
except Exception as e:
    st.error("Failed to process the dataset (unexpected format or missing columns).")
    st.code(str(e))
    st.stop()


# =============================
# PAGE 01: INTRODUCTION
# =============================
if page == "01 Introduction":
    st.subheader("Project idea (very easy to explain)")
    st.write(
        """
A train **arrives first**, then **departs**.  
So we predict **departure delay** using information known at arrival time.

**Target (y):** `departure_delay_m`  
**Features (X):**
- `arrival_delay_m` (how late the train arrives)
- `planned_dwell_m` (planned stop time = departure_plan - arrival_plan)
- `category` (station category 1–7)
- `hour` (arrival hour from arrival_plan)
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Raw rows", f"{len(raw):,}")
    with c2:
        metric_card("Rows used (cleaned)", f"{len(df):,}")
    with c3:
        metric_card("Raw columns", f"{raw.shape[1]}")
    with c4:
        metric_card("Features", ", ".join(FEATURE_COLS))

    st.subheader("Required columns")
    st.code("\n".join(REQUIRED_COLS))

    st.subheader("Raw preview (first 25 rows)")
    st.dataframe(raw.head(25))

    st.subheader("Modeling preview (first 25 rows)")
    st.dataframe(df.head(25))


# =============================
# PAGE 02: VISUALIZATION
# =============================
elif page == "02 Data Visualization":
    st.subheader("1) Departure delay distribution")
    fig = plt.figure()
    plt.hist(df["departure_delay_m"], bins=50)
    plt.xlabel("departure_delay_m (minutes)")
    plt.ylabel("count")
    st.pyplot(fig)

    st.subheader("2) Arrival delay vs Departure delay")
    fig = plt.figure()
    plt.scatter(df["arrival_delay_m"], df["departure_delay_m"], s=6, alpha=0.35)
    plt.xlabel("arrival_delay_m (minutes)")
    plt.ylabel("departure_delay_m (minutes)")
    st.pyplot(fig)

    st.subheader("3) Planned dwell time vs Departure delay")
    fig = plt.figure()
    plt.scatter(df["planned_dwell_m"], df["departure_delay_m"], s=6, alpha=0.35)
    plt.xlabel("planned_dwell_m (minutes)")
    plt.ylabel("departure_delay_m (minutes)")
    st.pyplot(fig)

    st.subheader("4) Departure delay by station category")
    fig = plt.figure()
    cats = sorted(df["category"].unique())
    data = [df.loc[df["category"] == c, "departure_delay_m"] for c in cats]
    plt.boxplot(data, labels=cats)
    plt.xlabel("category (1=hub ... 7=small station)")
    plt.ylabel("departure_delay_m (minutes)")
    st.pyplot(fig)

    st.subheader("5) Correlation heatmap")
    fig = plt.figure()
    corr = df[["departure_delay_m", "arrival_delay_m", "planned_dwell_m", "category", "hour"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", square=True)
    st.pyplot(fig)


# =============================
# PAGE 03: PREDICTION
# =============================
else:
    st.subheader("Train & evaluate Linear Regression")
    model, mae, r2, y_test, pred = train_lr(df)

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("MAE (minutes)", f"{mae:.2f}")
    with c2:
        metric_card("R²", f"{r2:.3f}")
    with c3:
        metric_card("Target", TARGET_COL)

    st.subheader("Model coefficients")
    coef_df = pd.DataFrame({"feature": FEATURE_COLS, "coefficient": model.coef_})
    st.dataframe(coef_df)

    st.subheader("Actual vs Predicted")
    max_points = st.slider("Max points to plot (speed)", 2000, 50000, 20000, step=2000)

    y_arr = np.array(y_test)
    p_arr = np.array(pred)

    if len(y_arr) > max_points:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y_arr), size=max_points, replace=False)
        y_plot = y_arr[idx]
        p_plot = p_arr[idx]
    else:
        y_plot = y_arr
        p_plot = p_arr

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(p_plot, y_plot, s=12, alpha=0.35)

    mn = float(min(p_plot.min(), y_plot.min()))
    mx = float(max(p_plot.max(), y_plot.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=3)  # y=x reference

    plt.title("Actual vs Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Try a prediction (same-station departure delay)")
    st.write("Enter arrival info and station info to predict departure delay.")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        arr = st.number_input("arrival_delay_m (0–180)", 0.0, 180.0, 5.0)
    with col2:
        dwell = st.number_input("planned_dwell_m (0–60)", 0.0, 60.0, 2.0)
    with col3:
        cat = st.number_input("category (1–7)", 1, 7, 3)
    with col4:
        hr = st.number_input("hour (0–23)", 0, 23, 8)

    X_new = pd.DataFrame([{
        "arrival_delay_m": arr,
        "planned_dwell_m": dwell,
        "category": cat,
        "hour": hr
    }])[FEATURE_COLS]

    pred_one = float(model.predict(X_new)[0])
    st.markdown(f"### Predicted departure delay: **{pred_one:.1f} minutes**")

    if pred_one > 20:
        st.error("⚠ High departure delay risk")
    elif pred_one > 10:
        st.warning("⚠ Moderate departure delay risk")
    else:
        st.success("✅ Low departure delay risk")
