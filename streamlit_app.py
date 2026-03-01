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

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Deutsche Bahn Delay Predictor", layout="wide")
st.title("🚆 Deutsche Bahn Delay Predictor")
st.caption("Intro Data Science app: predict arrival delay (minutes) using a simple linear regression model.")

# -----------------------------
# Constants
# -----------------------------
DATA_URL = "https://github.com/hz3396/DBDelayPredict/releases/download/v1.0/db_sample.csv"
DATA_FILE = "db_sample.csv"

REQUIRED_COLS = ["arrival_delay_m", "departure_delay_m", "category", "arrival_plan"]
FEATURE_COLS = ["departure_delay_m", "category", "hour"]
TARGET_COL = "arrival_delay_m"


# -----------------------------
# UI helpers
# -----------------------------
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


# -----------------------------
# Data functions
# -----------------------------
@st.cache_data
def load_csv_from_upload(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

@st.cache_data
def download_and_load_default() -> pd.DataFrame:
    """
    Download once (GitHub Release), then load. Cached by Streamlit.
    """
    if not os.path.exists(DATA_FILE):
        r = requests.get(DATA_URL, timeout=120)
        r.raise_for_status()
        with open(DATA_FILE, "wb") as f:
            f.write(r.content)
    return pd.read_csv(DATA_FILE)

def build_model_table(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal feature engineering:
    - Parse arrival_plan -> hour
    - Keep numeric columns
    - Drop missing
    - Clip to reasonable ranges for stability
    """
    df = raw.copy()

    # Check required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse time and extract hour
    df["arrival_plan"] = pd.to_datetime(df["arrival_plan"], errors="coerce")
    df["hour"] = df["arrival_plan"].dt.hour

    # Convert numeric
    for c in ["arrival_delay_m", "departure_delay_m", "category", "hour"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[["arrival_delay_m", "departure_delay_m", "category", "hour"]].dropna()

    # Reasonable ranges
    df = df[df["arrival_delay_m"].between(0, 180)]
    df = df[df["departure_delay_m"].between(0, 180)]
    df = df[df["category"].between(1, 7)]
    df = df[df["hour"].between(0, 23)]

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


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("📌 Controls")
page = st.sidebar.radio(
    "Select Page",
    ["01 Introduction", "02 Data Visualization", "03 Prediction"],
    index=0
)

uploaded = st.sidebar.file_uploader(
    "Upload DBtrainrides CSV (optional)",
    type=["csv"]
)

# Load data: uploaded OR default
try:
    if uploaded is not None:
        raw = load_csv_from_upload(uploaded)
        st.sidebar.success("Using uploaded dataset")
    else:
        raw = download_and_load_default()
        st.sidebar.success("Using default dataset (GitHub Release)")
except Exception as e:
    st.error("Failed to load dataset.")
    st.code(str(e))
    st.stop()

# Build modeling table
try:
    df = build_model_table(raw)
except Exception as e:
    st.error("Failed to process dataset (check required columns and formats).")
    st.code(str(e))
    st.stop()


# -----------------------------
# Page 01: Introduction
# -----------------------------
if page == "01 Introduction":
    st.subheader("What we are predicting")
    st.write(
        """
We build a **Linear Regression** model to predict **arrival delay (minutes)**.

**Target (y):** `arrival_delay_m`  
**Features (X):** `departure_delay_m`, `category`, `hour`
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Raw rows", f"{len(raw):,}")
    with c2:
        metric_card("Model rows (cleaned)", f"{len(df):,}")
    with c3:
        metric_card("Raw columns", f"{raw.shape[1]}")
    with c4:
        metric_card("Features used", f"{len(FEATURE_COLS)}")

    st.subheader("Required columns check")
    st.write("The app expects these columns to exist in the dataset:")
    st.code("\n".join(REQUIRED_COLS))

    st.subheader("Missing values (required columns)")
    miss = raw[REQUIRED_COLS].isna().sum()
    st.dataframe(miss.rename("missing_count"))

    st.subheader("Raw dataset preview (first 25 rows)")
    st.dataframe(raw.head(25))

    st.subheader("Modeling table preview (first 25 rows)")
    st.dataframe(df.head(25))


# -----------------------------
# Page 02: Data Visualization
# -----------------------------
elif page == "02 Data Visualization":
    st.subheader("1) Arrival delay distribution")
    fig = plt.figure()
    plt.hist(df["arrival_delay_m"], bins=50)
    plt.xlabel("arrival_delay_m (minutes)")
    plt.ylabel("count")
    st.pyplot(fig)

    st.subheader("2) Departure delay vs Arrival delay")
    fig = plt.figure()
    plt.scatter(df["departure_delay_m"], df["arrival_delay_m"], s=6, alpha=0.35)
    plt.xlabel("departure_delay_m (minutes)")
    plt.ylabel("arrival_delay_m (minutes)")
    st.pyplot(fig)

    st.subheader("3) Arrival delay by station category")
    fig = plt.figure()
    cats = sorted(df["category"].unique())
    data = [df.loc[df["category"] == c, "arrival_delay_m"] for c in cats]
    plt.boxplot(data, labels=cats)
    plt.xlabel("category (1=hub ... 7=small station)")
    plt.ylabel("arrival_delay_m (minutes)")
    st.pyplot(fig)

    st.subheader("4) Mean arrival delay by hour")
    fig = plt.figure()
    mean_by_hour = df.groupby("hour")["arrival_delay_m"].mean()
    plt.plot(mean_by_hour.index, mean_by_hour.values)
    plt.xlabel("hour (0-23)")
    plt.ylabel("mean arrival_delay_m (minutes)")
    st.pyplot(fig)

    st.subheader("5) Correlation heatmap")
    fig = plt.figure()
    corr = df[["arrival_delay_m", "departure_delay_m", "category", "hour"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", square=True)
    st.pyplot(fig)


# -----------------------------
# Page 03: Prediction
# -----------------------------
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
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=3)

    plt.title("Actual vs Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Make a prediction")
    st.write("Enter feature values to predict arrival delay (minutes).")

    col1, col2, col3 = st.columns(3)
    with col1:
        dep = st.number_input("departure_delay_m (0–180)", 0.0, 180.0, 5.0)
    with col2:
        cat = st.number_input("category (1–7)", 1, 7, 3)
    with col3:
        hr = st.number_input("hour (0–23)", 0, 23, 8)

    X_new = pd.DataFrame([{
        "departure_delay_m": dep,
        "category": cat,
        "hour": hr
    }])[FEATURE_COLS]

    pred_one = float(model.predict(X_new)[0])
    st.markdown(f"### Predicted arrival delay: **{pred_one:.1f} minutes**")

    if pred_one > 20:
        st.error("⚠ High delay risk")
    elif pred_one > 10:
        st.warning("⚠ Moderate delay risk")
    else:
        st.success("✅ Low delay risk")

    st.caption("This is an intro-level model with a simple feature set.")

    st.caption("Note: This is an intro-level linear regression model with basic features.")
