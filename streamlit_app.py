import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

DATA_URL = "https://github.com/hz3396/DBDelayPredict/releases/download/v1.0/db_sample.csv"
DATA_FILE = "db_sample.csv"

@st.cache_data
def load_default_data() -> pd.DataFrame:
    """
    Download the default dataset once (from GitHub Release), then load it.
    """
    if not os.path.exists(DATA_FILE):
        with st.spinner("Downloading default dataset... (first time only)"):
            r = requests.get(DATA_URL, timeout=60)
            r.raise_for_status()
            with open(DATA_FILE, "wb") as f:
                f.write(r.content)

    return pd.read_csv(DATA_FILE)

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Deutsche Bahn Delay Predictor", layout="wide")
st.title("🚆 Deutsche Bahn Delay Predictor")
st.caption("Intro Data Science app: predict arrival delay (minutes) using a simple linear regression model.")

# -----------------------------
# Small UI helper
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
# Data loading & processing
# -----------------------------
@st.cache_data

def build_model_table(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Target: arrival_delay_m
    Features: departure_delay_m, category, hour (from arrival_plan)

    Minimal, intro-friendly cleaning.
    """
    df = raw.copy()

    required_cols = ["arrival_delay_m", "departure_delay_m", "category", "arrival_plan"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse time and extract hour
    df["arrival_plan"] = pd.to_datetime(df["arrival_plan"], errors="coerce")
    df["hour"] = df["arrival_plan"].dt.hour

    # Convert to numeric
    for c in ["arrival_delay_m", "departure_delay_m", "category", "hour"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep modeling columns and drop missing
    df = df[["arrival_delay_m", "departure_delay_m", "category", "hour"]].dropna()

    # Keep reasonable ranges (helps visuals + stability)
    df = df[df["arrival_delay_m"].between(0, 180)]
    df = df[df["departure_delay_m"].between(0, 180)]
    df = df[df["category"].between(1, 7)]
    df = df[df["hour"].between(0, 23)]

    return df

def train_lr(df: pd.DataFrame, feature_cols, target_col="arrival_delay_m"):
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    # Return y_test and pred for plotting "Actual vs Predicted"
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

if uploaded is not None:
    raw = load_csv(uploaded)
    st.sidebar.success("Using uploaded dataset")
else:
    raw = load_default_data()
    st.sidebar.success("Using default dataset (GitHub Release)")
    )
    st.stop()

raw = load_csv(uploaded)

try:
    df = build_model_table(raw)
except Exception as e:
    st.error("Could not process the dataset.")
    st.write("Error details:")
    st.code(str(e))
    st.stop()

feature_cols = ["departure_delay_m", "category", "hour"]


# -----------------------------
# Page 1: Introduction
# -----------------------------
if page == "01 Introduction":
    st.subheader("What this app does")
    st.write(
        """
We build a simple **Linear Regression** model to predict **arrival delay (minutes)**.

**Target (y):** `arrival_delay_m`  
**Features (X):** `departure_delay_m`, `category`, `hour`

Model form:
\[
arrival\_delay\_m = \\beta_0 + \\beta_1 \\cdot departure\_delay\_m + \\beta_2 \\cdot category + \\beta_3 \\cdot hour
\]
        """
    )

    st.subheader("Quick dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Raw rows", f"{len(raw):,}")
    with c2:
        metric_card("Model rows (cleaned)", f"{len(df):,}")
    with c3:
        metric_card("Raw columns", f"{raw.shape[1]}")
    with c4:
        metric_card("Features used", f"{len(feature_cols)}")

    st.subheader("Missing values (required columns)")
    required_cols = ["arrival_delay_m", "departure_delay_m", "category", "arrival_plan"]
    miss = raw[required_cols].isna().sum()
    st.dataframe(miss.rename("missing_count"))

    st.subheader("Raw dataset preview")
    st.dataframe(raw.head(25))

    st.subheader("Modeling table preview (after cleaning + hour feature)")
    st.dataframe(df.head(25))


# -----------------------------
# Page 2: Visualization
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
    plt.scatter(df["departure_delay_m"], df["arrival_delay_m"], s=6, alpha=0.4)
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

    st.subheader("5) Correlation Heatmap")
    st.write("Shows correlations among variables used in the model (plus the target).")
    fig = plt.figure()
    corr = df[["arrival_delay_m", "departure_delay_m", "category", "hour"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", square=True)
    st.pyplot(fig)


# -----------------------------
# Page 3: Prediction
# -----------------------------
else:
    st.subheader("Train & evaluate Linear Regression")
    model, mae, r2, y_test, pred = train_lr(df, feature_cols, target_col="arrival_delay_m")

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("MAE (minutes)", f"{mae:.2f}")
    with c2:
        metric_card("R²", f"{r2:.3f}")
    with c3:
        metric_card("Features", ", ".join(feature_cols))

    st.subheader("Model coefficients")
    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": model.coef_})
    st.dataframe(coef_df)

    # ---- Actual vs Predicted plot (like your example) ----
    st.subheader("Actual vs Predicted")

    max_points = st.slider("Max points to plot (for speed)", 2000, 50000, 20000, step=2000)
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
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=3)  # y = x dashed line

    plt.title("Actual vs Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # ---- Interactive prediction ----
    st.subheader("Make a prediction")
    st.write("Enter feature values to predict arrival delay (minutes).")

    col1, col2, col3 = st.columns(3)
    with col1:
        dep = st.number_input("departure_delay_m (0–180)", min_value=0.0, max_value=180.0, value=5.0)
    with col2:
        cat = st.number_input("category (1–7)", min_value=1, max_value=7, value=3)
    with col3:
        hr = st.number_input("hour (0–23)", min_value=0, max_value=23, value=8)

    X_new = pd.DataFrame([{
        "departure_delay_m": dep,
        "category": cat,
        "hour": hr
    }])[feature_cols]

    pred_one = float(model.predict(X_new)[0])
    st.markdown(f"### Predicted arrival delay: **{pred_one:.1f} minutes**")

    if pred_one > 20:
        st.error("⚠ High delay risk")
    elif pred_one > 10:
        st.warning("⚠ Moderate delay risk")
    else:
        st.success("✅ Low delay risk")

    st.caption("Note: This is an intro-level linear regression model with basic features.")
