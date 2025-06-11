import streamlit as st
import pandas as pd
import altair as alt
import joblib

@st.cache_data
def load_artifacts():
    model  = joblib.load("models/churn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    df     = pd.read_csv("data/predictions/predictions.csv", parse_dates=["join_date"])
    return model, scaler, df

model, scaler, df = load_artifacts()

st.title("🕵️‍♂️ ChurnLens Dashboard")
# ── Sidebar Filters ────────────────────────────────────────────
st.sidebar.header("Filters")
cohorts = sorted(df["cohort"].unique())
sel_cohorts = st.sidebar.multiselect("Select Cohorts", cohorts, default=cohorts)
min_prob    = st.sidebar.slider("Min Churn Probability", 0.0, float(df.pred_churn_proba.max()), 0.0, 0.01)

# Apply filters
df = df[df["cohort"].isin(sel_cohorts) & (df["pred_churn_proba"] >= min_prob)]

# ── KPI Summary Cards ──────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Avg Churn Prob",   f"{df.pred_churn_proba.mean():.2%}")
col2.metric("Anomalies Flagged", int(df.anomaly_flag.sum()))
col3.metric("Total Customers",   len(df))


# 1) Churn Probability Histogram
st.subheader("Predicted Churn Probability")
hist = alt.Chart(df).mark_bar().encode(
    alt.X("pred_churn_proba:Q", bin=alt.Bin(maxbins=30), title="Churn Probability"),
    y="count():Q",
    tooltip=["count()"]
).properties(height=300)
st.altair_chart(hist, use_container_width=True)

# 2) Anomaly Scatter by Cohort
st.subheader("Anomaly Flags by Cohort")
scatter = alt.Chart(df).mark_circle(size=60).encode(
    x="pred_churn_proba:Q",
    y=alt.Y("cohort:N", title="Join Cohort"),
    color=alt.Color("anomaly_flag:N", title="Anomaly"),
    tooltip=["customer_id","pred_churn_proba","anomaly_flag"]
).properties(height=400)
st.altair_chart(scatter, use_container_width=True)

# 3) Cohort Heatmap
st.subheader("Average Churn by Cohort")
cohort_avg = df.groupby("cohort")["pred_churn_proba"].mean().reset_index()
heat = alt.Chart(cohort_avg).mark_rect().encode(
    x=alt.X("cohort:N", sort=cohort_avg["cohort"].tolist(), title="Cohort"),
    y=alt.value(1),
    color=alt.Color("pred_churn_proba:Q", scale=alt.Scale(scheme="viridis"), title="Avg Churn"),
    tooltip=["cohort","pred_churn_proba"]
).properties(height=50)
st.altair_chart(heat, use_container_width=True)
