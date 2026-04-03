import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")

st.title("Time-Series Anomaly Detection Dashboard")

# Load Data
# -----------------------------
scores_df = pd.read_csv("results/anomaly_scores.csv")
anomalies_df = pd.read_csv("results/anomalies_percentile.csv")

test_data = np.load("data/processed/test.npy")

# Sidebar Controls
# -----------------------------
st.sidebar.header("Controls")

threshold = st.sidebar.slider(
    "Threshold",
    float(scores_df["smoothed_error"].min()),
    float(scores_df["smoothed_error"].max()),
    float(scores_df["smoothed_error"].quantile(0.95))
)

channel = st.sidebar.slider("Select Channel", 0, test_data.shape[2]-1, 0)

# Plot: Anomaly Score Timeline
# -----------------------------
st.subheader("Anomaly Score Timeline")

st.line_chart(scores_df["smoothed_error"])

st.write(f"Current Threshold: {threshold}")

# Highlight anomalies
# -----------------------------
detected = scores_df[scores_df["smoothed_error"] > threshold]

st.write(f"Detected anomalies: {len(detected)}")

# Signal Visualization
# -----------------------------
st.subheader("Signal Explorer")

signal = test_data[:, :, channel].mean(axis=1)

signal_df = pd.DataFrame({
    "signal": signal
})

st.line_chart(signal_df)

# Reconstruction Insight (approx)
# -----------------------------
st.subheader("High Error Points")

top_anomalies = scores_df.sort_values(by="smoothed_error", ascending=False).head(10)

st.write(top_anomalies)

# Generate Report
# -----------------------------
if st.button("Generate Full Report"):
    report = {
        "signalData": signal.tolist(),
        "anomalyScores": scores_df["smoothed_error"].tolist(),
        "topAnomalies": top_anomalies.to_dict()
    }

    import json
    with open("results/streamlit_report.json", "w") as f:
        json.dump(report, f)

    st.success("Report generated!")