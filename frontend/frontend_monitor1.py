import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta

# Setup path
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

# ---- Streamlit App UI ----
st.title("📊 Mean Absolute Error (MAE) by Pickup Hour")

# Sidebar input
st.sidebar.header("Settings")
past_hours = st.sidebar.slider(
    "Number of Past Hours to Plot",
    min_value=12,
    max_value=24 * 7,
    value=24,
    step=1,
)

# ---- Mock Data ----
# Create hourly timestamps for the past `past_hours` hours
now = datetime.now()
hours = [now - timedelta(hours=i) for i in reversed(range(past_hours))]

# Simulate some MAE values with slight noise
np.random.seed(42)
mae_values = np.abs(np.random.normal(loc=5, scale=1.5, size=past_hours)).round(2)

# Create DataFrame
mae_by_hour = pd.DataFrame({
    "pickup_hour": hours,
    "MAE": mae_values
})

# ---- Plot ----
fig = px.line(
    mae_by_hour,
    x="pickup_hour",
    y="MAE",
    title=f"📈 Simulated MAE for the Past {past_hours} Hours",
    labels={"pickup_hour": "Pickup Hour", "MAE": "Mean Absolute Error"},
    markers=True,
)

# Display
st.plotly_chart(fig)
st.write(f"📌 Average MAE: `{mae_by_hour['MAE'].mean():.2f}`")
