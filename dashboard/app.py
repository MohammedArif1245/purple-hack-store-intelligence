import streamlit as st
import requests
import os
import pandas as pd
import plotly.express as px

# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Store Intelligence Dashboard",
    page_icon="🏪",
    layout="wide",
)

st.title("🏪 Live Store Intelligence")
st.sidebar.title("Configuration")
STORE_ID = st.sidebar.selectbox("Store", ["STORE_BLR_002", "STORE_BLR_003"])
st.markdown(f"**Store:** `{STORE_ID}` | **Status:** 🔴 Live")

# Setup placeholders for dynamic content
top_metrics = st.columns(4)
col1, col2 = st.columns([2, 1])
heatmap_placeholder = col1.empty()
funnel_placeholder = col2.empty()
anomalies_placeholder = st.empty()

def fetch_data(endpoint: str):
    try:
        response = requests.get(f"{API_BASE_URL}/stores/{STORE_ID}/{endpoint}", timeout=2)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None

@st.fragment(run_every=2)
def update_dashboard():
    metrics = fetch_data("metrics")
    funnel = fetch_data("funnel")
    heatmap = fetch_data("heatmap")
    anomalies = fetch_data("anomalies")

    if metrics:
        top_metrics[0].metric("Unique Visitors Today", metrics["total_visitors"])
        top_metrics[1].metric("Conversion Rate", f"{metrics['conversion_rate']*100:.1f}%")
        top_metrics[2].metric("Avg Dwell Time", f"{metrics['avg_dwell_ms']/1000/60:.1f} min")
        top_metrics[3].metric("Checkout Queue Depth", metrics["queue_depth"])

    if heatmap and "zones" in heatmap:
        df_heat = pd.DataFrame(heatmap["zones"])
        if not df_heat.empty:
            fig = px.bar(df_heat, x="zone_id", y="visit_freq", color="score",
                         title="Live Zone Activity (Heatmap)",
                         labels={"visit_freq": "Visits", "zone_id": "Zone", "score": "Intensity Score"},
                         color_continuous_scale="Reds")
            heatmap_placeholder.plotly_chart(fig, use_container_width=True)
        else:
            heatmap_placeholder.info("No zone data available yet.")

    if funnel and "stages" in funnel:
        df_funnel = pd.DataFrame(funnel["stages"])
        if not df_funnel.empty:
            fig_funnel = px.funnel(df_funnel, x='count', y='stage', title="Live Conversion Funnel")
            funnel_placeholder.plotly_chart(fig_funnel, use_container_width=True)
        else:
            funnel_placeholder.info("No funnel data available yet.")
            
    if anomalies and "anomalies" in anomalies and anomalies["anomalies"]:
        with anomalies_placeholder.container():
            st.subheader("🚨 Active Alerts")
            for anomaly in anomalies["anomalies"]:
                severity_color = "red" if anomaly["severity"] == "CRITICAL" else ("orange" if anomaly["severity"] == "WARN" else "blue")
                st.markdown(f"""
                <div style="padding:10px;border-left:5px solid {severity_color};background-color:#f8f9fa;margin-bottom:10px;">
                    <strong>{anomaly['anomaly_type']}</strong> ({anomaly['severity']})<br>
                    {anomaly['description']}<br>
                    <em>Action: {anomaly['suggested_action']}</em>
                </div>
                """, unsafe_allow_html=True)
    else:
        anomalies_placeholder.empty()

update_dashboard()
