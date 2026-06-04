import streamlit as st

def render_metric_card(title: str, value: str, delta: str = None):
    """Render a styled metric card."""
    st.metric(label=title, value=value, delta=delta)

def render_queue_bar(depth: int, max_depth: int = 10):
    """Render a visual bar representing queue depth."""
    st.write(f"Queue Depth: {depth}")
    progress = min(depth / max_depth, 1.0)
    st.progress(progress)
