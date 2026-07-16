"""
Custom CSS for the Streamlit prototype (banner, cards, empty state).
"""

import streamlit as st

CUSTOM_CSS = """
<style>
.result-banner {
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    color: white;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
}
.result-banner h2 {
    margin: 0 0 0.35rem 0;
    font-size: 1.7rem;
    letter-spacing: 0.02em;
}
.result-banner p {
    margin: 0;
    opacity: 0.95;
    font-size: 1.05rem;
}
.metric-card {
    background: #f8f9fa;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
    height: 100%;
}
.metric-card .label {
    color: #6b7280;
    font-size: 0.85rem;
    margin-bottom: 0.25rem;
}
.metric-card .value {
    color: #111827;
    font-size: 1.35rem;
    font-weight: 700;
}
.action-box {
    border-left: 5px solid #3498db;
    background: #eaf2f8;
    border-radius: 0 12px 12px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0 1rem 0;
}
.empty-state {
    border: 2px dashed #cbd5e1;
    border-radius: 16px;
    padding: 2.5rem 1.5rem;
    text-align: center;
    color: #64748b;
    background: #f8fafc;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}
</style>
"""


def inject_styles() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
