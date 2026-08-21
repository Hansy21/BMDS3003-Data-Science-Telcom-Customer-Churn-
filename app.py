"""
Telco Customer Churn — Streamlit deployment prototype
=====================================================
BMDS2003 Data Science project entry point.

Run from the project root:
    python -m streamlit run app.py

All UI logic is split under the ``prototype/`` package so this file
only wires modules together (easy to read).
"""

import streamlit as st

from prototype.config import DEFAULTS
from prototype.features import apply_preset
from prototype.loaders import (
    list_model_names,
    load_best_model_name,
    load_feature_columns,
    load_scaler,
)
from prototype.sidebar import render_sidebar
from prototype.styles import inject_styles
from prototype.tabs import render_analysis_tab, render_insights_tab, render_predict_tab

# ---------------------------------------------------------------------------
# 1. Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

# ---------------------------------------------------------------------------
# 2. Load shared artifacts (scaler, columns, model list)
# ---------------------------------------------------------------------------
scaler = load_scaler()
feature_columns = load_feature_columns()
best_model_name = load_best_model_name()
model_names = list_model_names()

if not model_names:
    st.error(
        "No trained models found in models/. Run each member's training "
        "script first (e.g. `python member1_KNN/KNN.py`), then reload this app."
    )
    st.stop()

# ---------------------------------------------------------------------------
# 3. Session state defaults (form presets + last prediction)
# ---------------------------------------------------------------------------
if "inputs_ready" not in st.session_state:
    apply_preset(DEFAULTS)
    st.session_state.inputs_ready = True
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ---------------------------------------------------------------------------
# 4. Sidebar (all inputs) → main tabs (results only)
# ---------------------------------------------------------------------------
sidebar = render_sidebar(model_names, best_model_name)

st.title("Telco Customer Churn Predictor")
st.caption(
    "BMDS2003 · CRISP-DM deployment prototype · "
    "Set customer details in the **sidebar**, then view results here."
)

predict_tab, insights_tab, analysis_tab = st.tabs(
    ["Prediction Dashboard", "Model Insights", "Data Analysis"]
)

with predict_tab:
    render_predict_tab(sidebar, feature_columns, scaler)

with insights_tab:
    render_insights_tab(
        model_names=model_names,
        best_model_name=best_model_name,
        selected_model_name=sidebar["model_name"],
        has_test=sidebar["has_test"],
        X_test=sidebar["X_test"],
        y_test=sidebar["y_test"],
    )

with analysis_tab:
    render_analysis_tab()
