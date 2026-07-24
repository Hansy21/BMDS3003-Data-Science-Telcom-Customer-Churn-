"""
Streamlit deployment prototype for Telco Customer Churn.

Package layout
--------------
config.py     – paths, presets, constants
loaders.py    – load models, scaler, test data (cached)
features.py   – form → model-ready dataframe, risk bands
charts.py     – Plotly chart builders
styles.py     – custom CSS
sidebar.py    – all user inputs (sidebar UI)
tabs/
  predict.py  – prediction dashboard
  insights.py – live model comparison
"""

__all__ = ["config", "loaders", "features", "charts", "styles", "sidebar"]
