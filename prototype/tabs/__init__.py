"""Main-page tabs for the Streamlit prototype."""

from prototype.tabs.analysis import render_analysis_tab
from prototype.tabs.insights import render_insights_tab
from prototype.tabs.predict import render_predict_tab

__all__ = ["render_predict_tab", "render_insights_tab", "render_analysis_tab"]
