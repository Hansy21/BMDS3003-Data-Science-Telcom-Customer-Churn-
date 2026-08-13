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
/* Stat row: one asymmetric grid replacing the old 3 identical grey cards.
   Column 1 (2fr) is the hero churn-probability stat; columns 2/3 are
   compact and visually distinct from each other, tied together by the
   shared accent colour instead of a repeated card shell. */
.stat-row {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr;
    gap: 0;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 0.9rem;
    background: #fff;
}
.stat-cell {
    padding: 0.85rem 1.1rem;
    border-left: 1px solid #eceef1;
}
.stat-cell:first-child { border-left: none; }
.stat-eyebrow {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 0.3rem;
}
.stat-hero-value {
    font-size: 2.1rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    line-height: 1;
    color: #111827;
}
.stat-hero-value span {
    font-size: 1.1rem;
    font-weight: 600;
    color: #6b7280;
    margin-left: 0.15rem;
}
.stat-track {
    margin-top: 0.55rem;
    height: 5px;
    border-radius: 3px;
    background: #eef0f2;
    position: relative;
    overflow: hidden;
}
.stat-track-fill {
    position: absolute;
    inset: 0;
    border-radius: 3px;
    width: var(--fill, 0%);
    background: var(--accent, #6b7280);
}
.stat-track-baseline {
    position: absolute;
    top: -2px;
    bottom: -2px;
    width: 2px;
    left: var(--baseline, 26.5%);
    background: #4b5563;
}
.stat-band-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.28rem 0.65rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--accent, #6b7280);
    background: color-mix(in srgb, var(--accent, #6b7280) 12%, white);
    border: 1px solid color-mix(in srgb, var(--accent, #6b7280) 35%, white);
}
.stat-decision {
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--accent, #6b7280);
}
.stat-decision small {
    font-size: 0.75rem;
    font-weight: 600;
    color: #9ca3af;
}
/* Risk meter: replaces the circular Plotly gauge. A horizontal track
   with LOW/MEDIUM/HIGH zones sized to the model's actual threshold
   (not fixed bands), a grey tick for the threshold, and a flag marker
   for the current probability. Deliberately not another donut/gauge. */
.risk-meter {
    margin: 1.5rem 0 0.4rem;
    padding-top: 2.1rem;
}
.risk-meter-track {
    position: relative;
    height: 12px;
    border-radius: 6px;
    overflow: visible;
}
.risk-meter-threshold {
    position: absolute;
    top: -5px;
    bottom: -5px;
    width: 2px;
    background: #4b5563;
}
.risk-meter-marker {
    position: absolute;
    top: -2.05rem;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
}
.risk-meter-marker-label {
    font-size: 0.85rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    color: var(--accent, #6b7280);
    white-space: nowrap;
    margin-bottom: 1px;
}
.risk-meter-marker-flag {
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 8px solid var(--accent, #6b7280);
}
.risk-meter-zones {
    display: flex;
    margin-top: 0.4rem;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
}
.risk-meter-zones span:first-child { text-align: left; }
.risk-meter-zones span:nth-child(2) { text-align: center; }
.risk-meter-zones span:last-child { text-align: right; }
.action-box {
    border-left: 5px solid #3498db;
    background: #eaf2f8;
    color: #1b4f72;
    border-radius: 0 12px 12px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0 1rem 0;
}
.action-box b {
    color: #154360;
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
