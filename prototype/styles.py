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
/* Risk meter: A horizontal spectrum track with LOW/MEDIUM/HIGH zones
   sized to the model's actual threshold, a prominent threshold tick,
   and a high-contrast probability marker. */
.risk-meter {
    margin: 1.1rem 0 0.5rem;
    padding-top: 1.9rem;
    position: relative;
}
.risk-meter-track {
    position: relative;
    height: 14px;
    border-radius: 7px;
    overflow: visible;
    border: 1px solid rgba(0, 0, 0, 0.08);
}
.risk-meter-threshold {
    position: absolute;
    top: -6px;
    bottom: -6px;
    width: 3px;
    background: #1e293b;
    border-radius: 2px;
    z-index: 2;
    box-shadow: 0 0 4px rgba(0,0,0,0.3);
}
.risk-meter-marker {
    position: absolute;
    top: -1.95rem;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    z-index: 3;
}
.risk-meter-marker-label {
    font-size: 0.88rem;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
    color: var(--accent, #1e293b);
    background: #ffffff;
    padding: 0.08rem 0.4rem;
    border-radius: 4px;
    border: 1.5px solid var(--accent, #6b7280);
    box-shadow: 0 2px 5px rgba(0,0,0,0.12);
    white-space: nowrap;
    line-height: 1.2;
    margin-bottom: 2px;
}
.risk-meter-marker-flag {
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 7px solid var(--accent, #6b7280);
}
.risk-meter-zones {
    display: flex;
    margin-top: 0.45rem;
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.04em;
}
.risk-meter-zones span:first-child { text-align: left; }
.risk-meter-zones span:nth-child(2) { text-align: center; }
.risk-meter-zones span:last-child { text-align: right; }
.action-box {
    border-left: 5px solid var(--accent, #3b82f6);
    background: #f8fafc;
    color: #0f172a;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0 1.2rem 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.action-box b {
    color: var(--accent, #1e3a8a);
}
.empty-state {
    border: 2px dashed #cbd5e1;
    border-radius: 16px;
    padding: 2.5rem 1.5rem;
    text-align: center;
    color: #64748b;
    background: #f8fafc;
}
.ground-truth-banner {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 5px solid #6366f1;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.ground-truth-title {
    font-weight: 700;
    font-size: 0.95rem;
    color: #1e293b;
}
.ground-truth-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
}
.model-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin-bottom: 1.25rem;
}
.model-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-top: 4px solid var(--accent, #6b7280);
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.model-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(0, 0, 0, 0.08);
}
.model-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.model-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1f2937;
    margin: 0;
}
.best-badge {
    background: #eef2ff;
    color: #4f46e5;
    border: 1px solid #c7d2fe;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 0.12rem 0.45rem;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.model-card-prob {
    font-size: 1.8rem;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
    color: var(--accent, #111827);
    line-height: 1.1;
    margin-bottom: 0.4rem;
}
.model-card-prob span {
    font-size: 1rem;
    font-weight: 600;
    color: #6b7280;
    margin-left: 0.1rem;
}
.model-card-track {
    height: 6px;
    background: #f1f5f9;
    border-radius: 3px;
    position: relative;
    overflow: hidden;
    margin-bottom: 0.65rem;
}
.model-card-track-fill {
    height: 100%;
    background: var(--accent, #6b7280);
    border-radius: 3px;
    width: var(--fill, 0%);
}
.model-card-footer {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 0.8rem;
}
.model-card-decision {
    font-weight: 700;
    color: var(--accent, #6b7280);
}
.model-card-thresh {
    font-size: 0.72rem;
    color: #9ca3af;
}
.overall-decision-box {
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    border: 1px solid;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
}
.overall-decision-box.unanimous-churn {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border-color: #fca5a5;
    border-left: 6px solid #dc2626;
    color: #991b1b;
}
.overall-decision-box.unanimous-stay {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border-color: #86efac;
    border-left: 6px solid #16a34a;
    color: #166534;
}
.overall-decision-box.majority-churn {
    background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
    border-color: #fdba74;
    border-left: 6px solid #ea580c;
    color: #9a3412;
}
.overall-decision-box.majority-stay {
    background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%);
    border-color: #5eead4;
    border-left: 6px solid #0d9488;
    color: #115e59;
}
.overall-decision-box.split {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border-color: #fcd34d;
    border-left: 6px solid #d97706;
    color: #92400e;
}
.decision-box-title {
    font-size: 1.3rem;
    font-weight: 800;
    margin: 0 0 0.35rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.decision-box-sub {
    font-size: 0.95rem;
    margin: 0 0 0.85rem 0;
    opacity: 0.95;
    line-height: 1.4;
}
.decision-pills-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
}
.decision-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    background: rgba(255, 255, 255, 0.75);
    border: 1px solid rgba(0, 0, 0, 0.08);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.model-4col-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 4px solid var(--accent, #6b7280);
    border-radius: 12px;
    padding: 1.1rem 1rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    display: flex;
    flex-direction: column;
    height: 100%;
}
.model-4col-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 18px rgba(0, 0, 0, 0.08);
}
.model-4col-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
}
.model-4col-title {
    font-size: 1rem;
    font-weight: 700;
    color: #1e293b;
    margin: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.model-decision-badge {
    display: block;
    text-align: center;
    padding: 0.5rem 0.6rem;
    border-radius: 7px;
    font-weight: 800;
    font-size: 0.95rem;
    letter-spacing: 0.04em;
    color: white;
    margin-bottom: 0.75rem;
}
.model-decision-badge.churn {
    background: linear-gradient(135deg, #dc2626, #b91c1c);
    box-shadow: 0 2px 6px rgba(220, 38, 38, 0.25);
}
.model-decision-badge.stay {
    background: linear-gradient(135deg, #16a34a, #15803d);
    box-shadow: 0 2px 6px rgba(22, 163, 74, 0.25);
}
.model-4col-prob {
    font-size: 1.95rem;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
    color: var(--accent, #111827);
    line-height: 1;
    margin-bottom: 0.35rem;
}
.model-4col-prob span {
    font-size: 1.05rem;
    font-weight: 600;
    color: #6b7280;
}
.model-4col-sub {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.8rem;
    color: #64748b;
    margin-bottom: 0.65rem;
}
.compact-action-box {
    background: #f8fafc;
    border-left: 4px solid var(--accent, #3b82f6);
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 0.85rem;
    font-size: 0.88rem;
    color: #1e293b;
    margin-top: 0.75rem;
    line-height: 1.45;
    border-top: 1px solid #f1f5f9;
    border-right: 1px solid #f1f5f9;
    border-bottom: 1px solid #f1f5f9;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}
</style>
"""


def inject_styles() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
