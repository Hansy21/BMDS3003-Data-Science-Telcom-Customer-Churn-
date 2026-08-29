"""
Custom CSS for the Streamlit prototype (animations, banners, cards, empty state).
"""

import streamlit as st

CUSTOM_CSS = """
<style>
/* ==========================================================================
   Telco Churn Predictor — Modern Dynamic Animations & Design System
   ========================================================================== */

/* Keyframe Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translate3d(0, 30px, 0) scale(0.96);
        filter: blur(4px);
    }
    to {
        opacity: 1;
        transform: translate3d(0, 0, 0) scale(1);
        filter: blur(0);
    }
}

@keyframes fadeInScale {
    from {
        opacity: 0;
        transform: scale(0.92);
        filter: blur(4px);
    }
    to {
        opacity: 1;
        transform: scale(1);
        filter: blur(0);
    }
}

@keyframes slideInRight {
    from {
        opacity: 0;
        transform: translate3d(-30px, 0, 0);
        filter: blur(4px);
    }
    to {
        opacity: 1;
        transform: translate3d(0, 0, 0);
        filter: blur(0);
    }
}

@keyframes fadeOutDown {
    from {
        opacity: 1;
        transform: translate3d(0, 0, 0) scale(1);
        filter: blur(0);
    }
    to {
        opacity: 0;
        transform: translate3d(0, 30px, 0) scale(0.96);
        filter: blur(4px);
    }
}

@keyframes pulseGlow {
    0% {
        box-shadow: 0 0 0 0 rgba(79, 70, 229, 0.4);
    }
    70% {
        box-shadow: 0 0 0 8px rgba(79, 70, 229, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(79, 70, 229, 0);
    }
}

@keyframes fillTrack {
    from { width: 0%; }
    to { width: var(--fill, 0%); }
}

@keyframes popBadge {
    0% { transform: scale(0.7); opacity: 0; }
    70% { transform: scale(1.08); }
    100% { transform: scale(1); opacity: 1; }
}

/* Animation Utility Classes */
.anim-fade-in {
    animation: fadeInUp 0.85s cubic-bezier(0.2, 0.8, 0.2, 1) both;
}

.anim-scale-in {
    animation: fadeInScale 0.85s cubic-bezier(0.2, 0.8, 0.2, 1) both;
}

.anim-slide-in {
    animation: slideInRight 0.85s cubic-bezier(0.2, 0.8, 0.2, 1) both;
}

.anim-fade-out-down {
    animation: fadeOutDown 0.6s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
}

.anim-stagger-1 { animation-delay: 0.12s; }
.anim-stagger-2 { animation-delay: 0.24s; }
.anim-stagger-3 { animation-delay: 0.36s; }
.anim-stagger-4 { animation-delay: 0.48s; }
.anim-stagger-5 { animation-delay: 0.60s; }

/* Preset Notification Banner */
.preset-notification-banner {
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 1.15rem;
    color: #1e1b4b;
    display: flex;
    align-items: center;
    justify-content: space-between;
    animation: fadeInScale 0.85s cubic-bezier(0.2, 0.8, 0.2, 1) both;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
}

.preset-notification-banner.loyal {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 1px solid #bbf7d0;
    border-left: 5px solid #16a34a;
    color: #14532d;
}

.preset-notification-banner.at_risk {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border: 1px solid #fecaca;
    border-left: 5px solid #dc2626;
    color: #7f1d1d;
}

.preset-notification-banner.random {
    background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
    border: 1px solid #e9d5ff;
    border-left: 5px solid #9333ea;
    color: #581c87;
}

.preset-notification-banner.reset {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-left: 5px solid #64748b;
    color: #334155;
}

.preset-icon-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    margin-right: 0.6rem;
    animation: popBadge 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) both;
}

/* Result Banner */
.result-banner {
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    color: white;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: fadeInUp 0.85s cubic-bezier(0.2, 0.8, 0.2, 1) both;
}
.result-banner:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(0,0,0,0.12);
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

/* Stat row */
.stat-row {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr;
    gap: 0;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 0.9rem;
    background: #fff;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    transition: box-shadow 0.2s ease;
    animation: fadeInUp 0.85s cubic-bezier(0.2, 0.8, 0.2, 1) 0.08s both;
}
.stat-row:hover {
    box-shadow: 0 6px 16px rgba(0,0,0,0.06);
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
    height: 6px;
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
    transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    animation: fillTrack 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
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
    transition: transform 0.2s ease;
}
.stat-band-pill:hover {
    transform: scale(1.05);
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

/* Risk meter */
.risk-meter {
    margin: 1.1rem 0 0.5rem;
    padding-top: 1.9rem;
    position: relative;
    animation: fadeInUp 0.85s cubic-bezier(0.2, 0.8, 0.2, 1) 0.15s both;
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
    transition: left 0.6s cubic-bezier(0.16, 1, 0.3, 1);
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
    transition: transform 0.2s ease;
}
.risk-meter-marker:hover .risk-meter-marker-label {
    transform: scale(1.08);
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

/* Action Box */
.action-box {
    border-left: 5px solid var(--accent, #3b82f6);
    background: #f8fafc;
    color: #0f172a;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0 1.2rem 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    animation: fadeInUp 0.85s cubic-bezier(0.2, 0.8, 0.2, 1) 0.2s both;
}
.action-box:hover {
    transform: translateX(3px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
.action-box b {
    color: var(--accent, #1e3a8a);
}

/* Empty State */
.empty-state {
    border: 2px dashed #cbd5e1;
    border-radius: 16px;
    padding: 2.5rem 1.5rem;
    text-align: center;
    color: #64748b;
    background: #f8fafc;
    animation: fadeInScale 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) both;
    transition: border-color 0.2s ease, background 0.2s ease;
}
.empty-state:hover {
    border-color: #94a3b8;
    background: #f1f5f9;
}

/* Ground Truth Banner */
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
    animation: fadeInUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) both;
    transition: transform 0.2s ease;
}
.ground-truth-banner:hover {
    transform: translateY(-1px);
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
    animation: popBadge 0.7s cubic-bezier(0.2, 0.8, 0.2, 1) both;
}

/* Top Overall Decision Box */
.overall-decision-box {
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    border: 1px solid;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
    animation: fadeInScale 0.85s cubic-bezier(0.2, 0.8, 0.2, 1) both;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.overall-decision-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
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
    transition: transform 0.15s ease, background 0.15s ease;
}
.decision-pill:hover {
    transform: scale(1.04);
    background: #ffffff;
}

/* 4-Column Side-by-Side Cards */
.model-4col-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 4px solid var(--accent, #6b7280);
    border-radius: 12px;
    padding: 1.1rem 1rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    display: flex;
    flex-direction: column;
    height: 100%;
    animation: fadeInUp 0.85s cubic-bezier(0.2, 0.8, 0.2, 1) both;
}
.model-4col-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 22px rgba(0, 0, 0, 0.09);
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
    animation: popBadge 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) both;
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
    transition: transform 0.2s ease;
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
    transition: transform 0.2s ease, background 0.2s ease;
}
.compact-action-box:hover {
    transform: translateX(2px);
    background: #f1f5f9;
}

/* EDA Image Presentation Card */
.eda-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.1rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    animation: fadeInUp 0.85s cubic-bezier(0.2, 0.8, 0.2, 1) both;
}
.eda-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
}
.eda-card h4 {
    margin-top: 0;
    color: #1e293b;
    font-size: 1.05rem;
    font-weight: 700;
}

/* Model Insights Container */
.insights-container {
    animation: fadeInUp 0.85s cubic-bezier(0.2, 0.8, 0.2, 1) both;
}

/* Button & Interactive micro-animations */
button[kind="primary"], .stButton > button {
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
button[kind="primary"]:hover, .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12) !important;
}
button[kind="primary"]:active, .stButton > button:active {
    transform: translateY(1px) !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}
</style>
"""


def inject_styles() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
