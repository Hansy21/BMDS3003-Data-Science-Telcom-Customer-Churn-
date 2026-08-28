"""
Prediction Dashboard tab — supports both Single Model focus view and
Compare All Models 4-model probability comparison view.
"""

import textwrap
import pandas as pd
import streamlit as st

from prototype.charts import make_multimodel_prob_comparison_chart
from prototype.features import (
    build_input_dataframe,
    compute_risk_factors,
    get_top_drivers,
    profile_summary,
    risk_band,
)
from prototype.loaders import load_model


def _get_risk_meter_html(probability: float, threshold: float, accent: str) -> str:
    """Generate the horizontal risk meter track HTML with zones, threshold tick, and marker."""
    thr_pct = threshold * 100
    high_start = max(70.0, thr_pct)
    pct = probability * 100
    clamped_marker = max(8.0, min(92.0, pct))
    return textwrap.dedent(f"""
<div class="risk-meter" style="margin: 0.9rem 0 0.4rem; padding-top: 1.85rem;">
    <div class="risk-meter-track"
         style="background: linear-gradient(to right,
             #bbf7d0 0%, #bbf7d0 {thr_pct}%,
             #fef08a {thr_pct}%, #fef08a {high_start}%,
             #fecaca {high_start}%, #fecaca 100%);">
        <div class="risk-meter-threshold" style="left:{thr_pct}%"
             title="Decision threshold: {threshold:.2f}"></div>
        <div class="risk-meter-marker" style="left:{clamped_marker}%; --accent:{accent}">
            <div class="risk-meter-marker-label">{pct:.1f}%</div>
            <div class="risk-meter-marker-flag"></div>
        </div>
    </div>
    <div class="risk-meter-zones">
        <span style="flex:{max(thr_pct, 0.001)} 0 0; color:#15803d;">LOW</span>
        <span style="flex:{max(high_start - thr_pct, 0.001)} 0 0; color:#a16207;">MED</span>
        <span style="flex:{max(100 - high_start, 0.001)} 0 0; color:#b91c1c;">HIGH</span>
    </div>
</div>
""").strip()


def _run_prediction(sidebar: dict, feature_columns, scaler) -> None:
    """Score the form according to selected mode (Single Model or Compare All Models)."""
    input_df = build_input_dataframe(sidebar["form"], feature_columns, scaler)
    mode = sidebar.get("mode", "Single Model")
    best_model_name = sidebar.get("best_model_name")
    rand_info = sidebar.get("random_customer_info")

    if mode == "Single Model":
        model_name = sidebar["model_name"]
        model = sidebar["model"] if sidebar.get("model") is not None else load_model(model_name)
        threshold = sidebar["threshold"] if sidebar.get("threshold") is not None else 0.5
        prob = float(model.predict_proba(input_df)[0][1])
        pred = int(prob >= threshold)
        band, accent, action = risk_band(prob, threshold)

        st.session_state.last_prediction = {
            "mode": "Single Model",
            "model_name": model_name,
            "threshold": threshold,
            "probability": prob,
            "prediction": pred,
            "band": band,
            "accent": accent,
            "action": action,
            "profile": profile_summary(sidebar["form"]),
            "form": sidebar["form"],
            "input_df": input_df,
            "random_customer_info": rand_info,
        }
    else:
        # Compare All Models mode
        all_models = sidebar.get("all_models", {})
        all_thresholds = sidebar.get("all_thresholds", {})
        model_results = {}
        for name, model_obj in all_models.items():
            prob = float(model_obj.predict_proba(input_df)[0][1])
            thresh = all_thresholds.get(name, 0.5)
            pred = int(prob >= thresh)
            band, accent, action = risk_band(prob, thresh)
            model_results[name] = {
                "model_name": name,
                "probability": prob,
                "prediction": pred,
                "threshold": thresh,
                "band": band,
                "accent": accent,
                "action": action,
                "is_best": name == best_model_name,
            }

        st.session_state.last_prediction = {
            "mode": "Compare All Models",
            "model_results": model_results,
            "best_model_name": best_model_name,
            "profile": profile_summary(sidebar["form"]),
            "form": sidebar["form"],
            "input_df": input_df,
            "random_customer_info": rand_info,
        }


def _render_empty_state() -> None:
    st.markdown(
        textwrap.dedent("""
        <div class="empty-state">
            <h3 style="margin-top:0;">No prediction yet</h3>
            <p>Use the <b>sidebar</b> to set customer details or click <b>🎲 Random Sample</b>,<br>
            then click <b>Predict churn</b>.</p>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )


def _render_driver_list(drivers: list) -> None:
    """Plain-English list of what pushed THIS customer's score up/down."""
    for d in drivers:
        if d["direction"] == "up":
            icon, note = "↑", "pushes risk UP"
        elif d["direction"] == "down":
            icon, note = "↓", "pushes risk DOWN"
        else:
            icon, note = "·", "a factor the model weighs heavily"
        st.markdown(f"{icon} **{d['label']}** — {note}")


def _render_pattern_list(factors: list) -> None:
    """Plain-English list matching this customer's answers against known churn-rate patterns."""
    for f in factors:
        pct_str = f"{f['rate']:.1f}% churn rate"
        st.markdown(
            f"{f['icon']} **{f['label']}: {f['value']}** — {pct_str} (vs. 26.5% overall)"
        )


def _render_ground_truth_banner(rand_info: dict, match_note: str | None = None) -> None:
    cust_id = rand_info.get("customer_id", "Unknown")
    actual_churn = rand_info.get("actual_churn", "Unknown")
    actual_bool = 1 if str(actual_churn).lower() in ["yes", "1", "true"] else 0

    badge_bg = "#fee2e2" if actual_bool == 1 else "#dcfce7"
    badge_color = "#991b1b" if actual_bool == 1 else "#166534"
    actual_label = "CHURNED (Yes)" if actual_bool == 1 else "RETAINED (No)"

    note_html = (
        f'<span style="font-size:0.85rem; color:#4f46e5; margin-left:0.75rem; font-weight:600;">{match_note}</span>'
        if match_note
        else ""
    )

    st.markdown(
        textwrap.dedent(f"""
        <div class="ground-truth-banner">
            <div>
                <span class="ground-truth-title">🎲 Random Dataset Sample:</span>
                <span style="font-family: monospace; font-size:0.9rem; color:#475569; margin-left:0.35rem;">Customer ID: {cust_id}</span>
            </div>
            <div>
                <span style="font-size:0.85rem; color:#64748b; margin-right:0.5rem;">Historical Ground Truth:</span>
                <span class="ground-truth-badge" style="background:{badge_bg}; color:{badge_color};">
                    {actual_label}
                </span>
                {note_html}
            </div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )


def _render_single_model_result(result: dict, feature_columns) -> None:
    is_churn = result["prediction"] == 1
    banner_color = "#c0392b" if is_churn else "#1e8449"
    banner_title = "LIKELY TO CHURN" if is_churn else "LIKELY TO STAY"
    pct = result["probability"] * 100
    decision_word = "CHURN" if is_churn else "STAY"

    st.markdown(
        textwrap.dedent(f"""
        <div class="result-banner"
             style="background: linear-gradient(135deg, {banner_color}, {banner_color}cc);">
            <h2>{banner_title}</h2>
            <p style="margin-top:0.4rem; font-size:0.95rem;">
                Model: <b>{result['model_name']}</b> ·
                Risk band: <b>{result['band']}</b>
            </p>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

    # Stat row
    st.markdown(
        textwrap.dedent(f"""
        <div class="stat-row">
            <div class="stat-cell" style="--accent:{result['accent']}">
                <div class="stat-eyebrow">Churn probability</div>
                <div class="stat-hero-value">{pct:.1f}<span>%</span></div>
                <div class="stat-track" style="--accent:{result['accent']}; --fill:{pct:.1f}%">
                    <div class="stat-track-fill"></div>
                    <div class="stat-track-baseline" title="26.5% dataset baseline"></div>
                </div>
            </div>
            <div class="stat-cell">
                <div class="stat-eyebrow">Risk level</div>
                <span class="stat-band-pill" style="--accent:{result['accent']}">{result['band']}</span>
            </div>
            <div class="stat-cell">
                <div class="stat-eyebrow">Decision</div>
                <div class="stat-decision" style="--accent:{result['accent']}">
                    {decision_word}
                </div>
                <small style="color:#9ca3af; font-size:0.72rem;">
                    @ {result['threshold']:.2f} threshold
                </small>
            </div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

    # Risk meter diagram
    st.markdown(
        _get_risk_meter_html(result["probability"], result["threshold"], result["accent"]),
        unsafe_allow_html=True,
    )

    # Action box
    st.markdown(
        textwrap.dedent(f"""
        <div class="action-box" style="--accent:{result['accent']};">
            <div style="font-weight:800; font-size:1.05rem; color:{result['accent']}; margin-bottom:0.35rem;">
                🎯 Recommended Action ({result['band']} Risk):
            </div>
            <div style="font-size:0.95rem; color:#0f172a; line-height:1.5;">
                {result['action']}
            </div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

    # Customer snapshot + two explanation panels
    left, right = st.columns([1, 1.2])
    with left:
        st.subheader("Customer snapshot")
        snap = pd.DataFrame(
            {
                "Field": list(result["profile"].keys()),
                "Value": list(result["profile"].values()),
            }
        )
        st.dataframe(snap, use_container_width=True, hide_index=True)

        st.subheader("Matches known risk patterns")
        factors = compute_risk_factors(result["form"])
        if factors:
            _render_pattern_list(factors)
        else:
            st.info("No strong pattern matches for this profile.")

    with right:
        st.subheader(f"Why {result['model_name']} made this call")
        model = load_model(result["model_name"])
        drivers = get_top_drivers(
            model, result["model_name"], result["form"], result["input_df"], feature_columns
        )
        if drivers:
            _render_driver_list(drivers)
        else:
            st.info(
                f"**{result['model_name']}** is an instance-based model (e.g. KNN) "
                "which calculates predictions based on distance to nearest neighbours "
                "rather than global coefficients or feature importances."
            )


def _render_overall_decision_box(model_results: dict) -> None:
    """Render a prominent verdict comparison box at the top of the 4-model results."""
    total_models = len(model_results)
    churn_count = sum(1 for m in model_results.values() if m["prediction"] == 1)
    stay_count = total_models - churn_count
    avg_prob = sum(m["probability"] for m in model_results.values()) / max(1, total_models) * 100

    if churn_count == total_models:
        box_class = "unanimous-churn"
        icon = "🚨"
        title = f"UNANIMOUS CHURN ({churn_count}/{total_models} Models Agree)"
        sub = "All 4 models (DecisionTree, KNN, LogisticRegression, RandomForest) predict this customer is at high risk of churning."
        badge_text = "High Risk · Immediate Retention Action Required"
    elif stay_count == total_models:
        box_class = "unanimous-stay"
        icon = "✅"
        title = f"UNANIMOUS RETENTION ({stay_count}/{total_models} Models Agree)"
        sub = "All 4 models (DecisionTree, KNN, LogisticRegression, RandomForest) predict this customer will stay and remain with the service."
        badge_text = "Low Risk · Stable & Loyal Profile"
    elif churn_count > stay_count:
        box_class = "majority-churn"
        icon = "⚠️"
        title = f"MAJORITY CHURN ({churn_count} Churn vs. {stay_count} Stay)"
        sub = f"Strong churn inclination: {churn_count} models predict Churn, while {stay_count} model predicts Stay."
        badge_text = "Elevated Risk · Proactive Engagement Recommended"
    elif stay_count > churn_count:
        box_class = "majority-stay"
        icon = "🟢"
        title = f"MAJORITY RETENTION ({stay_count} Stay vs. {churn_count} Churn)"
        sub = f"Moderate-low risk: {stay_count} models predict Stay, while {churn_count} model predicts Churn."
        badge_text = "Moderate Risk · Standard Retention Monitoring"
    else:
        # Split (e.g. 2 churn, 2 stay)
        box_class = "split"
        icon = "⚖️"
        title = f"SPLIT DECISION ({churn_count} Churn vs. {stay_count} Stay)"
        sub = "Evenly divided prediction: 2 models predict Churn and 2 models predict Stay."
        badge_text = "Borderline Customer · Individual Model Review Recommended"

    html = f"""<div class="overall-decision-box {box_class}">
<div class="decision-box-title">{icon} {title}</div>
<div class="decision-box-sub">{sub}</div>
<div class="decision-pills-row">
<span class="decision-pill"><b>Average Churn Probability:</b> {avg_prob:.1f}%</span>
<span class="decision-pill"><b>Verdict:</b> {badge_text}</span>
<span class="decision-pill"><b>Baseline Churn Rate:</b> 26.5%</span>
</div>
</div>"""

    st.markdown(html, unsafe_allow_html=True)


def _render_model_column_card(name: str, m: dict) -> None:
    """Render a compact vertical card for 4-column side-by-side layout."""
    pct = m["probability"] * 100
    is_churn = m["prediction"] == 1
    badge_class = "churn" if is_churn else "stay"
    badge_label = "CHURN" if is_churn else "STAY"
    best_badge_html = '<span class="best-badge">Best</span>' if m.get("is_best") else ""

    meter_html = _get_risk_meter_html(m["probability"], m["threshold"], m["accent"])

    card_html = f"""<div class="model-4col-card" style="--accent:{m['accent']};">
<div class="model-4col-header">
<div class="model-4col-title" title="{name}">{name}</div>
{best_badge_html}
</div>
<div class="model-decision-badge {badge_class}">
{badge_label}
</div>
<div class="model-4col-prob">
{pct:.1f}<span>%</span>
</div>
<div class="stat-track" style="--accent:{m['accent']}; --fill:{pct:.1f}%; margin-bottom: 0.5rem;">
<div class="stat-track-fill"></div>
<div class="stat-track-baseline" title="26.5% baseline"></div>
</div>
<div class="model-4col-sub">
<span class="stat-band-pill" style="--accent:{m['accent']}; font-size: 0.78rem; padding: 0.18rem 0.55rem;">{m['band']}</span>
<span style="font-weight:600; font-size:0.8rem; color:#475569;">Thr: {m['threshold']:.2f}</span>
</div>
{meter_html}
<div class="compact-action-box" style="--accent:{m['accent']};">
<div style="font-weight:800; color:{m['accent']}; font-size:0.85rem; margin-bottom:0.25rem;">🎯 Action Advice:</div>
<div style="font-size:0.88rem; color:#1e293b; line-height:1.45;">{m['action']}</div>
</div>
</div>"""

    st.markdown(card_html, unsafe_allow_html=True)


def _render_compare_all_result(result: dict) -> None:
    model_results = result["model_results"]

    # 1. Top Decision / Comparison Box (Unanimous / Majority / Split)
    _render_overall_decision_box(model_results)

    st.markdown("### 📊 4-Model Prediction Comparison")
    st.caption("Side-by-side predictions across all 4 trained models:")

    # 2. 4 Side-by-Side Columns for All 4 Models
    col1, col2, col3, col4 = st.columns(4)
    model_names = list(model_results.keys())
    cols = [col1, col2, col3, col4]

    for col, name in zip(cols, model_names):
        with col:
            _render_model_column_card(name, model_results[name])

    # 3. Comparative Probability Chart
    st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
    prob_chart = make_multimodel_prob_comparison_chart(model_results)
    st.plotly_chart(
        prob_chart,
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )

    # 4. Snapshot & Known Patterns Below
    left, right = st.columns([1, 1.2])
    with left:
        st.subheader("Customer snapshot")
        snap = pd.DataFrame(
            {
                "Field": list(result["profile"].keys()),
                "Value": list(result["profile"].values()),
            }
        )
        st.dataframe(snap, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Matches known risk patterns")
        factors = compute_risk_factors(result["form"])
        if factors:
            _render_pattern_list(factors)
        else:
            st.info("No strong pattern matches for this profile.")


def _render_result(result: dict, feature_columns) -> None:
    rand_info = result.get("random_customer_info")
    mode = result.get("mode", "Single Model")

    # Render ground truth banner if sampled from dataset
    if rand_info and rand_info.get("customer_id"):
        actual_churn = rand_info.get("actual_churn", "Unknown")
        actual_bool = 1 if str(actual_churn).lower() in ["yes", "1", "true"] else 0
        if mode == "Single Model":
            matched = result.get("prediction") == actual_bool
            match_note = "✓ Model prediction matched actual outcome" if matched else "✗ Model prediction differed from actual"
        else:
            model_results = result.get("model_results", {})
            matches = sum(1 for m in model_results.values() if m["prediction"] == actual_bool)
            match_note = f"✓ {matches}/{len(model_results)} models matched"
        _render_ground_truth_banner(rand_info, match_note)

    if mode == "Single Model":
        _render_single_model_result(result, feature_columns)
    else:
        _render_compare_all_result(result)


def render_predict_tab(sidebar: dict, feature_columns, scaler) -> None:
    """Entry point called from app.py inside the Prediction Dashboard tab."""
    if sidebar["predict_clicked"]:
        _run_prediction(sidebar, feature_columns, scaler)

    result = st.session_state.get("last_prediction")
    if result is None or result.get("mode") != sidebar.get("mode"):
        _render_empty_state()
    else:
        _render_result(result, feature_columns)

