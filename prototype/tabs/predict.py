"""
Prediction Dashboard tab — banner, KPIs, risk meter, snapshot, and two
"why" panels: a model-specific driver breakdown (get_top_drivers) and
an EDA-grounded risk-pattern match (compute_risk_factors).
"""

import pandas as pd
import streamlit as st

from prototype.features import (
    build_input_dataframe,
    compute_risk_factors,
    get_top_drivers,
    profile_summary,
    risk_band,
)
from prototype.loaders import load_model


def _run_prediction(sidebar: dict, feature_columns, scaler) -> None:
    """Score the current form and store the result in session_state."""
    input_df = build_input_dataframe(sidebar["form"], feature_columns, scaler)
    probability = float(sidebar["model"].predict_proba(input_df)[0][1])
    prediction = int(probability >= sidebar["threshold"])
    band, accent, action = risk_band(probability, sidebar["threshold"])

    st.session_state.last_prediction = {
        "model": sidebar["model_name"],
        "threshold": sidebar["threshold"],
        "probability": probability,
        "prediction": prediction,
        "band": band,
        "accent": accent,
        "action": action,
        "profile": profile_summary(sidebar["form"]),
        "form": sidebar["form"],
        "input_df": input_df,
    }


def _render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <h3>No prediction yet</h3>
            <p>Use the <b>sidebar</b> to pick a model and customer profile,<br>
            try a <b>Loyal</b> or <b>At-risk</b> example, then click
            <b>Predict churn</b>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Tip for demos: click **🔴 At-risk** in the sidebar → **Predict churn**, "
        "then try **🟢 Loyal** and compare the dashboard."
    )


def _render_driver_list(drivers: list) -> None:
    """Plain-English list of what pushed THIS customer's score up/down,
    from the model's own coefficients/importances — faster to read aloud
    during a demo than parsing a bar chart."""
    for d in drivers:
        if d["direction"] == "up":
            icon, note = "🔺", "pushes risk UP"
        elif d["direction"] == "down":
            icon, note = "🔻", "pushes risk DOWN"
        else:
            icon, note = "🔹", "a factor the model weighs heavily"
        st.markdown(f"{icon} **{d['label']}** — {note}")


def _render_pattern_list(factors: list) -> None:
    """Plain-English list matching this customer's answers against known
    churn-rate patterns from the report's EDA (Section 2.3/2.5)."""
    for f in factors:
        pct_str = f"{f['rate']:.1f}% churn rate"
        st.markdown(
            f"{f['icon']} **{f['label']}: {f['value']}** — {pct_str} (vs. 26.5% overall)"
        )


def _render_result(result: dict, feature_columns) -> None:
    is_churn = result["prediction"] == 1
    banner_color = "#c0392b" if is_churn else "#1e8449"
    banner_title = "⚠️ LIKELY TO CHURN" if is_churn else "✅ LIKELY TO STAY"
    banner_sub = (
        "This customer is predicted to leave the service."
        if is_churn
        else "This customer is predicted to remain with the company."
    )

    st.markdown(
        f"""
        <div class="result-banner"
             style="background: linear-gradient(135deg, {banner_color}, {banner_color}cc);">
            <h2>{banner_title}</h2>
            <p>{banner_sub}</p>
            <p style="margin-top:0.55rem; font-size:0.95rem;">
                Model: <b>{result['model']}</b> ·
                Risk band: <b>{result['band']}</b> ·
                Threshold: <b>{result['threshold']:.2f}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stat row — one asymmetric grid instead of 3 identical grey cards.
    # The hero cell carries the actual number + a fill track benchmarked
    # against the dataset's 26.5% baseline churn rate; the other two cells
    # read as a pill and an inline decision rather than repeating the same
    # boxed-card shell three times.
    pct = result["probability"] * 100
    decision_icon = "⚠️" if is_churn else "✅"
    decision_word = "CHURN" if is_churn else "STAY"

    st.markdown(
        f"""
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
                    {decision_icon} {decision_word}
                </div>
                <small style="color:#9ca3af; font-size:0.72rem;">
                    @ {result['threshold']:.2f} threshold
                </small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="action-box">
            <b>Recommended action</b><br>{result['action']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Risk meter — replaces the circular gauge. The number is already the
    # hero stat above, so this doesn't repeat it in a new shape; instead it
    # shows WHERE 67.1% (or whatever the value is) sits relative to this
    # model's actual LOW/MEDIUM/HIGH zone widths, which move with the
    # threshold slider instead of being fixed.
    thr_pct = result["threshold"] * 100
    high_start = max(70.0, thr_pct)  # keep zone 3 sane if threshold pushed past 70
    pct = result["probability"] * 100
    accent = result["accent"]

    st.markdown(
        f"""
        <div class="risk-meter">
            <div class="risk-meter-track"
                 style="background: linear-gradient(to right,
                     #d5f5e3 0%, #d5f5e3 {thr_pct}%,
                     #fdebd0 {thr_pct}%, #fdebd0 {high_start}%,
                     #f5b7b1 {high_start}%, #f5b7b1 100%);">
                <div class="risk-meter-threshold" style="left:{thr_pct}%"
                     title="Decision threshold: {result['threshold']:.2f}"></div>
                <div class="risk-meter-marker" style="left:{pct}%; --accent:{accent}">
                    <div class="risk-meter-marker-label">{pct:.1f}%</div>
                    <div class="risk-meter-marker-flag"></div>
                </div>
            </div>
            <div class="risk-meter-zones">
                <span style="flex:{max(thr_pct, 0.001)} 0 0; color:#1e8449;">LOW</span>
                <span style="flex:{max(high_start - thr_pct, 0.001)} 0 0; color:#d68910;">MEDIUM</span>
                <span style="flex:{max(100 - high_start, 0.001)} 0 0; color:#c0392b;">HIGH</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Grey tick marks this model's decision threshold ({result['threshold']:.2f}). "
        "Zone widths scale with the threshold, not fixed bands."
    )

    # Snapshot + two "why" panels
    left, right = st.columns([1, 1.2])
    with left:
        st.subheader("📋 Customer snapshot")
        snap = pd.DataFrame(
            {
                "Field": list(result["profile"].keys()),
                "Value": list(result["profile"].values()),
            }
        )
        st.dataframe(snap, use_container_width=True, hide_index=True)

        st.subheader("🎯 Matches known risk patterns")
        st.caption(
            "From the project's EDA (Section 2.3/2.5) — same figures as the written report."
        )
        factors = compute_risk_factors(result["form"])
        if factors:
            _render_pattern_list(factors)
        else:
            st.info("No strong pattern matches for this profile.")

    with right:
        st.subheader(f"🧭 Why {result['model']} made this call")
        model = load_model(result["model"])
        drivers = get_top_drivers(
            model, result["model"], result["form"], result["input_df"], feature_columns
        )
        if drivers:
            _render_driver_list(drivers)
            st.caption(
                "Specific to this customer's actual answers, not a generic top-10."
            )
        else:
            st.info(
                f"**{result['model']}** does not expose per-feature drivers "
                "(e.g. KNN is instance-based — it has no coefficients or "
                "importances to explain a single prediction). Try Logistic "
                "Regression, Decision Tree, or Random Forest instead."
            )


def render_predict_tab(sidebar: dict, feature_columns, scaler) -> None:
    """Entry point called from app.py inside the Prediction Dashboard tab."""
    if sidebar["predict_clicked"]:
        _run_prediction(sidebar, feature_columns, scaler)

    result = st.session_state.get("last_prediction")
    if result is None:
        _render_empty_state()
    else:
        _render_result(result, feature_columns)
