"""
Prediction Dashboard tab — banner, KPIs, charts, snapshot, feature drivers.
"""

import numpy as np
import pandas as pd
import streamlit as st

from prototype.charts import (
    make_donut,
    make_gauge,
    make_importance_chart,
    make_risk_bar,
)
from prototype.features import build_input_dataframe, profile_summary, risk_band
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

    # KPI cards
    cols = st.columns(4)
    cards = [
        ("Churn probability", f"{result['probability'] * 100:.1f}%"),
        ("Stay probability", f"{(1 - result['probability']) * 100:.1f}%"),
        ("Risk level", result["band"]),
        ("Decision", "CHURN" if is_churn else "STAY"),
    ]
    for col, (label, value) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
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

    # Charts
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(
            make_gauge(result["probability"], result["threshold"], result["accent"]),
            use_container_width=True,
        )
    with g2:
        st.plotly_chart(
            make_donut(result["probability"], result["accent"]),
            use_container_width=True,
        )

    st.plotly_chart(
        make_risk_bar(result["probability"], result["accent"]),
        use_container_width=True,
    )

    # Snapshot + feature drivers
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
        st.caption(
            "Black line on the gauge marks this model’s decision threshold. "
            "Probability at or above the threshold → CHURN."
        )

    with right:
        st.subheader("🧭 What drives this model?")
        imp_model = load_model(result["model"])
        importances = None
        if hasattr(imp_model, "feature_importances_"):
            importances = imp_model.feature_importances_
        elif hasattr(imp_model, "coef_"):
            importances = np.abs(imp_model.coef_[0])

        if importances is not None:
            st.plotly_chart(
                make_importance_chart(feature_columns, importances),
                use_container_width=True,
            )
            st.caption(
                "Global importance for the model (not unique to this single customer)."
            )
        else:
            st.info(
                f"**{result['model']}** does not expose feature importances "
                "(e.g. KNN is instance-based). Compare tree/linear models for drivers."
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
