"""
Sidebar UI: model picker, quick presets, full customer form, Predict button.

Returns a dict of selections so the main page can run predictions.
"""

import streamlit as st

from prototype.config import AT_RISK, DEFAULTS, LOYAL, OPTIONS
from prototype.features import apply_preset
from prototype.loaders import (
    has_test_set,
    load_model,
    load_model_threshold,
    load_test_set,
)


def render_sidebar(model_names: list[str], best_model_name: str | None) -> dict:
    """
    Draw the full sidebar and return:

        {
          "model_name", "model", "threshold",
          "form",          # dict of all customer fields
          "predict_clicked",
          "X_test", "y_test", "has_test",
        }
    """
    with st.sidebar:
        st.header("Controls")
        st.caption("All inputs live here. Results appear on the main page.")

        # ── Model ──────────────────────────────────────────────────────────
        default_index = (
            model_names.index(best_model_name)
            if best_model_name in model_names
            else 0
        )
        model_name = st.selectbox(
            "Prediction model",
            model_names,
            index=default_index,
            help="Every .pkl file in models/ appears here.",
        )
        model = load_model(model_name)
        threshold = load_model_threshold(model_name)

        if model_name == best_model_name:
            st.success(f"Best by F1: **{model_name}**")
        st.caption(f"Decision threshold: **{threshold:.2f}**")

        if has_test_set():
            X_test, y_test = load_test_set()
            st.caption(f"Test set available: {len(X_test)} customers")
            test_ok = True
        else:
            X_test, y_test = None, None
            test_ok = False
            st.warning("Run `python shared/preprocessing.py` for Model Insights.")

        # ── Quick examples ─────────────────────────────────────────────────
        st.divider()
        st.subheader("Quick examples")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🟢 Loyal", use_container_width=True):
                apply_preset(LOYAL)
                st.session_state.last_prediction = None
                st.rerun()
        with col_b:
            if st.button("🔴 At-risk", use_container_width=True):
                apply_preset(AT_RISK)
                st.session_state.last_prediction = None
                st.rerun()
        # Added a clean, functional UI icon for reset
        if st.button("↺ Reset form", use_container_width=True):
            apply_preset(DEFAULTS)
            st.session_state.last_prediction = None
            st.rerun()

        # ── Customer profile ───────────────────────────────────────────────
        st.divider()
        st.subheader("Customer profile")

        with st.expander("Demographics", expanded=True):
            gender = st.selectbox("Gender", OPTIONS["gender"], key="gender")
            senior_citizen = st.selectbox(
                "Senior citizen", OPTIONS["no_yes"], key="senior_citizen"
            )
            partner = st.selectbox("Partner", OPTIONS["yes_no"], key="partner")
            dependents = st.selectbox(
                "Dependents", OPTIONS["yes_no"], key="dependents"
            )

        with st.expander("Services", expanded=True):
            tenure = st.slider("Tenure (months)", 0, 72, key="tenure")
            phone_service = st.selectbox(
                "Phone service", OPTIONS["yes_no"], key="phone_service"
            )
            multiple_lines = st.selectbox(
                "Multiple lines", OPTIONS["multiple_lines"], key="multiple_lines"
            )
            internet_service = st.selectbox(
                "Internet service",
                OPTIONS["internet_service"],
                key="internet_service",
            )
            online_security = st.selectbox(
                "Online security",
                OPTIONS["internet_addon"],
                key="online_security",
            )
            online_backup = st.selectbox(
                "Online backup", OPTIONS["internet_addon"], key="online_backup"
            )
            device_protection = st.selectbox(
                "Device protection",
                OPTIONS["internet_addon"],
                key="device_protection",
            )
            tech_support = st.selectbox(
                "Tech support", OPTIONS["internet_addon"], key="tech_support"
            )
            streaming_tv = st.selectbox(
                "Streaming TV", OPTIONS["internet_addon"], key="streaming_tv"
            )
            streaming_movies = st.selectbox(
                "Streaming movies",
                OPTIONS["internet_addon"],
                key="streaming_movies",
            )

        with st.expander("Billing", expanded=True):
            contract = st.selectbox(
                "Contract", OPTIONS["contract"], key="contract"
            )
            paperless_billing = st.selectbox(
                "Paperless billing", OPTIONS["yes_no"], key="paperless_billing"
            )
            payment_method = st.selectbox(
                "Payment method",
                OPTIONS["payment_method"],
                key="payment_method",
            )
            monthly_charges = st.slider(
                "Monthly charges (RM)", 20.0, 120.0, key="monthly_charges"
            )
            total_charges = st.slider(
                "Total charges (RM)", 0.0, 9000.0, key="total_charges"
            )

        st.divider()
        predict_clicked = st.button(
            "Predict churn",
            type="primary",
            use_container_width=True,
        )

    form = {
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
    }

    return {
        "model_name": model_name,
        "model": model,
        "threshold": threshold,
        "form": form,
        "predict_clicked": predict_clicked,
        "X_test": X_test,
        "y_test": y_test,
        "has_test": test_ok,
    }
