"""
Sidebar UI: model picker, quick presets, full customer form, Predict button.

Returns a dict of selections so the main page can run predictions.
"""

import time
import streamlit as st

from prototype.config import AT_RISK, DEFAULTS, LOYAL, OPTIONS
from prototype.features import apply_preset
from prototype.loaders import (
    has_test_set,
    load_model,
    load_model_threshold,
    load_test_set,
    sample_random_customer,
)


def render_sidebar(model_names: list[str], best_model_name: str | None) -> dict:
    """
    Draw the full sidebar and return:

        {
          "model_name", "model", "threshold",
          "selected_models", "all_models", "all_thresholds",
          "form",          # dict of all customer fields
          "predict_clicked",
          "X_test", "y_test", "has_test",
          "random_customer_info",
        }
    """
    with st.sidebar:
        st.header("Controls")
        st.caption("Configure profile or load random sample, then predict.")

        # ── Prediction Mode Selection ──────────────────────────────────────
        mode = st.radio(
            "Prediction Mode",
            ["Single Model", "Compare All Models"],
            horizontal=True,
            help="Single Model allows picking a specific model; Compare All Models evaluates all 4 models together.",
        )

        if "current_mode" not in st.session_state:
            st.session_state.current_mode = mode
        elif st.session_state.current_mode != mode:
            if st.session_state.get("last_prediction"):
                st.session_state.trigger_exit = True
            else:
                with st.spinner(f"Switching to {mode}..."):
                    time.sleep(0.8)
            st.session_state.current_mode = mode
            st.rerun()

        # ── Model Picker (only in Single Model mode) ──────────────────────
        if mode == "Single Model":
            default_index = (
                model_names.index(best_model_name)
                if best_model_name in model_names
                else 0
            )
            model_name = st.selectbox(
                "Select model to predict",
                model_names,
                index=default_index,
                help="Choose which model to run.",
            )
            model = load_model(model_name)
            threshold = load_model_threshold(model_name)

            if model_name == best_model_name:
                st.success(f"Best model by F1: **{model_name}**")
            st.caption(f"Decision threshold: **{threshold:.2f}**")
        else:
            model_name = best_model_name if best_model_name in model_names else model_names[0]
            model = None
            threshold = None
            st.info("⚡ Comparing all 4 models: **DecisionTree**, **KNN**, **LogisticRegression**, **RandomForest**.")

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
        st.subheader("Quick profiles & data input")

        # Visual feedback if preset or sample was applied
        if "preset_applied" in st.session_state and st.session_state.preset_applied:
            preset = st.session_state.preset_applied
            st.markdown(
                f"""
                <div class="preset-notification-banner {preset['type']}">
                    <div style="display:flex; align-items:center;">
                        <span class="preset-icon-badge">{preset['icon']}</span>
                        <div>
                            <div style="font-weight:700; font-size:0.86rem;">{preset['title']}</div>
                            <div style="font-size:0.75rem; opacity:0.85;">{preset['desc']}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🟢 Loyal", use_container_width=True):
                if st.session_state.get("last_prediction"):
                    st.session_state.trigger_exit = True
                else:
                    with st.spinner("Loading..."):
                        time.sleep(0.6)
                apply_preset(LOYAL)
                st.session_state.random_customer_info = None
                st.session_state.preset_applied = {
                    "type": "loyal",
                    "icon": "🟢",
                    "title": "Loyal Profile Loaded",
                    "desc": "Long tenure (65m), Two-year contract, Fiber Optic",
                }
                st.toast("🟢 Loaded Loyal Customer Profile!", icon="✨")
                st.rerun()
        with col_b:
            if st.button("🔴 At-risk", use_container_width=True):
                if st.session_state.get("last_prediction"):
                    st.session_state.trigger_exit = True
                else:
                    with st.spinner("Loading..."):
                        time.sleep(0.6)
                apply_preset(AT_RISK)
                st.session_state.random_customer_info = None
                st.session_state.preset_applied = {
                    "type": "at_risk",
                    "icon": "🔴",
                    "title": "At-Risk Profile Loaded",
                    "desc": "Short tenure (2m), Month-to-month, Fiber Optic",
                }
                st.toast("🔴 Loaded At-Risk Customer Profile!", icon="⚠️")
                st.rerun()

        col_c, col_d = st.columns(2)
        with col_c:
            if st.button("🎲 Random Sample", use_container_width=True, help="Sample a random customer from Telco_Cusomer_Churn.csv"):
                if st.session_state.get("last_prediction"):
                    st.session_state.trigger_exit = True
                else:
                    with st.spinner("Sampling..."):
                        time.sleep(0.6)
                sampled_form, cust_id, actual_churn = sample_random_customer()
                if sampled_form:
                    apply_preset(sampled_form)
                    st.session_state.random_customer_info = {
                        "customer_id": cust_id,
                        "actual_churn": actual_churn,
                    }
                    st.session_state.preset_applied = {
                        "type": "random",
                        "icon": "🎲",
                        "title": f"Random Customer Loaded",
                        "desc": f"ID: {cust_id} · Ground Truth: {actual_churn}",
                    }
                    st.toast(f"🎲 Sampled Customer {cust_id} from dataset!", icon="📊")
                    st.rerun()
        with col_d:
            if st.button("↺ Reset form", use_container_width=True):
                if st.session_state.get("last_prediction"):
                    st.session_state.trigger_exit = True
                else:
                    with st.spinner("Resetting..."):
                        time.sleep(0.6)
                apply_preset(DEFAULTS)
                st.session_state.random_customer_info = None
                st.session_state.preset_applied = {
                    "type": "reset",
                    "icon": "↺",
                    "title": "Form Reset to Defaults",
                    "desc": "All fields restored to initial default values",
                }
                st.toast("↺ Form reset to default inputs.", icon="🧹")
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

    # Build models mapping based on mode
    if mode == "Compare All Models":
        selected_models = model_names
    else:
        selected_models = [model_name]

    all_models = {name: load_model(name) for name in selected_models}
    all_thresholds = {name: load_model_threshold(name) for name in selected_models}

    return {
        "mode": mode,
        "model_name": model_name,
        "model": model,
        "threshold": threshold,
        "selected_models": selected_models,
        "all_models": all_models,
        "all_thresholds": all_thresholds,
        "all_model_names": model_names,
        "best_model_name": best_model_name,
        "form": form,
        "predict_clicked": predict_clicked,
        "X_test": X_test,
        "y_test": y_test,
        "has_test": test_ok,
        "random_customer_info": st.session_state.get("random_customer_info"),
    }
