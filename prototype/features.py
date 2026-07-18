"""
Helpers that turn sidebar form values into a model-ready feature row,
plus risk-band logic for the prediction dashboard.
"""

import pandas as pd
import streamlit as st


def apply_preset(preset: dict) -> None:
    """Copy a preset dict into session_state keys used by sidebar widgets."""
    for key, value in preset.items():
        st.session_state[key] = value


def risk_band(probability: float, threshold: float):
    """
    Map a churn probability to (band_name, accent_colour, action_text).

    Bands are aligned with the model's decision threshold so the UI
    stays consistent with CHURN / STAY classification.
    """
    if probability >= max(0.70, threshold + 0.20):
        return (
            "HIGH",
            "#c0392b",
            "Consider a discount, contract upgrade, or support call.",
        )
    if probability >= threshold:
        return (
            "MEDIUM",
            "#d68910",
            "Monitor closely and offer a light retention incentive.",
        )
    return (
        "LOW",
        "#1e8449",
        "Customer is likely stable — routine engagement is enough.",
    )


def build_input_dataframe(form_values: dict, feature_columns, scaler):
    """
    Encode sidebar form values the same way as shared/preprocessing.py,
    then scale tenure / MonthlyCharges / TotalCharges with the saved scaler.
    """
    input_data = {
        "gender": 1 if form_values["gender"] == "Male" else 0,
        "SeniorCitizen": 1 if form_values["senior_citizen"] == "Yes" else 0,
        "Partner": 1 if form_values["partner"] == "Yes" else 0,
        "Dependents": 1 if form_values["dependents"] == "Yes" else 0,
        "tenure": form_values["tenure"],
        "PhoneService": 1 if form_values["phone_service"] == "Yes" else 0,
        "PaperlessBilling": 1 if form_values["paperless_billing"] == "Yes" else 0,
        "MonthlyCharges": form_values["monthly_charges"],
        "TotalCharges": form_values["total_charges"],
    }

    multi_value_cols = {
        "MultipleLines": form_values["multiple_lines"],
        "InternetService": form_values["internet_service"],
        "OnlineSecurity": form_values["online_security"],
        "OnlineBackup": form_values["online_backup"],
        "DeviceProtection": form_values["device_protection"],
        "TechSupport": form_values["tech_support"],
        "StreamingTV": form_values["streaming_tv"],
        "StreamingMovies": form_values["streaming_movies"],
        "Contract": form_values["contract"],
        "PaymentMethod": form_values["payment_method"],
    }
    for col, value in multi_value_cols.items():
        for cat in feature_columns:
            if cat.startswith(col + "_"):
                input_data[cat] = 1 if cat == f"{col}_{value}" else 0

    input_df = pd.DataFrame([input_data]).reindex(
        columns=feature_columns, fill_value=0
    )
    input_df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
        input_df[["tenure", "MonthlyCharges", "TotalCharges"]]
    )
    return input_df.astype(float)


def profile_summary(form_values: dict) -> dict:
    """Short field→value map shown on the prediction dashboard."""
    return {
        "Contract": form_values["contract"],
        "Tenure (months)": form_values["tenure"],
        "Internet": form_values["internet_service"],
        "Monthly charges": f"RM {form_values['monthly_charges']:.2f}",
        "Payment": form_values["payment_method"],
        "Tech support": form_values["tech_support"],
        "Online security": form_values["online_security"],
    }
