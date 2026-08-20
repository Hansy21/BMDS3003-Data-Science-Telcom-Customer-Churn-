"""
Helpers that turn sidebar form values into a model-ready feature row,
plus risk-band logic for the prediction dashboard.
"""

import numpy as np
import pandas as pd
import streamlit as st

# Prefixes used by the one-hot encoded columns -> a readable display label.
_READABLE_PREFIX = {
    "MultipleLines": "Multiple Lines",
    "InternetService": "Internet Service",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "Contract": "Contract",
    "PaymentMethod": "Payment Method",
}


def humanize_feature(col_name: str, form_values: dict) -> str:
    """Turn an encoded column name into a plain-English label using the
    customer's actual entered value (not the scaled/encoded number)."""
    if col_name == "tenure":
        return f"Tenure: {form_values['tenure']} months"
    if col_name == "MonthlyCharges":
        return f"Monthly Charges: RM {form_values['monthly_charges']:.2f}"
    if col_name == "TotalCharges":
        return f"Total Charges: RM {form_values['total_charges']:.2f}"
    if col_name == "gender":
        return f"Gender: {form_values['gender']}"
    if col_name == "SeniorCitizen":
        return f"Senior Citizen: {form_values['senior_citizen']}"
    if col_name == "Partner":
        return f"Partner: {form_values['partner']}"
    if col_name == "Dependents":
        return f"Dependents: {form_values['dependents']}"
    if col_name == "PhoneService":
        return f"Phone Service: {form_values['phone_service']}"
    if col_name == "PaperlessBilling":
        return f"Paperless Billing: {form_values['paperless_billing']}"
    if "_" in col_name:
        prefix, _, suffix = col_name.partition("_")
        label = _READABLE_PREFIX.get(prefix, prefix)
        return f"{label}: {suffix}"
    return col_name


def get_top_drivers(
    model,
    model_name: str,
    form_values: dict,
    input_df: pd.DataFrame,
    feature_columns,
    top_n: int = 5,
):
    """
    Build a SPECIFIC, per-customer explanation instead of a generic global
    importance chart:

    - Logistic Regression: uses signed coefficients, so we can say whether
      each factor pushed the prediction UP or DOWN for this exact customer
      (contribution = coefficient x this customer's encoded value).
    - Random Forest / Decision Tree: feature_importances_ has no sign, so
      we only claim "this is a factor the model weighs heavily" (no
      direction), and we filter to features that are actually TRUE for
      this customer (continuous features always included; one-hot
      category features only included if this customer is in that
      category) so the list is relevant to this profile, not a generic
      top-10 for the whole dataset.
    - KNN: has neither — returns None so the caller can show a fallback.
    """
    row = input_df.iloc[0]

    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        contributions = coefs * row.values
        order = np.argsort(-np.abs(contributions))[:top_n]
        drivers = []
        for i in order:
            col = feature_columns[i]
            direction = "up" if contributions[i] > 0 else "down"
            drivers.append(
                {
                    "label": humanize_feature(col, form_values),
                    "direction": direction,
                    "magnitude": abs(contributions[i]),
                }
            )
        return drivers

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        candidate_idx = [
            i
            for i, col in enumerate(feature_columns)
            if "_" not in col or row.iloc[i] == 1
        ]
        candidate_idx.sort(key=lambda i: -importances[i])
        drivers = []
        for i in candidate_idx[:top_n]:
            col = feature_columns[i]
            drivers.append(
                {
                    "label": humanize_feature(col, form_values),
                    "direction": None,  # tree importance has no sign
                    "magnitude": importances[i],
                }
            )
        return drivers

    return None


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

    input_df = pd.DataFrame([input_data]).reindex(columns=feature_columns, fill_value=0)
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


# ---------------------------------------------------------------------------
# Churn rates by category, taken directly from Section 2.3/2.5 of the
# written report (computed on the full 7,043-row dataset). Reusing these
# exact figures means the dashboard's "why this customer" explanation
# always matches the numbers already presented in the report — no
# separate, harder-to-explain "importance score" needed.
# Overall baseline churn rate across all customers: 26.5%.
# ---------------------------------------------------------------------------
_OVERALL_CHURN_RATE = 26.5

_CHURN_RATE_LOOKUP = {
    ("contract", "Contract"): {
        "Month-to-month": 42.7,
        "One year": 11.3,
        "Two year": 2.8,
    },
    ("internet_service", "Internet service"): {
        "Fiber optic": 41.9,
        "DSL": 19.0,
        "No": 7.4,
    },
    ("payment_method", "Payment method"): {
        "Electronic check": 45.3,
        "Mailed check": 19.1,
        "Bank transfer (automatic)": 16.7,
        "Credit card (automatic)": 15.2,
    },
    ("online_security", "Online security"): {
        "No": 41.8,
        "Yes": 14.6,
        "No internet service": 7.4,
    },
    ("tech_support", "Tech support"): {
        "No": 41.6,
        "Yes": 15.2,
        "No internet service": 7.4,
    },
    ("senior_citizen", "Senior citizen"): {"Yes": 41.7, "No": 23.6},
    ("partner", "Has a partner"): {"No": 33.0, "Yes": 19.7},
    ("dependents", "Has dependents"): {"No": 31.3, "Yes": 15.5},
}


def compute_risk_factors(form_values: dict, top_n: int = 5) -> list[dict]:
    """
    Turn this specific customer's actual answers into a ranked, plain-
    English list of risk/protective factors, using the real churn-rate
    percentages from the report's EDA (Section 2.3/2.5) rather than an
    abstract model-importance score. Sorted by how far each factor's
    rate sits from the 26.5% overall baseline, so the most explanatory
    factors for THIS customer come first.
    """
    factors = []
    for (field_key, label), rate_map in _CHURN_RATE_LOOKUP.items():
        value = form_values.get(field_key)
        rate = rate_map.get(value)
        if rate is None:
            continue
        delta = rate - _OVERALL_CHURN_RATE
        if delta >= 8:
            tag, icon = "risk", "!"
        elif delta <= -8:
            tag, icon = "protective", "✓"
        else:
            tag, icon = "neutral", "-"
        factors.append(
            {
                "label": label,
                "value": value,
                "rate": rate,
                "delta": delta,
                "tag": tag,
                "icon": icon,
            }
        )

    # Tenure gets its own rule (continuous, not a lookup table) — short
    # tenure on a month-to-month contract is the single riskiest combo
    # identified in Section 2.5.3 (51.4% churn in the first year).
    tenure = form_values.get("tenure", 0)
    contract = form_values.get("contract")
    if contract == "Month-to-month" and tenure <= 12:
        factors.append(
            {
                "label": "Tenure + Contract combo",
                "value": f"{tenure} months on a month-to-month plan",
                "rate": 51.4,
                "delta": 51.4 - _OVERALL_CHURN_RATE,
                "tag": "risk",
                "icon": "!",
            }
        )
    elif tenure >= 49:
        factors.append(
            {
                "label": "Tenure",
                "value": f"{tenure} months (long-standing customer)",
                "rate": 12.0,
                "delta": 12.0 - _OVERALL_CHURN_RATE,
                "tag": "protective",
                "icon": "✓",
            }
        )

    factors.sort(key=lambda f: abs(f["delta"]), reverse=True)
    return factors[:top_n]
