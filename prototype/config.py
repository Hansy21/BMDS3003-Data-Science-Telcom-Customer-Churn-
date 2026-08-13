"""
Paths, customer presets, and other constants used by the Streamlit app.
"""

import os

# ---------------------------------------------------------------------------
# Project folders (relative to where you run `streamlit run app.py`)
# ---------------------------------------------------------------------------
MODELS_DIR = "models"
RESULTS_DIR = "results"
PROCESSED_DIR = os.path.join("shared", "processed")
EDA_DIR = os.path.join("results", "eda")

# Files that are not trained member models
NON_MODEL_PICKLES = ("best_model.pkl", "scaler.pkl", "feature_columns.pkl")

# ---------------------------------------------------------------------------
# Customer form presets (sidebar quick-fill buttons)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "gender": "Female",
    "senior_citizen": "No",
    "partner": "Yes",
    "dependents": "No",
    "tenure": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 65.0,
    "total_charges": 1500.0,
}

AT_RISK = {
    **DEFAULTS,
    "tenure": 2,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "tech_support": "No",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 95.0,
    "total_charges": 190.0,
}

LOYAL = {
    **DEFAULTS,
    "tenure": 60,
    "contract": "Two year",
    "internet_service": "DSL",
    "online_security": "Yes",
    "tech_support": "Yes",
    "paperless_billing": "No",
    "payment_method": "Credit card (automatic)",
    "monthly_charges": 45.0,
    "total_charges": 2700.0,
}

# Form select options (keep in one place so sidebar stays tidy)
OPTIONS = {
    "gender": ["Male", "Female"],
    "yes_no": ["Yes", "No"],
    "no_yes": ["No", "Yes"],
    "multiple_lines": ["No", "Yes", "No phone service"],
    "internet_service": ["DSL", "Fiber optic", "No"],
    "internet_addon": ["No", "Yes", "No internet service"],
    "contract": ["Month-to-month", "One year", "Two year"],
    "payment_method": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}