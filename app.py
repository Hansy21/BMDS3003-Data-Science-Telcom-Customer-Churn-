import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Load saved model, scaler, and feature columns ──────────────────────────
with open("models/best_model_rf.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor", page_icon="📡", layout="centered"
)

st.title("📡 Telco Customer Churn Predictor")
st.write("Fill in the customer details below to predict whether they will churn.")
st.divider()


# ── Input Form ───────────────────────────────────────────────────────────────
st.subheader("👤 Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

with col2:
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox(
        "Online Security", ["No", "Yes", "No internet service"]
    )
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox(
        "Device Protection", ["No", "Yes", "No internet service"]
    )
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

st.divider()
st.subheader("💳 Billing Details")

col3, col4 = st.columns(2)

with col3:
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox(
        "Streaming Movies", ["No", "Yes", "No internet service"]
    )
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

with col4:
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly_charges = st.slider("Monthly Charges (RM)", 20.0, 120.0, 65.0)
    total_charges = st.slider("Total Charges (RM)", 0.0, 9000.0, 1500.0)


# ── Predict Button ───────────────────────────────────────────────────────────
st.divider()

if st.button("🔍 Predict Churn", use_container_width=True):

    # Build input dictionary matching training columns
    input_data = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if phone_service == "Yes" else 0,
        "PaperlessBilling": 1 if paperless_billing == "Yes" else 0,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # One-hot encode multi-value columns
    multi_value_cols = {
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaymentMethod": payment_method,
    }

    for col, value in multi_value_cols.items():
        for cat in feature_columns:
            if cat.startswith(col + "_"):
                input_data[cat] = 1 if cat == f"{col}_{value}" else 0

    # Create DataFrame with correct column order
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Scale numerical columns
    input_df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
        input_df[["tenure", "MonthlyCharges", "TotalCharges"]]
    )

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # ── Show Result ──────────────────────────────────────────────────────────
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ This customer is **likely to CHURN**")
    else:
        st.success(f"✅ This customer is **likely to STAY**")

    st.metric(label="Churn Probability", value=f"{probability:.1%}")
    st.progress(float(probability))

    st.divider()

    # Risk level
    if probability >= 0.7:
        st.warning("🔴 HIGH RISK — Consider offering a discount or contract upgrade")
    elif probability >= 0.4:
        st.warning("🟡 MEDIUM RISK — Monitor this customer closely")
    else:
        st.info("🟢 LOW RISK — Customer is likely to stay")
