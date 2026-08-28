"""Prepare the customer data for every churn model.

Run this first. It cleans the original data, removes collinear categorical
redundancies, derives high-signal financial and behavioral ratios, keeps a
separate test set, and saves the prepared data. All models receive exactly the
same data, making comparison fair and methodologically sound.

It saves:
- Data splits: X_train.csv, X_test.csv, y_train.csv, y_test.csv
- Preprocessing artifacts: scaler.pkl, feature_columns.pkl, num_cols.pkl

Run: python shared/preprocessing.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CSV_PATH = os.path.join(PROJECT_ROOT, "Telco_Cusomer_Churn.csv")
OUT_DIR = os.path.join(HERE, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Continuous numeric columns to scale using StandardScaler
NUM_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "CostPerService",
    "HistoricalAvgMonthly",
    "ChargeDiscrepancy",
    "ChargeRatio",
    "EffectivePaidMonths",
    "UnpaidMonthDiff",
    "TotalServices",
    "SecuritySupportScore",
    "LogTotalCharges",
]


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform initial data hygiene: drop duplicates, impute TotalCharges, remove customerID,
    and collapse redundant 'No internet service'/'No phone service' to eliminate collinear noise."""
    df = df.copy()

    # 1. Remove duplicate rows
    n_duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Removed {n_duplicates} duplicate rows")

    # 2. Fix TotalCharges non-numeric blanks (customers with tenure = 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isnull().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    print(f"Fixed TotalCharges ({n_missing} missing values filled with 0.0)")

    # 3. Remove customerID label
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # 4. Collapse redundant 'No internet service' to 'No' across all 6 add-ons
    sub_services = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in sub_services:
        if col in df.columns:
            df[col] = df[col].replace({"No internet service": "No"})

    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer high-precision domain features, financial ratios, and behavioral flags.

    Features derived:
    1. Is_AutoPay: Frictionless billing (Auto Bank/Card vs. Manual Check)
    2. TotalServices: Total active subscribed services count (0 to 8)
    3. SecuritySupportScore: Total security/support services (0 to 4)
    4. HasFamilyPlan: Household anchor (Partner=Yes and Dependents=Yes)
    5. HistoricalAvgMonthly: Estimated historical monthly spend
    6. ChargeDiscrepancy: MonthlyCharges - HistoricalAvgMonthly (Bill shock indicator)
    7. CostPerService: Perceived value metric (MonthlyCharges / (TotalServices + 1))
    8. ChargeRatio: Rate shock ratio (MonthlyCharges / (HistoricalAvgMonthly + 1.0))
    9. EffectivePaidMonths: Reconstructed paid months (TotalCharges / (MonthlyCharges + 1e-5))
    10. UnpaidMonthDiff: Discrepancy between tenure and paid months (Billing dispute flag)
    11. LogTotalCharges: Log-transformed TotalCharges for skewness stabilization
    12. Is_New_Customer_Risk: High onboarding risk (tenure <= 6 on Month-to-month)
    13. HighSpenderFiberNoContract: High-value churn hazard (Fiber + >= RM75/mo on Month-to-month)
    14. LoyalContractCustomer: Retention anchor (tenure >= 24 on 1/2-year contract)
    15. FiberWithoutSupport: High frustration risk (Fiber optic without Tech Support)
    16. SeniorLivingAlone: Vulnerable demographic segment (Senior without partner/dependents)
    17. IsStreamingLover: Both StreamingTV and StreamingMovies active
    """
    df = df.copy()

    # Rule 1: Payment method automation
    df["Is_AutoPay"] = df["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)

    # Rule 2: Active service count (0 to 8)
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["TotalServices"] = (df[service_cols] == "Yes").sum(axis=1)

    # Rule 3: Security & Support Protection Score (0 to 4)
    sec_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    df["SecuritySupportScore"] = (df[sec_cols] == "Yes").sum(axis=1)

    # Rule 4: Family / Household plan anchor
    df["HasFamilyPlan"] = (
        (df["Partner"] == "Yes") & (df["Dependents"] == "Yes")
    ).astype(int)

    # Rule 5: Financial ratios and billing dispute proxies
    df["HistoricalAvgMonthly"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeDiscrepancy"] = df["MonthlyCharges"] - df["HistoricalAvgMonthly"]
    df["CostPerService"] = df["MonthlyCharges"] / (df["TotalServices"] + 1)
    df["ChargeRatio"] = df["MonthlyCharges"] / (df["HistoricalAvgMonthly"] + 1.0)
    df["EffectivePaidMonths"] = df["TotalCharges"] / (df["MonthlyCharges"] + 1e-5)
    df["UnpaidMonthDiff"] = df["tenure"] - df["EffectivePaidMonths"]
    df["LogTotalCharges"] = np.log1p(df["TotalCharges"])

    # Rule 6: High-precision domain interaction flags
    df["Is_New_Customer_Risk"] = (
        (df["tenure"] <= 6) & (df["Contract"] == "Month-to-month")
    ).astype(int)

    df["HighSpenderFiberNoContract"] = (
        (df["MonthlyCharges"] >= 75)
        & (df["InternetService"] == "Fiber optic")
        & (df["Contract"] == "Month-to-month")
    ).astype(int)

    df["LoyalContractCustomer"] = (
        (df["tenure"] >= 24) & (df["Contract"] != "Month-to-month")
    ).astype(int)

    df["FiberWithoutSupport"] = (
        (df["InternetService"] == "Fiber optic") & (df["TechSupport"] == "No")
    ).astype(int)

    df["SeniorLivingAlone"] = (
        (df["SeniorCitizen"] == 1) & (df["Partner"] == "No") & (df["Dependents"] == "No")
    ).astype(int)

    df["IsStreamingLover"] = (
        (df["StreamingTV"] == "Yes") & (df["StreamingMovies"] == "Yes")
    ).astype(int)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode binary columns and one-hot encode categorical features."""
    df = df.copy()

    # Target variable
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Binary columns
    binary_cols = {
        "gender": {"Male": 1, "Female": 0},
        "SeniorCitizen": {1: 1, 0: 0, "1": 1, "0": 0},
        "Partner": {"Yes": 1, "No": 0},
        "Dependents": {"Yes": 1, "No": 0},
        "PhoneService": {"Yes": 1, "No": 0},
        "PaperlessBilling": {"Yes": 1, "No": 0},
        "MultipleLines": {"Yes": 1, "No": 0},
        "OnlineSecurity": {"Yes": 1, "No": 0},
        "OnlineBackup": {"Yes": 1, "No": 0},
        "DeviceProtection": {"Yes": 1, "No": 0},
        "TechSupport": {"Yes": 1, "No": 0},
        "StreamingTV": {"Yes": 1, "No": 0},
        "StreamingMovies": {"Yes": 1, "No": 0},
    }
    for col, mapping in binary_cols.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # Multi-category columns for one-hot encoding (clean set without redundant dummy columns)
    multi_cols = ["InternetService", "Contract", "PaymentMethod"]
    existing_multi = [c for c in multi_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing_multi, drop_first=True)

    return df


def main():
    # 1. Load the original customer data
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Clean data & remove collinear noise
    df = clean_raw_data(df)

    # 3. Domain feature engineering & financial ratios
    df = engineer_features(df)

    # 4. Encoding
    df = encode_features(df)
    print(f"After engineering & encoding: {df.shape[1]} total columns")

    # 5. Separate features X and target y
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 6. Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Standardize numeric features (Fit strictly on train set)
    scaler = StandardScaler()
    X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
    X_test[NUM_COLS] = scaler.transform(X_test[NUM_COLS])

    # 8. Save prepared datasets
    X_train.to_csv(os.path.join(OUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUT_DIR, "y_test.csv"), index=False)

    feature_cols = X_train.columns.tolist()

    # Save preprocessing metadata to both processed/ and models/ for compatibility
    for dest_dir in [OUT_DIR, MODELS_DIR]:
        with open(os.path.join(dest_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(dest_dir, "feature_columns.pkl"), "wb") as f:
            pickle.dump(feature_cols, f)
        with open(os.path.join(dest_dir, "num_cols.pkl"), "wb") as f:
            pickle.dump(NUM_COLS, f)

    print("\n[OK] Preprocessing complete. Saved to shared/processed/ and models/")
    print(f"     Train set: {X_train.shape[0]} rows")
    print(f"     Test set:  {X_test.shape[0]} rows")
    print(f"     Features:  {X_train.shape[1]} columns")
    print(f"     Scaled numerical columns: {len(NUM_COLS)}")
    print("\nNext step: each member runs their own train_*.py script.")


if __name__ == "__main__":
    main()

