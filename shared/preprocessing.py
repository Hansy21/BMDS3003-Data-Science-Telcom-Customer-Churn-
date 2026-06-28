"""
============================================================================
 SHARED PREPROCESSING  —  RUN THIS FIRST, BEFORE ANY MODEL
============================================================================
 Why this file exists:
   All 4 members must train on the EXACT SAME data, otherwise comparing the
   models is unfair and meaningless. This script does the cleaning, encoding,
   scaling and the train/test split ONCE, then saves the result so every
   member loads identical data.

 What it produces (inside shared/processed/):
   - X_train.csv, X_test.csv, y_train.csv, y_test.csv   (the split data)
   - scaler.pkl              (the fitted StandardScaler, needed by the app)
   - feature_columns.pkl     (list of column names, needed by the app)

 How to run (from the project root folder):
   python shared/preprocessing.py
============================================================================
"""

import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Always read/write using paths relative to THIS file -------------------
# (so it works no matter which folder you run it from)
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CSV_PATH = os.path.join(PROJECT_ROOT, "Telco_Cusomer_Churn.csv")
OUT_DIR = os.path.join(HERE, "processed")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # ── 1. Load the raw data ────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # ── 2. Fix TotalCharges (stored as text, has 11 blank values) ───────────
    # Blank strings become NaN; those are brand-new customers (tenure = 0)
    # who have never been billed, so 0 is the correct value.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isnull().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    print(f"Fixed TotalCharges ({n_missing} missing values filled with 0)")

    # ── 3. Drop customerID (just an ID, useless for prediction) ─────────────
    df = df.drop(columns=["customerID"])

    # ── 4. Encode the target column: Churn  Yes->1, No->0 ───────────────────
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # ── 5. Encode binary text columns (only 2 possible values) → 1/0 ────────
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    df["Partner"] = df["Partner"].map({"Yes": 1, "No": 0})
    df["Dependents"] = df["Dependents"].map({"Yes": 1, "No": 0})
    df["PhoneService"] = df["PhoneService"].map({"Yes": 1, "No": 0})
    df["PaperlessBilling"] = df["PaperlessBilling"].map({"Yes": 1, "No": 0})

    # ── 6. One-hot encode the multi-value text columns ──────────────────────
    multi_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod",
    ]
    df = pd.get_dummies(df, columns=multi_cols)
    print(f"After encoding: {df.shape[1]} columns")

    # ── 7. Split into features (X) and target (y) ───────────────────────────
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # ── 8. Train/test split FIRST, then scale ───────────────────────────────
    # IMPORTANT (improvement over the original notebook):
    # We split BEFORE scaling and fit the scaler on the TRAINING data only.
    # Fitting the scaler on the full dataset would leak information from the
    # test set into training. stratify=y keeps the same churn ratio in both.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 9. Scale the 3 numeric columns (fit on train, apply to both) ────────
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # ── 10. Save everything so every member loads the SAME data ─────────────
    X_train.to_csv(os.path.join(OUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUT_DIR, "y_test.csv"), index=False)

    with open(os.path.join(OUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUT_DIR, "feature_columns.pkl"), "wb") as f:
        pickle.dump(X_train.columns.tolist(), f)

    print("\n[OK] Preprocessing complete. Saved to shared/processed/")
    print(f"     Train set: {X_train.shape[0]} rows")
    print(f"     Test set:  {X_test.shape[0]} rows")
    print(f"     Features:  {X_train.shape[1]} columns")
    print("\nNext step: each member runs their own train_*.py script.")


if __name__ == "__main__":
    main()
