"""Prepare the customer data for every churn model.

Run this first. It cleans the original data, turns text answers into numbers
that models can use, keeps a separate test set, and saves the prepared data.
All models then receive exactly the same data, making their comparison fair.

It saves four data files (training details, test details, training answers,
and test answers), plus the scaling settings and final list of columns. The
app needs the last two files to prepare new customer details in the same way.

Run: python shared/preprocessing.py
"""

import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Use file paths based on this file, so the script works from any folder.
# OUT_DIR is the folder where the prepared data and supporting files are saved.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CSV_PATH = os.path.join(PROJECT_ROOT, "Telco_Cusomer_Churn.csv")
# Create the output folder now if it does not already exist.
OUT_DIR = os.path.join(HERE, "processed")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # 1. Load the original customer data.
    # At this point, answers such as "Yes", "No", and contract types are still
    # text, so they cannot yet be used directly by most machine-learning models.
    df = pd.read_csv(CSV_PATH)
    # df is a table: every row is one customer and every column is one detail.
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Remove any repeated rows. Keeping duplicates could give the same customer
    # too much influence on the model.
    n_duplicates = df.duplicated().sum()
    # duplicated() marks repeated rows; sum() counts how many were found.
    df = df.drop_duplicates()
    print(f"Removed {n_duplicates} duplicate rows")

    # 2. Convert TotalCharges to numbers. Blank values mean no charges yet.
    # These customers have tenure = 0, so filling the blank with 0 is sensible.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # errors="coerce" turns non-numeric blanks into missing values instead of
    # stopping the script with an error.
    n_missing = df["TotalCharges"].isnull().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    print(f"Fixed TotalCharges ({n_missing} missing values filled with 0)")

    # 3. Remove the customer ID because it does not describe customer behaviour.
    # An ID is only a label; it should not influence a churn prediction.
    df = df.drop(columns=["customerID"])

    # 4. Turn the churn answer into numbers: Yes = 1 and No = 0.
    # This is the target: the answer the model will learn to predict.
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 5. Turn two-choice text answers into 1 and 0.
    # For example, Partner = 1 means "Yes" and Partner = 0 means "No".
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    df["Partner"] = df["Partner"].map({"Yes": 1, "No": 0})
    df["Dependents"] = df["Dependents"].map({"Yes": 1, "No": 0})
    df["PhoneService"] = df["PhoneService"].map({"Yes": 1, "No": 0})
    df["PaperlessBilling"] = df["PaperlessBilling"].map({"Yes": 1, "No": 0})
    # The original text values are replaced, so the whole data table is numeric.

    # 6. Turn answers with several choices into separate yes/no columns.
    # For example, a contract type becomes separate columns for the possible
    # contract choices. Models can then read each choice as a number.
    # One repeated column is left out because its meaning is already clear when
    # all the other columns for that question are 0. This avoids duplicate data.
    multi_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    # get_dummies creates columns such as Contract_One_year with 1 for yes and
    # 0 for no. drop_first=True omits one baseline choice per question.
    print(f"After encoding: {df.shape[1]} columns")

    # 7. Separate the customer details from the churn result we want to predict.
    # X contains the input details; y contains the known churn answer.
    X = df.drop(columns=["Churn"])
    # X is the information the model can look at when making a prediction.
    y = df["Churn"]
    # y is the correct answer that the model tries to learn.

    # 8. Put 80% of the data aside for learning and 20% aside for final testing.
    # Splitting before scaling keeps test-data information out of the training
    # process. stratify=y keeps a similar percentage of churners in both groups.
    # random_state=42 makes the same split happen every time the script runs.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # test_size=0.2 means 20% is held back for testing and 80% is for training.

    # 9. Put the three number-based columns on a similar scale for KNN.
    # KNN judges customers by distance. Scaling prevents large money values from
    # overpowering smaller values such as tenure. The scale is learned from the
    # training data only, then the same scale is applied to the test data.
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    # StandardScaler changes each numeric column so its average is near 0 and
    # its spread is similar to the other scaled columns.
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    # fit_transform learns the scale from the training data, then scales it.
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    # transform reuses the training scale; it does not learn from test data.

    # 10. Save the prepared data and supporting files for all team members.
    # Saving once means every model uses the same rows, columns, and test set.
    X_train.to_csv(os.path.join(OUT_DIR, "X_train.csv"), index=False)
    # index=False avoids saving pandas' row numbers as an unnecessary column.
    X_test.to_csv(os.path.join(OUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUT_DIR, "y_test.csv"), index=False)

    with open(os.path.join(OUT_DIR, "scaler.pkl"), "wb") as f:
        # Save the learned scale so the app can prepare new customers correctly.
        pickle.dump(scaler, f)
    with open(os.path.join(OUT_DIR, "feature_columns.pkl"), "wb") as f:
        # Save the column order because models expect new data in this order.
        pickle.dump(X_train.columns.tolist(), f)

    print("\n[OK] Preprocessing complete. Saved to shared/processed/")
    print(f"     Train set: {X_train.shape[0]} rows")
    print(f"     Test set:  {X_test.shape[0]} rows")
    print(f"     Features:  {X_train.shape[1]} columns")
    print("\nNext step: each member runs their own train_*.py script.")


if __name__ == "__main__":
    main()
