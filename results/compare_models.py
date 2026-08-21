"""Compare the results from all trained models.

Run this after every model has finished. Each model saves one small results
file containing its scores. This script reads those files, places the scores
side by side in a table and chart, and saves the winning model for the app.

The winner is chosen by F1 score. F1 balances precision (how often a churn
warning is correct) and recall (how many real churners are found), so it is
more useful than accuracy alone when churners are the smaller group.

Run: python results/compare_models.py
"""

import glob
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The project root is the parent folder of results/.
PROJECT_ROOT = os.path.dirname(HERE)
# This is where each trained model file is stored.
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Use F1 score to choose the winning model. The table is sorted with the
# highest F1 first, making the result easy to present.
RANK_BY = "f1"


def main():
    # 1. Load the saved results from every model.
    # Each member's training script creates a file ending in _metrics.json.
    # JSON is simply a small text file that stores named values, such as F1.
    files = glob.glob(os.path.join(HERE, "*_metrics.json"))
    # glob finds every result file whose name ends with _metrics.json.
    if not files:
        raise FileNotFoundError(
            "No *_metrics.json found in results/. Run the member train_*.py "
            "scripts first."
        )

    rows = []
    for path in files:
        # Read one model's saved scores and add them as one table row.
        with open(path) as f:
            rows.append(json.load(f))

    # Turn the list of model results into a table and keep only the scores
    # needed for the comparison chart and report.
    df = pd.DataFrame(rows)
    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    # Sort from highest to lowest F1 so the first row is the winning model.
    df = df[cols].sort_values(RANK_BY, ascending=False).reset_index(drop=True)

    print("\n=== MODEL COMPARISON (sorted by F1) ===")
    print(df.to_string(index=False))

    # 2. Save the comparison table for the report.
    # CSV can be opened directly in Excel or added to the written report.
    df.to_csv(os.path.join(HERE, "model_comparison.csv"), index=False)

    # 3. Create a chart that compares the model scores.
    # Every score ranges from 0 to 1, where a higher value is generally better.
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ax = df.set_index("model")[metrics].plot(
        kind="bar", figsize=(11, 6), edgecolor="black", width=0.8
    )
    ax.set_title("Model Comparison — Telco Customer Churn", fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    # Save the chart at a clear resolution for use in the report.
    plt.savefig(os.path.join(HERE, "model_comparison.png"), dpi=120)
    plt.close()

    # 4. Save a copy of the best model for the Streamlit app.
    # The app always looks for best_model.pkl, so this avoids changing the app
    # whenever a different model wins.
    best_name = df.iloc[0]["model"]
    # iloc[0] means the first row after sorting, which is the F1 winner.
    best_pkl = os.path.join(MODELS_DIR, f"{best_name}.pkl")
    if os.path.exists(best_pkl):
        # Make a copy with one fixed name that the Streamlit app expects.
        shutil.copyfile(best_pkl, os.path.join(MODELS_DIR, "best_model.pkl"))
        with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
            # Save the readable winner name so the app can display it.
            f.write(best_name)
        print(f"\n[OK] Best model: {best_name} "
              f"(F1 = {df.iloc[0]['f1']}, accuracy = {df.iloc[0]['accuracy']})")
        print("     Copied to models/best_model.pkl for the Streamlit app.")
    else:
        print(f"\n[WARN] {best_pkl} not found — re-run that member's script.")

    print("\nSaved: results/model_comparison.csv and results/model_comparison.png")


if __name__ == "__main__":
    main()
