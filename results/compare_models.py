"""
============================================================================
 COMPARE ALL MODELS  —  RUN THIS LAST, AFTER ALL 4 MEMBERS HAVE TRAINED
============================================================================
 This reads every *_metrics.json file the members produced and builds:
   - a printed comparison table
   - results/model_comparison.csv        (for the report)
   - results/model_comparison.png        (bar chart for the report)
   - models/best_model.pkl + best_model_name.txt   (used by the Streamlit app)

 The "best" model is chosen by F1 score, because for churn we care about
 correctly catching customers who will actually leave (recall), not just
 raw accuracy.

 HOW TO RUN (from the project root folder):
   python results/compare_models.py
============================================================================
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
PROJECT_ROOT = os.path.dirname(HERE)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Which metric decides the winner
RANK_BY = "f1"


def main():
    # ── 1. Load every metrics JSON the members saved ────────────────────────
    files = glob.glob(os.path.join(HERE, "*_metrics.json"))
    if not files:
        raise FileNotFoundError(
            "No *_metrics.json found in results/. Run the member train_*.py "
            "scripts first."
        )

    rows = []
    for path in files:
        with open(path) as f:
            rows.append(json.load(f))

    df = pd.DataFrame(rows)
    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    df = df[cols].sort_values(RANK_BY, ascending=False).reset_index(drop=True)

    print("\n=== MODEL COMPARISON (sorted by F1) ===")
    print(df.to_string(index=False))

    # ── 2. Save the table for the report ────────────────────────────────────
    df.to_csv(os.path.join(HERE, "model_comparison.csv"), index=False)

    # ── 3. Grouped bar chart of all metrics ─────────────────────────────────
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
    plt.savefig(os.path.join(HERE, "model_comparison.png"), dpi=120)
    plt.close()

    # ── 4. Promote the winning model for the Streamlit app ──────────────────
    best_name = df.iloc[0]["model"]
    best_pkl = os.path.join(MODELS_DIR, f"{best_name}.pkl")
    if os.path.exists(best_pkl):
        shutil.copyfile(best_pkl, os.path.join(MODELS_DIR, "best_model.pkl"))
        with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
            f.write(best_name)
        print(f"\n[OK] Best model: {best_name} "
              f"(F1 = {df.iloc[0]['f1']}, accuracy = {df.iloc[0]['accuracy']})")
        print("     Copied to models/best_model.pkl for the Streamlit app.")
    else:
        print(f"\n[WARN] {best_pkl} not found — re-run that member's script.")

    print("\nSaved: results/model_comparison.csv and results/model_comparison.png")


if __name__ == "__main__":
    main()
