"""
============================================================================
 SHARED EVALUATION HELPERS  —  used by every member's training script
============================================================================
 You do NOT run this file directly. Each member's train_*.py imports the two
 functions below so that ALL models are loaded, scored and saved in EXACTLY
 the same way. This guarantees a fair comparison.

   load_processed_data()  -> X_train, X_test, y_train, y_test
   evaluate_and_save(...)  -> prints metrics, saves plots + metrics + model
============================================================================
"""

import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")  # save plots to file without needing a screen
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    classification_report,
)

# --- Standard project folders (relative to this file) ----------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
PROCESSED_DIR = os.path.join(HERE, "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def load_processed_data():
    """Load the identical train/test split produced by preprocessing.py."""
    if not os.path.exists(os.path.join(PROCESSED_DIR, "X_train.csv")):
        raise FileNotFoundError(
            "Processed data not found. Run this first from the project root:\n"
            "    python shared/preprocessing.py"
        )
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    # .squeeze() turns the single-column DataFrame into a Series
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()
    return X_train, X_test, y_train, y_test


def evaluate_and_save(
    model_name, model, X_test, y_test, best_params=None, threshold=None
):
    """
    Score a trained model, print a report, and save:
      - results/<model_name>_metrics.json   (numbers for the compare script)
      - results/<model_name>_confusion.png   (confusion matrix image)
      - results/<model_name>_roc.png         (ROC curve image)
      - models/<model_name>.pkl              (the trained model)

    threshold : float or None
      Optional custom decision threshold for turning predict_proba into a
      class label (e.g. 0.35 instead of the default 0.5). Left as None,
      behaviour is 100% unchanged (uses model.predict() like before), so
      this is safe for every other member's script. Only pass a value here
      if you deliberately tuned it (e.g. via out-of-fold CV probabilities,
      never the test set) to trade some precision for higher recall.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Predictions ─────────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of churn (class 1)
    if threshold is None:
        y_pred = model.predict(X_test)  # unchanged default behaviour (0.5 cut)
    else:
        y_pred = (y_prob >= threshold).astype(int)

    # ── Metrics (focus on recall: catching real churners matters most) ──────
    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "best_params": best_params if best_params else {},
        "threshold": round(threshold, 4) if threshold is not None else 0.5,
    }

    print(f"\n========== {model_name} ==========")
    print(f"Accuracy : {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall   : {metrics['recall']}")
    print(f"F1-score : {metrics['f1']}")
    print(f"ROC-AUC  : {metrics['roc_auc']}")
    print(
        f"Threshold: {metrics['threshold']}"
        + ("" if threshold is None else "  (tuned, not the default 0.5)")
    )
    if best_params:
        print(f"Best params: {best_params}")
    print("---")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # ── Save metrics as JSON (the compare script reads these) ───────────────
    with open(os.path.join(RESULTS_DIR, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Confusion matrix image ──────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_confusion.png"), dpi=120)
    plt.close()

    # ── ROC curve image ─────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {metrics['roc_auc']})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} — ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_roc.png"), dpi=120)
    plt.close()

    # ── Save the trained model ──────────────────────────────────────────────
    with open(os.path.join(MODELS_DIR, f"{model_name}.pkl"), "wb") as f:
        pickle.dump(model, f)

    print(f"[OK] Saved metrics, plots and model for {model_name}")
    return metrics
