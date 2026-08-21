"""Shared functions for loading data and checking model performance.

Do not run this file directly. Each model imports these functions so every
model uses the same test data, scores, charts, and saving process. This makes
the final comparison fair because the models are judged in the same way.
"""

import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")  # Save charts as files without opening a window.
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
    precision_recall_curve,
    classification_report,
)
from sklearn.calibration import calibration_curve

# Project folders used by these helper functions. Keeping them here prevents
# every model script from repeating the same folder setup.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
PROCESSED_DIR = os.path.join(HERE, "processed")
# Results holds report files and charts; models holds reusable trained models.
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def load_processed_data():
    """Load the prepared training and test data saved by preprocessing.py.

    X contains customer details, while y contains the known answer: churn (1)
    or no churn (0). Every model receives the same split from these files.
    """
    if not os.path.exists(os.path.join(PROCESSED_DIR, "X_train.csv")):
        raise FileNotFoundError(
            "Processed data not found. Run this first from the project root:\n"
            "    python shared/preprocessing.py"
        )
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    # Read the same four files that preprocessing.py saved for every member.
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    # Change each one-column table into a simple list of churn answers.
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()
    return X_train, X_test, y_train, y_test


def evaluate_and_save(
    model_name, model, X_test, y_test, best_params=None, threshold=None
):
    """Check one trained model and save its scores, charts, and model file.

    The model receives unseen test customers and returns a churn probability
    for each one. We convert that probability into a churn/no-churn answer,
    calculate standard performance scores, and save the results for comparison.

    A custom threshold can be given when the project chooses a decision point
    other than the usual 0.5. For example, a lower threshold marks more people
    as churn risks. If no threshold is given, the model's usual rule is used.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # exist_ok=True means these commands are safe even if the folders exist.
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Get each customer's churn probability and predicted result.
    # Class 1 means "Churn", so [:, 1] selects the churn probability column.
    y_prob = model.predict_proba(X_test)[:, 1]  # Chance that the customer will churn.
    if threshold is None:
        y_pred = model.predict(X_test)  # Use the model's usual decision point.
    else:
        # A probability at or above the chosen threshold becomes a churn label (1).
        y_pred = (y_prob >= threshold).astype(int)

    # Calculate the main scores used to judge the model.
    # Accuracy = how many answers were correct overall.
    # Precision = how often a predicted churner really churned.
    # Recall = how many real churners the model found.
    # F1 = one balanced score that combines precision and recall.
    # ROC-AUC = how well the model separates higher-risk from lower-risk customers.
    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        # Record the settings used, which helps reproduce the model later.
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
    # The classification report gives separate precision, recall, and F1 scores
    # for customers who stayed and customers who churned.

    # Save the scores as a JSON file so compare_models.py can read them later.
    with open(os.path.join(RESULTS_DIR, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save a confusion matrix. It shows correct predictions and the two kinds
    # of mistake: a false alarm and a churner the model failed to find.
    cm = confusion_matrix(y_test, y_pred)
    # Rows are the real answers and columns are the model's predictions.
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_confusion.png"), dpi=120)
    plt.close()

    # Save an ROC curve. A curve closer to the top-left corner means the model
    # is better at separating churners from non-churners across many thresholds.
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    # The curve is based on probabilities, so it checks many possible thresholds.
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

    # Save a precision-recall curve. It shows the trade-off: finding more real
    # churners can also lead to more false churn warnings.
    precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
    # This curve is especially useful when churners are less common than stayers.
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, linewidth=2, color="purple", label=f"{model_name}")
    plt.xlabel("Recall (True Positive Rate)")
    plt.ylabel("Precision")
    plt.title(f"{model_name} — Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_prc.png"), dpi=120)
    plt.close()

    # Save a chart of predicted churn chances for both real groups. Good models
    # usually give higher chances to people who really churned.
    plt.figure(figsize=(7, 5))
    # Density makes the two groups comparable even when their sizes differ.
    plt.hist(y_prob[y_test == 0], bins=20, alpha=0.5, label="Actual: No Churn", color="green", density=True)
    # Select probabilities for customers who really stayed, then draw their shape.
    plt.hist(y_prob[y_test == 1], bins=20, alpha=0.5, label="Actual: Churn", color="red", density=True)
    # Do the same for real churners. Clear separation between the two shapes is good.
    plt.xlabel("Predicted Probability of Churn")
    plt.ylabel("Density")
    plt.title(f"{model_name} — Probability Distribution")
    plt.legend(loc="upper center")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_prob_dist.png"), dpi=120)
    plt.close()

    # Save a calibration chart. For example, if people given a 70% churn chance
    # really churn about 70% of the time, the probabilities are well calibrated.
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    # Put predictions into 10 groups, then compare predicted chance with real rate.
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=model_name, color="darkorange")
    plt.plot([0, 1], [0, 1], 'k--', label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives (Actual Churn Rate)")
    plt.title(f"{model_name} — Calibration Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_calibration.png"), dpi=120)
    plt.close()

    # Save the trained model so the app can make future predictions without
    # training the model again.
    with open(os.path.join(MODELS_DIR, f"{model_name}.pkl"), "wb") as f:
        # Pickle stores the fitted Python model in a file for later reuse.
        pickle.dump(model, f)

    print(f"[OK] Saved metrics, plots and model for {model_name}")
    return metrics
