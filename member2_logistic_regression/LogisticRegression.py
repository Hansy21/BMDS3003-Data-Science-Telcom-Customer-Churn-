"""
============================================================================
 MEMBER 2  —  LOGISTIC REGRESSION
============================================================================
 Role in this project:
   Logistic Regression is a linear, probabilistic classifier. It models the
   log-odds of churn as a weighted sum of the (already-scaled) features and
   squashes that through a sigmoid to get a probability. It is the natural
   "interpretable" counterpart to Member 1's KNN baseline: every feature
   gets a coefficient, so we can directly read off which factors push a
   customer toward or away from churning (see the coefficient plot saved
   by this script).

 Why Logistic Regression needs scaled features:
   Regularization (L1/L2) penalizes large coefficients uniformly across
   features. If features were on wildly different scales (e.g. tenure in
   months vs. TotalCharges in the thousands), the penalty would unfairly
   punish features with naturally larger raw values. shared/preprocessing.py
   already standardises tenure, MonthlyCharges and TotalCharges with
   StandardScaler before saving the data, so this script can load the
   processed data directly.

 Class imbalance:
   Unlike KNeighborsClassifier, LogisticRegression has a built-in
   class_weight parameter. Setting class_weight="balanced" automatically
   re-weights the loss function inversely proportional to class frequency,
   which compensates for the ~73/27 No-Churn/Churn split without needing
   the manual threshold-tuning trick Member 1 had to use for KNN.

 Hyperparameter tuning:
   We use GridSearchCV (5-fold, stratified, scoring="f1") to search over:
     - C          : inverse regularization strength (smaller = more
                    regularization / simpler model)
     - penalty    : "l2" (shrinks all coefficients) vs "l1" (can zero
                    some out — effectively does feature selection)
     - solver     : must be compatible with the chosen penalty
                    ("lbfgs" only supports l2; "liblinear"/"saga" support
                    both l1 and l2)
   Because l1 and l2 need different solvers, the grid is expressed as a
   LIST of parameter dictionaries rather than a single dictionary.

 Model justification (for the report):
   - Cox, D. R. (1958). The regression analysis of binary sequences.
     Journal of the Royal Statistical Society, Series B, 20(2), 215-242.
     — the original statistical foundation of logistic regression.
   - Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied
     Logistic Regression (3rd ed.). Wiley. — standard reference for
     logistic regression in applied/business classification problems.
   - Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.
     Journal of Machine Learning Research, 12, 2825-2830. — the
     LogisticRegression implementation used here.

 How to run (from the project root folder, AFTER shared/preprocessing.py):
   python member2_LogisticRegression/train_logistic_regression.py
============================================================================
"""

import os
import sys

import matplotlib
matplotlib.use("Agg") # save plots to file without needing a screen
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import precision_recall_curve, f1_score

# --- Make the shared/ folder importable no matter where this is run from --
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from shared.eval_utils import load_processed_data, evaluate_and_save, RESULTS_DIR

MODEL_NAME = "LogisticRegression"

def build_param_grid():
    """
    sklearn >= 1.8 deprecated the 'penalty' string in favor of 'l1_ratio'
    (0.0 = pure L2, 1.0 = pure L1). We use l1_ratio directly here to avoid
    the FutureWarning entirely, AND we keep two SEPARATE dicts in a list
    so GridSearchCV never tries an invalid solver/l1_ratio combination
    (e.g. lbfgs only supports l1_ratio=0 / pure L2).

    IMPORTANT: keep these as two separate dict entries in the returned
    list. Do NOT merge them into one dict — that creates every possible
    cartesian combination, including invalid ones like lbfgs + l1_ratio=1,
    which will fail and waste ~25% of your grid search fits.
    """
    l2_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "l1_ratio": [0.0],          # pure L2
        "solver": ["lbfgs", "liblinear"],
    }
    l1_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "l1_ratio": [1.0],          # pure L1
        "solver": ["liblinear", "saga"],
    }
    return [l2_grid, l1_grid]

def plot_coefficients(model, feature_names, top_n=15):
    """
    Saves a horizontal bar chart of the top +/- coefficients so the report
    can discuss which features push customers toward / away from churn.
    Positive coefficient -> increases odds of churn.
    Negative coefficient -> decreases odds of churn.
    """
    coefs = model.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1][:top_n]
    top_features = [feature_names[i] for i in order][::-1]
    top_values = [coefs[i] for i in order][::-1]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top_values]
    
    plt.figure(figsize=(8,6))
    plt.barh(top_features, top_values, color=colors, edgecolor="black")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Coefficient (log-odds impact)")
    plt.title(
        f"{MODEL_NAME} — Top {top_n} Feature Coefficients\n"
        "(red = increases churn odds, blue = decreases churn odds)"
    )
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_coefficients.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[OK] Saved coefficient plot to {out_path}")
    
def main():
    # ── 1. Load the identical train/test split every member uses ────────────
    X_train, X_test, y_train, y_test = load_processed_data()
    print(
        f"Loaded processed data: {X_train.shape[0]} train rows, "
        f"{X_test.shape[0]} test rows, {X_train.shape[1]} features"
    )
    
    # ── 2. Base model ─────────────────────────────────────────────────────
    # class_weight="balanced" handles the ~73/27 imbalance automatically.
    # max_iter raised from the default 100 because with many one-hot columns
    # the solver sometimes needs more iterations to converge.
    base_model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    
    param_grid = build_param_grid()
    print(param_grid)  # sanity check — should print: [ {...l2...}, {...l1...} ]
    assert isinstance(param_grid, list) and len(param_grid) == 2
    
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    
    # ── Threshold tuning ──────────────────────────────────────────────
    # class_weight="balanced" already shifts the score distribution toward
    # catching more churners, but predict() still applies a plain 0.5 cutoff.
    # Sweep thresholds against predict_proba() and pick the one that
    # maximizes F1 directly, instead of assuming 0.5 is optimal
    
    
    # ── 3. Search for the best combination of hyperparameters ───────────────
    print("\nRunning GridSearchCV (5-fold) over C / penalty / solver ...")
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_index = grid.best_index_
    cv_std = grid.cv_results_["std_test_score"][best_index]
    print(f"\n[OK] Best params found: {best_params}")
    print(
          f"     Best CV F1 score:   {grid.best_score_:.4f}  "
          f"(+/- {cv_std:.4f} std across the 5 folds)"
    )
    
    # ── Threshold tuning (LEAK-FREE) ─────────────────────────────────────
    # class_weight="balanced" already shifts the score distribution toward
    # catching more churners, but predict() still applies a plain 0.5 cutoff.
    # To find a better cutoff WITHOUT touching the test set, we get
    # out-of-fold probabilities on the TRAINING set via 5-fold
    # cross_val_predict — each row's prediction comes from a fold-model
    # that never saw that row during training — then sweep thresholds
    # against those and pick the one that maximizes F1. X_test/y_test are
    # not touched until evaluate_and_save() at the very end.
    oof_probs = cross_val_predict(
        best_model, X_train, y_train, cv=5, method="predict_proba", n_jobs=-1
    )[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, oof_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = f1_scores[:-1].argmax()
    best_threshold = float(thresholds[best_idx])

    print(f"\n[Threshold tuning] Best threshold (from training folds only): {best_threshold:.3f}")
    print(f"[Threshold tuning] Out-of-fold F1 at that threshold: {f1_scores[best_idx]:.4f}")
    
    # ── 4. Evaluate on the held-out test set and save everything ────────────
    # evaluate_and_save() (shared/eval_utils.py) prints the metrics and saves:
    #   results/LogisticRegression_metrics.json
    #   results/LogisticRegression_confusion.png
    #   results/LogisticRegression_roc.png
    #   models/LogisticRegression.pkl
    evaluate_and_save(
        model_name=MODEL_NAME,
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        best_params=best_params,
        threshold=best_threshold,
    )
    
    # ── 5. Extra: coefficient plot (interpretability is LR's strength) ──────
    plot_coefficients(best_model, X_train.columns.tolist())
    
    print("\nNext step: once all members have run their scripts, run:")
    print("    python results/compare_models.py")
    

if __name__ == "__main__":
    main()