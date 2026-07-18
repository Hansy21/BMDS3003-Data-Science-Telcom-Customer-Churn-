"""
============================================================================
 MEMBER 1  —  K-NEAREST NEIGHBORS (KNN)  —  BASELINE MODEL
============================================================================
 Role in this project:
   KNN is used as the BASELINE model for churn prediction. It makes no
   assumptions about the shape of the decision boundary (unlike Logistic
   Regression) and does not need to be "trained" in the usual sense — it
   simply memorises the training data and, at prediction time, looks at the
   'k' closest customers (by Euclidean/Manhattan distance) and votes on the
   majority class. This makes it a simple, transparent point of comparison
   for the more complex models (Random Forest, Gradient Boosting, etc.)
   built by the other members.

 Why KNN needs scaled features:
   KNN is a distance-based algorithm, so features on a larger numeric scale
   (e.g. TotalCharges, which can be in the thousands) would dominate the
   distance calculation if left unscaled. shared/preprocessing.py already
   standardises tenure, MonthlyCharges and TotalCharges with StandardScaler
   before the data is saved, so this script can load the processed data
   directly without repeating that work.

 Hyperparameter tuning:
   We use GridSearchCV (5-fold, stratified) to search over:
     - n_neighbors : how many neighbours vote on the prediction
     - weights     : 'uniform' (all neighbours count equally) vs
                     'distance' (closer neighbours count more)
     - p           : 1 = Manhattan distance, 2 = Euclidean distance
   Scoring is by F1, matching the project-wide convention in
   results/compare_models.py (churn is imbalanced, so F1 balances
   precision and recall better than plain accuracy).

 Class imbalance & the decision threshold (recall-focused improvement):
   The training data is ~73% "No Churn" / 27% "Churn". KNeighborsClassifier
   has no class_weight parameter (unlike DecisionTree/RandomForest/Logistic
   Regression), so it has no built-in way to compensate for that imbalance.
   Left at the default 0.5 cut-off, it tends to under-predict the minority
   "Churn" class — recall suffers even though accuracy still looks fine.
   Since this project explicitly cares about *catching real churners*
   (see eval_utils.py), we tune the probability threshold instead:
     1. Get out-of-fold predicted probabilities for the TRAINING set only,
        via cross_val_predict (5-fold) — this never touches X_test, so
        there is no leakage.
     2. Walk the precision/recall curve on those out-of-fold probabilities
        and pick the lowest threshold that still reaches a target recall
        (RECALL_TARGET below), preferring the option with the best
        precision among those that qualify.
     3. Apply that single, fixed threshold to the real test set.
   This is a standard alternative to resampling (SMOTE etc.) for handling
   imbalance with a model that can't take class weights directly, and it
   is fully reproducible since the threshold is chosen from training data.

 Model justification (see report for full discussion / citations):
   - Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification.
     IEEE Transactions on Information Theory, 13(1), 21-27. — the original
     theoretical basis for KNN and why it is a sound, low-bias baseline.
   - Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.
     Journal of Machine Learning Research, 12, 2825-2830. — the KNN
     implementation (KNeighborsClassifier) used in this script.
   Add these (and any churn-specific KNN papers you find) to the report's
   reference list in APA 7th edition format.

 How to run (from the project root folder, AFTER shared/preprocessing.py):
   python member1_KNN/KNN.py
============================================================================
"""

import os
import sys

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

# --- Make the shared/ folder importable no matter where this is run from --
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from shared.eval_utils import load_processed_data, evaluate_and_save

MODEL_NAME = "KNN"

# Minimum recall we want to guarantee when we pick the decision threshold.
# 0.70 means: "catch at least 70% of the customers who actually churn."
# Tune this number based on the business cost of a missed churner vs. a
# false alarm — raising it trades away some precision for more recall.
RECALL_TARGET = 0.70


def main():
    # ── 1. Load the identical train/test split every member uses ────────────
    X_train, X_test, y_train, y_test = load_processed_data()
    print(
        f"Loaded processed data: {X_train.shape[0]} train rows, "
        f"{X_test.shape[0]} test rows, {X_train.shape[1]} features"
    )

    # ── 2. Hyperparameter grid ───────────────────────────────────────────────
    # Odd values for n_neighbors avoid tie votes in binary classification.
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
        "weights": ["uniform", "distance"],
        "p": [1, 2],  # 1 = Manhattan, 2 = Euclidean
    }

    base_model = KNeighborsClassifier(n_jobs=-1)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    # ── 3. Search for the best combination of hyperparameters ───────────────
    print("\nRunning GridSearchCV (5-fold) over n_neighbors / weights / p ...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_index = grid.best_index_
    cv_std = grid.cv_results_["std_test_score"][best_index]
    print(f"\n[OK] Best params found: {best_params}")
    print(
        f"     Best CV F1 score:   {grid.best_score_:.4f}  (+/- {cv_std:.4f} std across the 5 folds)"
    )
    print(
        "     A small std means the score is stable across folds; a large "
        "one means performance depends heavily on which rows landed in "
        "which fold, so treat the number with more caution."
    )

    # ── 4. Tune the decision threshold to protect recall ─────────────────────
    # KNN can't use class_weight="balanced" like the tree models can, so we
    # compensate here instead of at 0.5. Everything below only ever looks at
    # X_train/y_train (via 5-fold out-of-fold predictions) — X_test is still
    # completely untouched until the very last line.
    print(
        f"\nTuning decision threshold for recall >= {RECALL_TARGET:.0%} "
        "using out-of-fold CV probabilities on the TRAINING set..."
    )
    oof_probs = cross_val_predict(
        best_model, X_train, y_train, cv=5, method="predict_proba", n_jobs=-1
    )[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, oof_probs)
    # precision_recall_curve returns one more precision/recall point than
    # thresholds (it appends the (1.0, 0.0) endpoint), so align them first.
    precision, recall = precision[:-1], recall[:-1]

    qualifying = np.where(recall >= RECALL_TARGET)[0]
    if len(qualifying) > 0:
        # Among thresholds that hit the recall target, take the one with
        # the best precision (i.e. fewest false alarms for that recall).
        best_i = qualifying[np.argmax(precision[qualifying])]
        chosen_threshold = float(thresholds[best_i])
        print(
            f"[OK] Chosen threshold: {chosen_threshold:.3f}  "
            f"(out-of-fold recall={recall[best_i]:.3f}, precision={precision[best_i]:.3f})"
        )
    else:
        # Recall target isn't reachable even at the most lenient threshold
        # tried (rare, but possible) — fall back to the default 0.5 cut.
        chosen_threshold = None
        print(
            f"[WARN] Could not reach {RECALL_TARGET:.0%} recall at any "
            "threshold on the training folds — keeping the default 0.5 cut."
        )

    # ── 5. Evaluate on the held-out test set and save everything ────────────
    # evaluate_and_save() (shared/eval_utils.py) prints the metrics and saves:
    #   results/KNN_metrics.json
    #   results/KNN_confusion.png
    #   results/KNN_roc.png
    #   models/KNN.pkl
    evaluate_and_save(
        model_name=MODEL_NAME,
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        best_params=best_params,
        threshold=chosen_threshold,
    )

    print("\nNext step: once all members have run their scripts, run:")
    print("    python results/compare_models.py")


if __name__ == "__main__":
    main()
