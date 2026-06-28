"""
============================================================================
 MEMBER 1  —  LOGISTIC REGRESSION   *** BASELINE MODEL ***
============================================================================
 The assignment requires ONE model to be the baseline. Logistic Regression
 is the natural choice: it is the simplest, most interpretable classifier,
 so every other model (Decision Tree, Random Forest, Gradient Boosting) is
 judged by whether it BEATS this baseline.

 What this script does:
   1. Loads the shared train/test data (same for all members)
   2. Tunes the model with GridSearchCV (tries several settings, keeps best)
   3. Evaluates it and saves metrics + plots + the trained model

 HOW TO RUN (from the project root folder):
   python shared/preprocessing.py                                   # once
   python member1_logistic_regression/train_logistic_regression.py

 YOUR JOB (Member 1):
   - Understand WHY logistic regression works as a baseline (linear model).
   - Try adding/removing values in PARAM_GRID below and see what changes.
   - Write up the results for the report (Model Selection + Evaluation).
============================================================================
"""

import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Let this script import the shared helpers from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.eval_utils import load_processed_data, evaluate_and_save

MODEL_NAME = "LogisticRegression"

# Settings GridSearchCV will try. It tests every value with 5-fold
# cross-validation and keeps the one with the best F1 score.
# We tune C (regularisation strength): smaller C = simpler, more regularised.
PARAM_GRID = {
    "C": [0.01, 0.1, 0.5, 1, 5, 10],
}


def main():
    # ── 1. Load the SAME data every member uses ─────────────────────────────
    X_train, X_test, y_train, y_test = load_processed_data()

    # ── 2. Base model. class_weight="balanced" handles the 73/27 imbalance ──
    base_model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )

    # ── 3. Tune with GridSearchCV (scoring on F1 = balance of precision/recall)
    print(f"Tuning {MODEL_NAME} ...")
    grid = GridSearchCV(
        base_model, PARAM_GRID, scoring="f1", cv=5, n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # ── 4. Evaluate + save everything (shared helper does the work) ─────────
    evaluate_and_save(
        MODEL_NAME, best_model, X_test, y_test, best_params=grid.best_params_
    )


if __name__ == "__main__":
    main()
