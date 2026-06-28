"""
============================================================================
 MEMBER 4  —  GRADIENT BOOSTING
============================================================================
 Gradient Boosting also builds many trees, but BUILDS THEM ONE AT A TIME,
 where each new tree focuses on the mistakes of the previous ones. This often
 gives the best accuracy of all the models, at the cost of slower training
 and more tuning.

 HOW TO RUN (from the project root folder):
   python shared/preprocessing.py                              # once (if not done)
   python member4_gradient_boosting/train_gradient_boosting.py

 YOUR JOB (Member 4):
   - Explain boosting vs bagging (how it differs from Random Forest).
   - Tune learning_rate + n_estimators (they trade off against each other).
   - Compare against all other members; discuss if the extra complexity is
     worth it for this dataset.

 NOTE: GradientBoostingClassifier has NO class_weight option. To handle the
 73/27 imbalance we pass sample_weight during fitting instead.
============================================================================
"""

import os
import sys

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.eval_utils import load_processed_data, evaluate_and_save

MODEL_NAME = "GradientBoosting"

PARAM_DIST = {
    "n_estimators": [100, 200, 300],       # number of boosting stages (trees)
    "learning_rate": [0.01, 0.05, 0.1],    # how much each tree contributes
    "max_depth": [2, 3, 4],                # depth of each small tree
    "subsample": [0.8, 1.0],               # fraction of rows per tree
}


def main():
    X_train, X_test, y_train, y_test = load_processed_data()

    # Weight each training row so the rarer "churn" class counts more,
    # since GradientBoosting has no class_weight parameter.
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    base_model = GradientBoostingClassifier(random_state=42)

    print(f"Tuning {MODEL_NAME} ...")
    search = RandomizedSearchCV(
        base_model, PARAM_DIST, n_iter=20, scoring="f1",
        cv=5, n_jobs=-1, random_state=42, verbose=1,
    )
    # Pass the weights through to every internal fit during the search
    search.fit(X_train, y_train, sample_weight=sample_weight)
    best_model = search.best_estimator_

    evaluate_and_save(
        MODEL_NAME, best_model, X_test, y_test, best_params=search.best_params_
    )


if __name__ == "__main__":
    main()
