"""
============================================================================
 MEMBER 3  —  RANDOM FOREST
============================================================================
 A Random Forest builds MANY decision trees on random subsets of the data
 and averages their votes. This usually beats a single tree because the
 errors of individual trees cancel out (an "ensemble"). It is more accurate
 but harder to interpret than one tree.

 HOW TO RUN (from the project root folder):
   python shared/preprocessing.py                          # once (if not done)
   python member3_random_forest/train_random_forest.py

 YOUR JOB (Member 3):
   - Explain bagging / ensembles for the report.
   - Show a feature-importance chart (Random Forest gives this for free).
   - Tune PARAM_GRID and compare against the baseline + decision tree.

 NOTE: RandomizedSearchCV is used here instead of GridSearchCV because the
 grid is large; randomised search samples combinations and runs much faster.
============================================================================
"""

import os
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.eval_utils import load_processed_data, evaluate_and_save

MODEL_NAME = "RandomForest"

PARAM_DIST = {
    "n_estimators": [100, 200, 300],       # number of trees in the forest
    "max_depth": [5, 10, 15, None],        # max depth of each tree
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],      # features considered per split
}


def main():
    X_train, X_test, y_train, y_test = load_processed_data()

    base_model = RandomForestClassifier(
        class_weight="balanced", random_state=42, n_jobs=-1
    )

    print(f"Tuning {MODEL_NAME} ...")
    # n_iter=25 -> tries 25 random combinations (raise it for a wider search)
    search = RandomizedSearchCV(
        base_model, PARAM_DIST, n_iter=25, scoring="f1",
        cv=5, n_jobs=-1, random_state=42, verbose=1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    evaluate_and_save(
        MODEL_NAME, best_model, X_test, y_test, best_params=search.best_params_
    )


if __name__ == "__main__":
    main()
