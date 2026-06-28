"""
============================================================================
 MEMBER 2  —  DECISION TREE
============================================================================
 A Decision Tree splits customers with a series of yes/no questions
 (e.g. "Is the contract month-to-month?"). It is easy to explain and shows
 which features matter most, but a single tree can overfit, which is why we
 limit its depth and tune it.

 HOW TO RUN (from the project root folder):
   python shared/preprocessing.py                          # once (if not done)
   python member2_decision_tree/train_decision_tree.py

 YOUR JOB (Member 2):
   - Explain how a tree splits data (Gini / entropy) for the report.
   - Tune PARAM_GRID and report how depth affects over/underfitting.
   - Compare your result against Member 1's baseline.
============================================================================
"""

import os
import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.eval_utils import load_processed_data, evaluate_and_save

MODEL_NAME = "DecisionTree"

PARAM_GRID = {
    "max_depth": [3, 5, 7, 10, None],      # how deep the tree can grow
    "min_samples_split": [2, 10, 20],      # min samples needed to split a node
    "min_samples_leaf": [1, 5, 10],        # min samples allowed in a leaf
    "criterion": ["gini", "entropy"],      # how a split's quality is measured
}


def main():
    X_train, X_test, y_train, y_test = load_processed_data()

    base_model = DecisionTreeClassifier(
        class_weight="balanced", random_state=42
    )

    print(f"Tuning {MODEL_NAME} ...")
    grid = GridSearchCV(
        base_model, PARAM_GRID, scoring="f1", cv=5, n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    evaluate_and_save(
        MODEL_NAME, best_model, X_test, y_test, best_params=grid.best_params_
    )


if __name__ == "__main__":
    main()
