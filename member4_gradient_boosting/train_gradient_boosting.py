"""
============================================================================
 MEMBER 4  —  HISTOGRAM GRADIENT BOOSTING
============================================================================
 Role in this project:
   Gradient Boosting builds trees sequentially: each new tree fits the
   residual errors of the previous ensemble (boosting). scikit-learn's
   HistGradientBoostingClassifier is a fast, modern implementation that
   bins continuous features (histogram-based) and usually performs very
   well on medium-sized tabular data such as this telco dataset.

   Why include it alongside Random Forest (Member 3):
     - Both are tree ensembles, so the report can compare bagging vs boosting
       within the same model family (allowed by the assignment brief).
     - Boosting often edges out bagging on structured prediction tasks, at
       the cost of more careful regularisation (learning_rate, max_depth,
       min_samples_leaf).

 Class imbalance:
   HistGradientBoostingClassifier supports class_weight="balanced" (sklearn
   >= 1.2), which reweights the loss so Churn examples matter more.

 Hyperparameter tuning:
   GridSearchCV (5-fold) over:
     - learning_rate     : step size of each boosting iteration
     - max_depth         : depth of individual trees
     - max_iter          : number of boosting iterations (trees)
     - min_samples_leaf  : leaf regularisation
   Scoring is F1.

 Model justification (cite in the report, APA 7th):
   - Friedman, J. H. (2001). Greedy function approximation: A gradient
     boosting machine. Annals of Statistics, 29(5), 1189-1232.
   - Natekin, A., & Knoll, A. (2013). Gradient boosting machines, a tutorial.
     Frontiers in Neurorobotics, 7, Article 21.

 How to run (from the project root, AFTER shared/preprocessing.py):
   python member4_gradient_boosting/train_gradient_boosting.py
============================================================================
"""

import os
import sys

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from shared.eval_utils import load_processed_data, evaluate_and_save

MODEL_NAME = "GradientBoosting"

PARAM_GRID = {
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5, 7],
    "max_iter": [100, 200],
    "min_samples_leaf": [10, 20],
}


def main():
    X_train, X_test, y_train, y_test = load_processed_data()
    print(
        f"Loaded processed data: {X_train.shape[0]} train rows, "
        f"{X_test.shape[0]} test rows, {X_train.shape[1]} features"
    )

    # Convert to float64: HistGradientBoosting prefers numeric float arrays;
    # our one-hot columns may be bool/int from get_dummies + CSV round-trip.
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    base_model = HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=42,
    )

    print(f"\nTuning {MODEL_NAME} with GridSearchCV (5-fold, F1) ...")
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_index = grid.best_index_
    cv_std = grid.cv_results_["std_test_score"][best_index]
    print(f"\n[OK] Best params: {best_params}")
    print(
        f"     Best CV F1: {grid.best_score_:.4f}  "
        f"(+/- {cv_std:.4f} std across folds)"
    )

    evaluate_and_save(
        model_name=MODEL_NAME,
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        best_params=best_params,
    )

    print("\nNext step: once all members have trained, run:")
    print("    python results/compare_models.py")


if __name__ == "__main__":
    main()
