"""
============================================================================
 MEMBER 3  —  RANDOM FOREST
============================================================================
 Role in this project:
   Random Forest builds many decision trees on bootstrap samples of the
   training data and averages their votes (bagging). Each split also
   considers only a random subset of features, which reduces correlation
   between trees and usually improves generalisation over a single tree.

   Strengths for churn:
     - Captures non-linear interactions (e.g. Contract × Tenure)
     - class_weight="balanced" handles the minority Churn class
     - feature_importances_ support business interpretation in the app
     - Robust to scale (numeric columns are still scaled for fair comparison
       with KNN / Logistic Regression on the shared pipeline)

 Hyperparameter tuning:
   GridSearchCV (5-fold) over:
     - n_estimators      : number of trees in the forest
     - max_depth         : tree depth cap (None = grow until pure/min samples)
     - min_samples_leaf  : minimum samples in a leaf (regularisation)
     - max_features      : feature subset size at each split ("sqrt" / "log2")
   Scoring is F1.

 Model justification (cite in the report, APA 7th):
   - Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
   - Idris, A., Rizwan, M., & Khan, A. (2012). Churn prediction in telecom
     using Random Forest and PSO based data balancing. Applied Soft
     Computing, 12(8), 2435-2446. — RF applied to telecom churn.

 How to run (from the project root, AFTER shared/preprocessing.py):
   python member3_random_forest/train_random_forest.py
============================================================================
"""

import os
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from shared.eval_utils import load_processed_data, evaluate_and_save

MODEL_NAME = "RandomForest"

# Keep the grid modest so a laptop finishes in a few minutes; expand if needed.
PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_leaf": [1, 5, 10],
    "max_features": ["sqrt", "log2"],
}


def main():
    X_train, X_test, y_train, y_test = load_processed_data()
    print(
        f"Loaded processed data: {X_train.shape[0]} train rows, "
        f"{X_test.shape[0]} test rows, {X_train.shape[1]} features"
    )

    base_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
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

    # Top feature importances (useful narrative for the report)
    importances = best_model.feature_importances_
    top_idx = importances.argsort()[::-1][:10]
    print("\nTop 10 features by mean decrease in impurity:")
    for i in top_idx:
        print(f"  {X_train.columns[i]:40s}  {importances[i]:.4f}")

    evaluate_and_save(
        model_name=MODEL_NAME,
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        best_params=best_params,
    )

    print("\nNext step: train remaining models, then:")
    print("    python results/compare_models.py")


if __name__ == "__main__":
    main()
