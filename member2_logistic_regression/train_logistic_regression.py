"""
============================================================================
 MEMBER 2  —  LOGISTIC REGRESSION
============================================================================
 Role in this project:
   Logistic Regression is a linear, probabilistic classifier that estimates
   P(Churn | features) via the sigmoid of a weighted sum of inputs. It is a
   natural choice for customer churn because:
     - coefficients are interpretable (which features raise/lower churn odds);
     - it is widely used as a strong, well-understood baseline for binary
       business outcomes;
     - with class_weight="balanced" it handles the ~73/27 class imbalance
       without resampling.

   Compared with Member 1's KNN baseline (instance-based, non-parametric),
   Logistic Regression assumes a roughly linear decision boundary in feature
   space — a useful contrast for the model-selection discussion in the report.

 Hyperparameter tuning:
   GridSearchCV (5-fold) searches over:
     - C         : inverse regularisation strength (smaller = stronger)
     - l1_ratio  : 0.0 = pure L2 (ridge), 1.0 = pure L1 (lasso)
   Scoring is F1 to match results/compare_models.py.

 Model justification (cite in the report, APA 7th):
   - Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied
     logistic regression (3rd ed.). Wiley.
   - Huang, B., Kechadi, M. T., & Buckley, B. (2012). Customer churn
     prediction in telecommunications. Expert Systems with Applications,
     39(1), 1414-1425. — logistic regression among common churn models.

 How to run (from the project root, AFTER shared/preprocessing.py):
   python member2_logistic_regression/train_logistic_regression.py
============================================================================
"""

import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from shared.eval_utils import load_processed_data, evaluate_and_save

MODEL_NAME = "LogisticRegression"

# sklearn >= 1.8: use l1_ratio instead of the deprecated penalty= argument.
# l1_ratio=0 → pure L2; l1_ratio=1 → pure L1 (requires a compatible solver).
PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.0, 1.0],
}


def main():
    X_train, X_test, y_train, y_test = load_processed_data()
    print(
        f"Loaded processed data: {X_train.shape[0]} train rows, "
        f"{X_test.shape[0]} test rows, {X_train.shape[1]} features"
    )

    # class_weight="balanced" up-weights the minority Churn class so the
    # linear model does not collapse to always predicting "No Churn".
    # saga supports elastic-net / l1_ratio and works well at this data size.
    base_model = LogisticRegression(
        class_weight="balanced",
        solver="saga",
        max_iter=5000,
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

    print("\nNext step: train remaining models, then:")
    print("    python results/compare_models.py")


if __name__ == "__main__":
    main()
