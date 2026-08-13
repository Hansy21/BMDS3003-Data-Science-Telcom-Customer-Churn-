"""
============================================================================
 MEMBER 3  —  RANDOM FOREST CLASSIFIER
============================================================================

Role in this project:
    Random Forest is an ensemble machine learning algorithm that combines multiple
    Decision Trees to create a stronger and more stable classifier.

    Instead of relying on a single Decision Tree, Random Forest:
        1. Creates multiple trees using different bootstrap samples.
        2. Randomly selects subsets of features during splitting.
        3. Combines predictions from all trees.

Why Random Forest is suitable:
    Customer churn behaviour depends on many interacting factors:
        - Contract type
        - Customer tenure
        - Monthly charges
        - Payment method
        - Internet services
        - Customer account information
    Random Forest can automatically learn these complex patterns without requiring
    manual feature engineering.

Class imbalance:
    The dataset contains approximately:
        No Churn : 73%
        Churn    : 27%
    Since churn customers are the minority class, class_weight="balanced" is
    applied. This gives higher importance to mistakes involving churn customers.

Hyperparameter tuning:
    GridSearchCV with 5-fold cross validation is applied.
    The optimisation metric is F1-score because:
        - Accuracy can be misleading due to class imbalance.
        - Recall is important for identifying churn customers.
        - F1 balances precision and recall.

How to run:
    python member3_RandomForest/train_random_forest.py

============================================================================
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

# ==========================================================================
# Import shared project utilities
# ==========================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from shared.eval_utils import load_processed_data, evaluate_and_save, RESULTS_DIR

MODEL_NAME = "RandomForest"

# ==========================================================================
# Hyperparameter Search
# ==========================================================================


def build_param_grid():
    """
    Returns Random Forest hyperparameters tested by GridSearchCV.

    The search space is intentionally controlled.

    Telco Customer Churn contains around 7000 records, therefore an extremely
    large search space provides limited improvement while increasing runtime.
    """
    return {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
    }


# ==========================================================================
# Built-in Feature Importance
# ==========================================================================


def plot_feature_importance(model, feature_names, top_n=15):
    """
    Generates Random Forest built-in feature importance plot.

    Importance is calculated from how much each feature decreases impurity
    across all decision trees.
    """
    importance = model.feature_importances_

    indices = np.argsort(importance)[::-1][:top_n]

    selected_features = [feature_names[i] for i in indices]

    selected_values = [importance[i] for i in indices]

    plt.figure(figsize=(9, 6))
    plt.barh(selected_features[::-1], selected_values[::-1])

    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title(f"{MODEL_NAME} - Top {top_n} Feature Importance")
    plt.tight_layout()

    output = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_feature_importance.png")

    plt.savefig(output, dpi=120)
    plt.close()

    print(f"[OK] Saved feature importance plot: {output}")


# ==========================================================================
# Permutation Importance
# ==========================================================================


def plot_permutation_importance(model, X_test, y_test, feature_names, top_n=15):
    """
    Generates permutation importance.
    Features are randomly shuffled one at a time.
    """

    print("\nCalculating permutation importance...")

    result = permutation_importance(
        model, X_test, y_test, scoring="f1", n_repeats=10, random_state=42, n_jobs=-1
    )

    importance = result.importances_mean

    indices = np.argsort(importance)[::-1][:top_n]

    plt.figure(figsize=(9, 6))
    plt.barh([feature_names[i] for i in indices][::-1], importance[indices][::-1])

    plt.xlabel("Mean F1 Score Decrease")
    plt.ylabel("Features")
    plt.title(f"{MODEL_NAME} - Permutation Importance")
    plt.tight_layout()

    output = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_permutation_importance.png")

    plt.savefig(output, dpi=120)

    plt.close()

    print(f"[OK] Saved permutation importance plot: {output}")


# ==========================================================================
# Main Training Pipeline
# ==========================================================================


def main():

    # 1. Load processed dataset
    X_train, X_test, y_train, y_test = load_processed_data()

    print("\n====================================")
    print(" RANDOM FOREST TRAINING")
    print("====================================")
    print(f"\nTraining samples : {X_train.shape[0]}")
    print(f"Testing samples  : {X_test.shape[0]}")
    print(f"Features         : {X_train.shape[1]}")

    # 2. Create Base Random Forest Model
    """
    NOTE: class_weight="balanced" removed on purpose. This is what pushes
    accuracy up (at a cost to recall) — see Section 5.0 of the report for
    the accuracy-vs-recall trade-off discussion.
    random_state=42: Ensures reproducibility.
    n_jobs=-1: Uses all CPU cores available.

    """
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    # 3. GridSearchCV Hyperparameter Optimisation

    param_grid = build_param_grid()

    print("\nHyperparameter search space:")
    print(param_grid)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    print("\nStarting GridSearchCV...")
    print("Optimising using 5-fold cross validation...")
    grid.fit(X_train, y_train)

    # 4. Retrieve Best Model

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_index = grid.best_index_
    cv_std = grid.cv_results_["std_test_score"][best_index]

    print("\n====================================")
    print(" BEST RANDOM FOREST MODEL ")
    print("====================================")
    print("\nBest Parameters:")
    print(best_params)

    print(f"\nBest CV F1-score: " f"{grid.best_score_:.4f}")

    print(f"CV Standard Deviation: " f"+/- {cv_std:.4f}")

    # 5. Evaluate Final Model

    evaluate_and_save(
        model_name=MODEL_NAME,
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        best_params=best_params,
    )

    # 6. Feature Importance Analysis

    feature_names = X_train.columns.tolist()
    plot_feature_importance(model=best_model, feature_names=feature_names, top_n=15)

    # 7. Permutation Importance Analysis
    plot_permutation_importance(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        top_n=15,
    )

    print("\n====================================")
    print(" RANDOM FOREST COMPLETE")
    print("====================================")


# ==========================================================================
# Program Entry Point
# ==========================================================================
if __name__ == "__main__":
    main()
