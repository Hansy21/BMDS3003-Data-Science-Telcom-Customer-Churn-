"""
============================================================================
 MEMBER 3  —  DECISION TREE
============================================================================
 Role in this project:
   Decision Tree is a non-linear, rule-based classifier. It repeatedly
   splits the data on the single feature/threshold that most reduces
   impurity (Gini or entropy), producing a flowchart-like set of if/else
   rules. It sits between Member 1's KNN (no assumptions, no
   interpretability) and Member 2's Logistic Regression (linear, fully
   interpretable via coefficients): a tree captures non-linear interactions
   between features AND is still fully interpretable, since you can draw
   the exact decision path for any prediction.

 Why Decision Tree does NOT need scaled features:
   Trees split on "is feature X > threshold?", one feature at a time. The
   split point adapts to whatever scale the feature is on, so unlike KNN
   and Logistic Regression, feature scaling has no effect on a tree's
   structure or performance. It's fine that shared/preprocessing.py scales
   tenure/MonthlyCharges/TotalCharges for the other members — those scaled
   columns work here too, they just didn't need to be scaled for this model.

 Class imbalance:
   Like LogisticRegression, DecisionTreeClassifier has a built-in
   class_weight parameter. Setting class_weight="balanced" re-weights the
   impurity calculation inversely proportional to class frequency, which
   compensates for the ~73/27 No-Churn/Churn split without needing KNN's
   manual out-of-fold threshold-tuning trick.

 Hyperparameter tuning:
   We use GridSearchCV (5-fold, stratified, scoring="f1") to search over:
     - criterion         : "gini" vs "entropy" (how impurity is measured)
     - max_depth         : caps tree depth — the main lever against
                            overfitting (an unconstrained tree will grow
                            until every leaf is pure, memorising noise)
     - min_samples_split : minimum samples required to split a node
     - min_samples_leaf  : minimum samples required in a leaf node
     - ccp_alpha         : cost-complexity pruning strength (0 = no
                            pruning; larger values prune more aggressively)

 Model justification (for the report):
   - Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984).
     Classification and Regression Trees. Wadsworth. — the original CART
     algorithm that DecisionTreeClassifier implements.
   - Quinlan, J. R. (1986). Induction of decision trees. Machine Learning,
     1(1), 81-106. — foundational entropy/information-gain based tree
     induction, relevant to the "entropy" criterion option.
   - Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.
     Journal of Machine Learning Research, 12, 2825-2830. — the
     DecisionTreeClassifier implementation used here.

 How to run (from the project root folder, AFTER shared/preprocessing.py):
   python member3_DecisionTree/DecisionTree.py
============================================================================
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # save plots to file without needing a screen
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import precision_recall_curve, f1_score

# --- Make the shared/ folder importable no matter where this is run from --
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from shared.eval_utils import load_processed_data, evaluate_and_save, RESULTS_DIR

MODEL_NAME = "DecisionTree"


def build_param_grid():
    """
    A moderate grid: wide enough to find a good bias/variance trade-off,
    small enough to run in reasonable time with 5-fold CV. max_depth=None
    is included deliberately so GridSearchCV can compare "unconstrained"
    against constrained trees and show, empirically, that the unconstrained
    one overfits (useful evidence for your report).
    """
    return {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7, 10, 15, None],
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [1, 5, 10],
        "ccp_alpha": [0.0, 0.001, 0.005],
    }


def plot_feature_importances(model, feature_names, top_n=15):
    """
    Saves a horizontal bar chart of the top feature importances (Gini
    importance / mean impurity decrease). Unlike Logistic Regression's
    coefficients, these are always >= 0 and don't indicate direction
    (increases vs decreases churn) — only how much a feature contributed
    to reducing impurity across the tree.
    """
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in order][::-1]
    top_values = [importances[i] for i in order][::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(top_features, top_values, color="#2ca02c", edgecolor="black")
    plt.xlabel("Feature importance (mean impurity decrease)")
    plt.title(f"{MODEL_NAME} — Top {top_n} Feature Importances")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_feature_importances.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[OK] Saved feature importance plot to {out_path}")


def plot_tree_diagram(model, feature_names, max_depth_to_show=3):
    """
    Saves a readable diagram of the top few levels of the tree (the full
    tree is usually too deep/wide to read). This is the visual explanation
    of the model's logic that KNN and, to a lesser extent, Logistic
    Regression can't offer as directly.
    """
    plt.figure(figsize=(16, 10))
    plot_tree(
        model,
        max_depth=max_depth_to_show,
        feature_names=feature_names,
        class_names=["No Churn", "Churn"],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title(
        f"{MODEL_NAME} — Decision Logic "
        f"(showing top {max_depth_to_show} levels only)"
    )
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_tree_diagram.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[OK] Saved tree diagram to {out_path}")


def main():
    # ── 1. Load the identical train/test split every member uses ────────────
    X_train, X_test, y_train, y_test = load_processed_data()
    print(
        f"Loaded processed data: {X_train.shape[0]} train rows, "
        f"{X_test.shape[0]} test rows, {X_train.shape[1]} features"
    )

    # ── 2. Base model ─────────────────────────────────────────────────────
    # class_weight="balanced" handles the ~73/27 imbalance automatically,
    # same as LogisticRegression — no manual threshold trick needed for
    # class balance itself (we still tune the threshold for F1 below).
    base_model = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
    )

    param_grid = build_param_grid()

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    # ── 3. Search for the best combination of hyperparameters ───────────────
    print(
        "\nRunning GridSearchCV (5-fold) over criterion / max_depth / "
        "min_samples_split / min_samples_leaf / ccp_alpha ..."
    )
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
    print(
        f"     Tree depth actually used: {best_model.get_depth()}, "
        f"leaves: {best_model.get_n_leaves()}"
    )

    # ── Threshold tuning (same approach as LogisticRegression.py) ───────────
    # class_weight="balanced" shifts the score distribution, but predict()
    # still applies a plain 0.5 cutoff. Sweep thresholds against
    # predict_proba() and pick the one that maximizes F1 directly.
    # ── Threshold tuning (LEAK-FREE) ─────────────────────────────────────
    # class_weight="balanced" shifts the score distribution, but predict()
    # still applies a plain 0.5 cutoff. To find a better cutoff WITHOUT
    # touching the test set, get out-of-fold probabilities on the TRAINING
    # set via 5-fold cross_val_predict (each row's prediction comes from a
    # fold-model that never saw that row), sweep thresholds against those,
    # and pick the one that maximizes F1. X_test/y_test stay untouched
    # until evaluate_and_save() at the very end.
    oof_probs = cross_val_predict(
        best_model, X_train, y_train, cv=5, method="predict_proba", n_jobs=-1
    )[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, oof_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = f1_scores[:-1].argmax()
    best_threshold = float(thresholds[best_idx])

    print(
        f"\n[Threshold tuning] Best threshold (from training folds only): {best_threshold:.3f}"
    )
    print(
        f"[Threshold tuning] Out-of-fold F1 at that threshold: {f1_scores[best_idx]:.4f}"
    )

    # ── 4. Evaluate on the held-out test set and save everything ────────────
    # evaluate_and_save() (shared/eval_utils.py) prints the metrics and saves:
    #   results/DecisionTree_metrics.json
    #   results/DecisionTree_confusion.png
    #   results/DecisionTree_roc.png
    #   models/DecisionTree.pkl
    evaluate_and_save(
        model_name=MODEL_NAME,
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        best_params=best_params,
        threshold=best_threshold,
    )

    # ── 5. Extra: feature importances + tree diagram (interpretability) ─────
    plot_feature_importances(best_model, X_train.columns.tolist())
    plot_tree_diagram(best_model, X_train.columns.tolist())

    print("\nNext step: once all members have run their scripts, run:")
    print("    python results/compare_models.py")


if __name__ == "__main__":
    main()
