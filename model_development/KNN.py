"""Train and evaluate the K-Nearest Neighbours (KNN) churn model.

KNN predicts whether a customer may churn by finding customers with similar
details in the training data. The nearest customers "vote": if most of them
churned, KNN predicts churn for the new customer. This is a simple baseline
model, so it gives us a clear starting point for comparing more complex models.

The prepared data already has its main number-based columns scaled. This is
important because KNN uses distance: without scaling, a large value such as
total charges could matter more than every other customer detail.

The script tests several KNN settings, evaluates the best version on data it
has not seen before, and saves the scores, charts, and trained model.

Run this after preprocessing with: python model_development/KNN.py
"""

import os
import sys

import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from functools import partial

# Find the project folder, then allow this file to use the shared helper
# functions. This means the script works even when it is run from another folder.
HERE = os.path.dirname(os.path.abspath(__file__))
# Move one level up from model_development to reach the main project folder.
PROJECT_ROOT = os.path.dirname(HERE)
# Add that folder to Python's search path before importing from shared/.
sys.path.insert(0, PROJECT_ROOT)

from shared.eval_utils import load_processed_data, evaluate_and_save

MODEL_NAME = "KNN"

# Recall measures how many real churners the model successfully finds.
# Here, 0.70 means a target of finding at least 70 out of every 100 customers
# who actually churn. Raising the target may find more churners but can also
# label more customers as churn risks when they would stay.
RECALL_TARGET = 0.70


def main():
    # 1. Load the shared training and test data.
    # Training data is used to build the model. Test data is kept aside until
    # the end, so it gives a more honest check of real-world performance.
    X_train, X_test, y_train, y_test = load_processed_data()
    # shape[0] is the number of customers; shape[1] is the number of columns.
    print(
        f"Loaded processed data: {X_train.shape[0]} train rows, "
        f"{X_test.shape[0]} test rows, {X_train.shape[1]} features"
    )

    # 2. Build the model steps and the settings to test.
    # First, SelectKBest keeps the columns that appear most useful for predicting
    # churn. Then KNN uses the remaining columns to find similar customers.
    # Keeping both steps together in one pipeline makes sure feature selection
    # is learned from training data only during each cross-validation round.
    # Odd neighbour counts help avoid a tied vote between churn and no churn.
    pipe = Pipeline(
        [
            # mutual_info_classif gives each column a usefulness score for churn.
            ("select", SelectKBest(partial(mutual_info_classif, random_state=42))),
            # n_jobs=-1 lets KNN use all available processor cores.
            ("knn", KNeighborsClassifier(n_jobs=-1)),
        ]
    )

    # The names use step__setting because these settings belong to steps inside
    # the pipeline. GridSearchCV will try every possible combination below.
    param_grid = {
        # Try keeping different numbers of the most useful customer details.
        "select__k": [10, 15, 20, 25, "all"],
        # Try different numbers of similar customers voting on each prediction.
        "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31, 41, 51],
        # Uniform gives every neighbour one vote; distance gives nearer
        # neighbours more influence because they are more similar.
        "knn__weights": ["uniform", "distance"],
        # p=1 counts the total step-by-step difference; p=2 measures straight-line
        # distance. We test both to see which definition of "similar" works best.
        "knn__p": [1, 2],
    }

    # Use five-fold cross-validation to compare settings fairly. The training
    # data is split into five parts; each part is used once for checking while
    # the other four parts are used for learning.
    # ROC-AUC checks whether people with higher predicted risk are generally
    # the people who really churn, without fixing one decision point too early.
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    # 3. Find the best combination of settings.
    # GridSearchCV tries every listed combination and keeps the one with the
    # strongest average ROC-AUC score across the five validation rounds.
    print(
        "\nRunning GridSearchCV (5-fold) over select__k / n_neighbors / "
        "weights / p (scoring = ROC-AUC) ..."
    )
    grid.fit(X_train, y_train)

    # best_estimator_ is the finished pipeline with the best settings found.
    best_model = grid.best_estimator_
    # Keep the winning settings for the report and results file.
    best_params = grid.best_params_
    best_index = grid.best_index_
    # Standard deviation shows how much the score changed across the five tests.
    cv_std = grid.cv_results_["std_test_score"][best_index]
    print(f"\n[OK] Best params found: {best_params}")
    print(
        f"     Best CV ROC-AUC:  {grid.best_score_:.4f}  (+/- {cv_std:.4f} std across the 5 folds)"
    )
    print(
        "     A small std means the score is stable across folds; a large "
        "one means performance depends heavily on which rows landed in "
        "which fold, so treat the number with more caution."
    )

    # Show the columns kept by feature selection. This is useful when explaining
    # which customer details KNN considered most helpful.
    select_step = best_model.named_steps["select"]
    # get_support() returns True for each column that was kept.
    if select_step.k != "all":
        kept = X_train.columns[select_step.get_support()]
        dropped = X_train.columns[~select_step.get_support()]
        print(f"\n     Features kept ({len(kept)}): {list(kept)}")
        print(f"     Features dropped ({len(dropped)}): {list(dropped)}")

    # 4. Use KNN's normal 0.5 decision point for the final prediction.
    # A churn probability of 0.5 or above is treated as "Churn". The check
    # below still shows the effect of other decision points for the report,
    # but it does not change the final model's decision point.
    print(
        "\nComputing out-of-fold CV probabilities on the TRAINING set "
        "(for the sensitivity table below; the model itself uses the "
        "default 0.5 threshold)..."
    )
    oof_probs = cross_val_predict(
        best_model, X_train, y_train, cv=5, method="predict_proba", n_jobs=-1
    )[:, 1]
    # Out-of-fold means every customer is predicted by a model that did not use
    # that customer's row for learning. [:, 1] keeps the probability of churn.
    precision, recall, thresholds = precision_recall_curve(y_train, oof_probs)
    # The curve produces one extra end point without a matching threshold,
    # so remove it before comparing the three lists.
    precision, recall = precision[:-1], recall[:-1]

    chosen_threshold = None

    # 4b. Compare several recall targets for the report.
    # For each target, choose the threshold with the best precision while still
    # meeting that recall target. Precision means: of the customers predicted
    # to churn, how many really churned.
    print("\nRecall-target sensitivity check (for report justification):")
    print(
        f"{'target':>8}{'threshold':>12}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>8}"
    )
    for target in [0.60, 0.65, 0.70, 0.75, 0.80]:
        # Find the thresholds that reach the current recall target.
        qual = np.where(recall >= target)[0]
        if len(qual) == 0:
            continue
        idx = qual[np.argmax(precision[qual])]
        # Use the qualifying threshold with the best precision.
        t = float(thresholds[idx])
        # Apply that temporary threshold to the test set for the comparison table.
        test_probs = best_model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= t).astype(int)
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        print(
            f"{target:>8.2f}{t:>12.3f}"
            f"{accuracy_score(y_test, test_preds):>10.3f}"
            f"{precision_score(y_test, test_preds):>11.3f}"
            f"{recall_score(y_test, test_preds):>9.3f}"
            f"{f1_score(y_test, test_preds):>8.3f}"
        )

    # 5. Test the final model on unseen data and save the results.
    # The helper prints the main scores, creates charts, and saves the model so
    # it can later be compared with the other models or used in the app.
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
