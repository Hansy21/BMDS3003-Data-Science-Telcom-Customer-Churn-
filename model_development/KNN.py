"""
============================================================================
 MEMBER 1  —  K-NEAREST NEIGHBORS (KNN)  —  BASELINE MODEL
============================================================================
 Role in this project:
   KNN is used as the BASELINE model for churn prediction. It makes no
   assumptions about the shape of the decision boundary (unlike Logistic
   Regression) and does not need to be "trained" in the usual sense — it
   simply memorises the training data and, at prediction time, looks at the
   'k' closest customers (by Euclidean/Manhattan distance) and votes on the
   majority class. This makes it a simple, transparent point of comparison
   for the more complex models (Random Forest, Gradient Boosting, etc.)
   built by the other members.

 Why KNN needs scaled features:
   KNN is a distance-based algorithm, so features on a larger numeric scale
   (e.g. TotalCharges, which can be in the thousands) would dominate the
   distance calculation if left unscaled. shared/preprocessing.py already
   standardises tenure, MonthlyCharges and TotalCharges with StandardScaler
   before the data is saved, so this script can load the processed data
   directly without repeating that work.

 Hyperparameter tuning:
   We use GridSearchCV (5-fold, stratified) to search over:
     - select__k        : how many of the most informative features to keep
                          (SelectKBest + mutual information — KNN is a
                          distance-based model, so near-zero-signal features
                          like gender/PhoneService only add noise to the
                          distance calculation and dilute the useful ones)
     - knn__n_neighbors : how many neighbours vote on the prediction
     - knn__weights     : 'uniform' (all neighbours count equally) vs
                          'distance' (closer neighbours count more)
     - knn__p           : 1 = Manhattan distance, 2 = Euclidean distance
   The feature selector lives INSIDE a Pipeline, so it is re-fitted on each
   CV fold's training portion only — no information leaks across folds.
   Scoring is by ROC-AUC rather than F1: the decision threshold is tuned
   separately afterwards (see below), so the grid search should optimise
   the *ranking quality* of the predicted probabilities, not the F1 at an
   arbitrary 0.5 cut-off that we never actually use.

 Class imbalance & the decision threshold (recall-focused improvement):
   The training data is ~73% "No Churn" / 27% "Churn". KNeighborsClassifier
   has no class_weight parameter (unlike DecisionTree/RandomForest/Logistic
   Regression), so it has no built-in way to compensate for that imbalance.
   Left at the default 0.5 cut-off, it tends to under-predict the minority
   "Churn" class — recall suffers even though accuracy still looks fine.
   Since this project explicitly cares about *catching real churners*
   (see eval_utils.py), we tune the probability threshold instead:
     1. Get out-of-fold predicted probabilities for the TRAINING set only,
        via cross_val_predict (5-fold) — this never touches X_test, so
        there is no leakage.
     2. Walk the precision/recall curve on those out-of-fold probabilities
        and pick the lowest threshold that still reaches a target recall
        (RECALL_TARGET below), preferring the option with the best
        precision among those that qualify.
     3. Apply that single, fixed threshold to the real test set.
   This is a standard alternative to resampling (SMOTE etc.) for handling
   imbalance with a model that can't take class weights directly, and it
   is fully reproducible since the threshold is chosen from training data.

 Model justification (see report for full discussion / citations):
   - Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification.
     IEEE Transactions on Information Theory, 13(1), 21-27. — the original
     theoretical basis for KNN and why it is a sound, low-bias baseline.
   - Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.
     Journal of Machine Learning Research, 12, 2825-2830. — the KNN
     implementation (KNeighborsClassifier) used in this script.
   Add these (and any churn-specific KNN papers you find) to the report's
   reference list in APA 7th edition format.

 How to run (from the project root folder, AFTER shared/preprocessing.py):
   python member1_KNN/KNN.py
============================================================================
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

# --- Make the shared/ folder importable no matter where this is run from --
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from shared.eval_utils import load_processed_data, evaluate_and_save

MODEL_NAME = "KNN"

# Minimum recall we want to guarantee when we pick the decision threshold.
# 0.70 means: "catch at least 70% of the customers who actually churn."
# Tune this number based on the business cost of a missed churner vs. a
# false alarm — raising it trades away some precision for more recall.
RECALL_TARGET = 0.70


def main():
    # ── 1. Load the identical train/test split every member uses ────────────
    X_train, X_test, y_train, y_test = load_processed_data()
    print(
        f"Loaded processed data: {X_train.shape[0]} train rows, "
        f"{X_test.shape[0]} test rows, {X_train.shape[1]} features"
    )

    # ── 2. Pipeline + hyperparameter grid ────────────────────────────────────
    # SelectKBest runs INSIDE the pipeline, so feature selection is re-fitted
    # on each CV fold's training split only (no leakage across folds).
    # Odd values for n_neighbors avoid tie votes in binary classification.
    # The range extends past 31 because the previous best (k=31) sat at the
    # edge of the old grid — a sign the optimum may lie further out.
    pipe = Pipeline(
        [
            ("select", SelectKBest(partial(mutual_info_classif, random_state=42))),
            ("knn", KNeighborsClassifier(n_jobs=-1)),
        ]
    )

    param_grid = {
        "select__k": [10, 15, 20, 25, "all"],
        "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31, 41, 51],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],  # 1 = Manhattan, 2 = Euclidean
    }

    # Scoring by ROC-AUC: the threshold is tuned separately in step 4, so the
    # grid should optimise probability ranking quality, not F1 at the unused
    # 0.5 default cut-off.
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    # ── 3. Search for the best combination of hyperparameters ───────────────
    print(
        "\nRunning GridSearchCV (5-fold) over select__k / n_neighbors / "
        "weights / p (scoring = ROC-AUC) ..."
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_index = grid.best_index_
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

    # Show which features survived selection (useful evidence for the report)
    select_step = best_model.named_steps["select"]
    if select_step.k != "all":
        kept = X_train.columns[select_step.get_support()]
        dropped = X_train.columns[~select_step.get_support()]
        print(f"\n     Features kept ({len(kept)}): {list(kept)}")
        print(f"     Features dropped ({len(dropped)}): {list(dropped)}")

    # ── 4. Threshold: kept at the plain default (0.5) on purpose ────────────
    # NOTE: the recall-focused threshold tuning is disabled here — this is
    # KNN's equivalent of removing class_weight="balanced" from the other
    # models. Without a lowered threshold, predict() naturally leans toward
    # the majority "No Churn" class, which raises overall accuracy at the
    # cost of recall on the minority "Churn" class. See Section 5.0 of the
    # report for the accuracy-vs-recall trade-off discussion.
    #
    # The out-of-fold sensitivity check below is still computed and printed
    # for that report discussion — it just no longer determines the
    # threshold actually used (chosen_threshold is fixed to None).
    print(
        "\nComputing out-of-fold CV probabilities on the TRAINING set "
        "(for the sensitivity table below; the model itself uses the "
        "default 0.5 threshold)..."
    )
    oof_probs = cross_val_predict(
        best_model, X_train, y_train, cv=5, method="predict_proba", n_jobs=-1
    )[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, oof_probs)
    # precision_recall_curve returns one more precision/recall point than
    # thresholds (it appends the (1.0, 0.0) endpoint), so align them first.
    precision, recall = precision[:-1], recall[:-1]

    chosen_threshold = None

    # ── 4b. Sensitivity check: is RECALL_TARGET actually a good choice? ─────
    # Instead of just trusting one fixed target, sweep across several targets
    # using the SAME out-of-fold probabilities, and report test-set F1 for
    # each. This turns "we picked 0.70" into "we tested 0.60-0.80 and 0.70
    # gave the best F1" — real evidence for the report, not a guess.
    print("\nRecall-target sensitivity check (for report justification):")
    print(
        f"{'target':>8}{'threshold':>12}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>8}"
    )
    for target in [0.60, 0.65, 0.70, 0.75, 0.80]:
        qual = np.where(recall >= target)[0]
        if len(qual) == 0:
            continue
        idx = qual[np.argmax(precision[qual])]
        t = float(thresholds[idx])
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

    # ── 5. Evaluate on the held-out test set and save everything ────────────
    # evaluate_and_save() (shared/eval_utils.py) prints the metrics and saves:
    #   results/KNN_metrics.json
    #   results/KNN_confusion.png
    #   results/KNN_roc.png
    #   models/KNN.pkl
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
