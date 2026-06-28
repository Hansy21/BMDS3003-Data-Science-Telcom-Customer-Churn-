# Member 4 — Gradient Boosting

## Run
```bash
python shared/preprocessing.py                              # if not done
python member4_gradient_boosting/train_gradient_boosting.py
```
Outputs go to `results/` and `models/GradientBoosting.pkl`.

## Your tasks for the report
- **Model Selection (CLO1):** explain boosting vs bagging — how each tree fixes
  the previous tree's mistakes. Cite ≥2 sources.
- **Parameter tuning:** explain the trade-off between `learning_rate` and
  `n_estimators`, plus `max_depth` and `subsample`.
- **Imbalance:** note that GradientBoosting has no `class_weight`, so we pass
  `sample_weight` instead.
- **Evaluation:** report all metrics, compare against everyone, and discuss
  whether the extra complexity is worth it for this dataset.
