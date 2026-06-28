# Member 3 — Random Forest

## Run
```bash
python shared/preprocessing.py                          # if not done
python member3_random_forest/train_random_forest.py
```
Outputs go to `results/` and `models/RandomForest.pkl`.

## Your tasks for the report
- **Model Selection (CLO1):** explain bagging / ensembles — why many trees beat
  one. Cite ≥2 sources.
- **Parameter tuning:** explain `n_estimators`, `max_depth`, `max_features`;
  note RandomizedSearchCV is used because the grid is large.
- **Evaluation:** report all metrics and compare against baseline + decision tree.
- Add a **feature-importance** chart (Random Forest provides this for free).
