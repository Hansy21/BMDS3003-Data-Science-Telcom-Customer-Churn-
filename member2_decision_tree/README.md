# Member 2 — Decision Tree

## Run
```bash
python shared/preprocessing.py                          # if not done
python member2_decision_tree/train_decision_tree.py
```
Outputs go to `results/` and `models/DecisionTree.pkl`.

## Your tasks for the report
- **Model Selection (CLO1):** explain how a tree splits data (Gini vs entropy)
  and why it is more interpretable than the baseline. Cite ≥2 sources.
- **Parameter tuning:** explain `max_depth`, `min_samples_split`,
  `min_samples_leaf`; show how depth controls over/underfitting.
- **Evaluation:** report all metrics and compare against Member 1's baseline.
- Optional: show the most important features / a small tree diagram.
