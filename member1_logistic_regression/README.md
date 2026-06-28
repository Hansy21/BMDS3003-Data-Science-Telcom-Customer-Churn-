# Member 1 — Logistic Regression (BASELINE)

## Run
```bash
python shared/preprocessing.py                                   # if not done
python member1_logistic_regression/train_logistic_regression.py
```
Outputs go to `results/` (metrics JSON, confusion matrix, ROC curve) and
`models/LogisticRegression.pkl`.

## Your tasks for the report
- **Model Selection (CLO1):** explain why Logistic Regression is the *baseline*
  — linear, simple, interpretable. Cite ≥2 sources.
- **Parameter tuning:** describe what `C`, `penalty`, and `solver` do and report
  the best values GridSearchCV found.
- **Evaluation:** report accuracy, precision, recall, F1, ROC-AUC. Explain why
  recall matters for churn (catching customers who actually leave).
- This is the bar the other 3 models must beat.
