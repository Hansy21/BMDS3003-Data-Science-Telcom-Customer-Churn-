# Telco Customer Churn — BMDS2003 Data Science Project

Predicting which telecom customers will **churn** (leave) using the CRISP-DM
framework. Group of 4 — each member owns one machine learning model.

## Project structure

```
TelcoChurn/
├── Telco_Cusomer_Churn.csv        # raw dataset (7043 customers)
├── Telco_churn.ipynb              # exploration notebook (EDA + reference models)
├── shared/
│   ├── preprocessing.py           # STEP 1 — cleans/encodes/splits data for everyone
│   ├── eval_utils.py              # shared scoring + plotting (do not run directly)
│   └── processed/                 # generated train/test data (created by step 1)
├── member1_logistic_regression/   # Member 1 — BASELINE model
├── member2_decision_tree/         # Member 2
├── member3_random_forest/         # Member 3
├── member4_gradient_boosting/     # Member 4
├── results/
│   ├── compare_models.py          # STEP 3 — compares all models, picks the best
│   └── *.png / *.json / *.csv     # generated metrics, plots, comparison table
├── models/                        # saved .pkl models (best_model.pkl used by app)
└── app.py                         # Streamlit deployment prototype
```

## How to run (in order)

```bash
# 0. Install dependencies (once)
pip install pandas numpy scikit-learn matplotlib seaborn streamlit

# 1. Build the shared train/test data (run ONCE, before any model)
python shared/preprocessing.py

# 2. Each member trains their own model (any order)
python member1_logistic_regression/train_logistic_regression.py
python member2_decision_tree/train_decision_tree.py
python member3_random_forest/train_random_forest.py
python member4_gradient_boosting/train_gradient_boosting.py

# 3. Compare all 4 models and pick the winner
python results/compare_models.py

# 4. Launch the deployment prototype
python -m streamlit run app.py
```

## Why this layout

All four members train on the **identical** train/test split produced by
`shared/preprocessing.py`. This makes the model comparison fair and lets each
member work independently on just their own file.

## Models & roles

| Member | Model | Role |
|--------|-------|------|
| 1 | Logistic Regression | **Baseline** (simplest, most interpretable) |
| 2 | Decision Tree | Single tree, interpretable splits |
| 3 | Random Forest | Ensemble of trees (bagging) |
| 4 | Gradient Boosting | Sequential ensemble (boosting) |

All models are tuned with cross-validated hyperparameter search and scored on
accuracy, precision, recall, F1, and ROC-AUC. The winner (by F1) is promoted to
`models/best_model.pkl` and served by the Streamlit app.
