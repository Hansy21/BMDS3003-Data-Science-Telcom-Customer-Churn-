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
├── member1_KNN/
│   └── KNN.py                     # Member 1 — BASELINE model (K-Nearest Neighbors)
├── member2_/
│   └── member2_model.py           # Member 2 — model TBD (rename folder + file once chosen)
├── member3_random_forest/
│   └── train_random_forest.py     # Member 3 — Random Forest (not yet implemented)
├── member4_/
│   └── member4_model.py           # Member 4 — model TBD (rename folder + file once chosen)
├── results/
│   ├── compare_models.py          # STEP 3 — compares all models, picks the best
│   └── *.png / *.json / *.csv     # generated metrics, plots, comparison table
├── models/                        # saved .pkl models (best_model.pkl used by app)
└── app.py                         # Streamlit deployment prototype
```

## How to run (in order)

```bash
# 0. Install dependencies (once)
pip install pandas numpy scikit-learn matplotlib seaborn streamlit plotly

# 1. Build the shared train/test data (run ONCE, before any model)
python shared/preprocessing.py

# 2. Each member trains their own model (any order)
python member1_KNN/KNN.py
python member2_/member2_model.py          # TODO: implement
python member3_random_forest/train_random_forest.py   # TODO: implement
python member4_/member4_model.py          # TODO: implement

# 3. Compare all models and pick the winner
python results/compare_models.py

# 4. Launch the deployment prototype
python -m streamlit run app.py
```

## About the Streamlit prototype (`app.py`)

- **Predict tab** — fill in a customer's details (or click a "Loyal
  customer" / "At-risk customer" example button to try it in one click)
  and get a live churn prediction from whichever trained model you pick
  in the sidebar. Shows a probability gauge and, for models that support
  it, a chart of which features drive that model's predictions most.
- **Model Insights tab** — every chart here (metrics table, grouped bar
  chart, confusion matrix, ROC curves) is *computed live* from the
  held-out test set each time the app runs, not a static image copied
  from a training run. This means it always reflects whatever models are
  currently sitting in `models/`, even if someone forgets to re-run
  `results/compare_models.py` after training a new model.
- The sidebar model dropdown lists every `.pkl` file found in `models/`,
  so it automatically picks up new models as members finish their scripts
  — no code changes needed.

> **Note:** `results/compare_models.py` automatically picks up every
> `*_metrics.json` file it finds in `results/`, so it works with however many
> members have run their script so far — you don't need all 4 done to test it,
> but the final comparison for the report should include all of them.

## Why this layout

All members train on the **identical** train/test split produced by
`shared/preprocessing.py`. This makes the model comparison fair and lets each
member work independently on just their own file.

## Models & roles

| Member | Model | Role | Status |
|--------|-------|------|--------|
| 1 | K-Nearest Neighbors (KNN) | **Baseline** (simple, distance-based, no assumptions about decision boundary) | ✅ Implemented |
| 2 | *TBD* | — | ⬜ Not started |
| 3 | Random Forest | Ensemble of trees (bagging) | ⬜ Not started |
| 4 | *TBD* | — | ⬜ Not started |

Once Members 2 and 4 decide on their models, rename `member2_` /
`member4_` to `member2_<model_name>` / `member4_<model_name>` and update this
table and the run commands above to match.

All models are tuned with cross-validated hyperparameter search (see each
member's script for the specific grid) and scored on accuracy, precision,
recall, F1, and ROC-AUC. The winner (by F1) is promoted to
`models/best_model.pkl` and served by the Streamlit app.

## Member 1 — KNN baseline notes

- Hyperparameters tuned via 5-fold `GridSearchCV` over `n_neighbors`
  (3–31), `weights` (uniform/distance), and `p` (Manhattan vs Euclidean
  distance), scored by F1.
- KNN is distance-based, so it relies on the numeric features
  (`tenure`, `MonthlyCharges`, `TotalCharges`) already being scaled by
  `shared/preprocessing.py` — no extra scaling is done in `KNN.py`.
- Outputs: `results/KNN_metrics.json`, `results/KNN_confusion.png`,
  `results/KNN_roc.png`, `models/KNN.pkl`.
- Suggested references for the report: Cover & Hart (1967) on the
  theoretical basis of the nearest-neighbor rule, and Pedregosa et al.
  (2011) for the scikit-learn implementation used.