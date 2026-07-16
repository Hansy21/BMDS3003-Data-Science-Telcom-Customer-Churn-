# Telco Customer Churn — BMDS2003 Data Science Project

Predicting which telecom customers will **churn** (leave) using the **CRISP-DM**
framework. Group of 4 — each member owns one machine learning model.

| Item | Detail |
|------|--------|
| Dataset | `Telco_Cusomer_Churn.csv` (7,043 customers) |
| Task | Binary classification (`Churn` Yes/No) |
| Baseline | K-Nearest Neighbors (Member 1) |
| Prototype | Streamlit app (`app.py`) |
| Report draft | `REPORT_DRAFT.md` (copy into Google Docs) |

## Project structure

```
TelcoChurn/
├── Telco_Cusomer_Churn.csv
├── Telco_churn.ipynb              # optional exploration notebook
├── REPORT_DRAFT.md                # written report skeleton for Google Docs
├── requirements.txt
├── app.py                         # Streamlit entry (thin — wires modules)
├── prototype/                     # Streamlit UI package (see prototype/README.md)
│   ├── config.py                  # paths, presets, select options
│   ├── loaders.py                 # load models / scaler / test set
│   ├── features.py                # encode form → model input
│   ├── charts.py                  # Plotly charts
│   ├── styles.py                  # CSS
│   ├── sidebar.py                 # all sidebar inputs
│   └── tabs/
│       ├── predict.py             # prediction dashboard
│       └── insights.py            # model comparison
├── shared/
│   ├── preprocessing.py           # STEP 1 — clean / encode / split (run once)
│   ├── eda.py                     # EDA figures → results/eda/
│   ├── eval_utils.py              # shared scoring helpers
│   └── processed/                 # train/test CSVs + scaler + feature list
├── member1_KNN/
│   └── KNN.py                     # Member 1 — BASELINE (KNN)
├── member2_logistic_regression/
│   └── train_logistic_regression.py
├── member3_random_forest/
│   └── train_random_forest.py
├── member4_gradient_boosting/
│   └── train_gradient_boosting.py
├── results/
│   ├── compare_models.py          # STEP 3 — pick best model by F1
│   ├── eda/                       # charts for the report
│   └── *_metrics.json / *.png
└── models/                        # *.pkl + best_model.pkl
```

## How to run (in order)

```bash
# 0. Install dependencies (once)
pip install -r requirements.txt

# 1. EDA figures for the report (optional but recommended)
python shared/eda.py

# 2. Shared train/test data (run ONCE before any model)
python shared/preprocessing.py

# 3. Train each model (any order after step 2)
python member1_KNN/KNN.py
python member2_logistic_regression/train_logistic_regression.py
python member3_random_forest/train_random_forest.py
python member4_gradient_boosting/train_gradient_boosting.py

# 4. Compare models and promote the winner
python results/compare_models.py

# 5. Launch the deployment prototype
python -m streamlit run app.py
```

## Models & roles

| Member | Model | Role | Script |
|--------|-------|------|--------|
| 1 | K-Nearest Neighbors | **Baseline** | `member1_KNN/KNN.py` |
| 2 | Logistic Regression | Linear / interpretable | `member2_logistic_regression/train_logistic_regression.py` |
| 3 | Random Forest | Bagging ensemble | `member3_random_forest/train_random_forest.py` |
| 4 | Hist. Gradient Boosting | Boosting ensemble | `member4_gradient_boosting/train_gradient_boosting.py` |

All models use **5-fold GridSearchCV** scored by **F1**, then evaluate once on the
held-out test set (accuracy, precision, recall, F1, ROC-AUC). The winner by F1
is copied to `models/best_model.pkl` for the app default.

## Streamlit prototype (`app.py`)

- **Sidebar** — all controls live here: model picker, Loyal / At-risk / Reset
  presets, full customer profile (demographics, services, billing), and
  **Predict churn**.
- **Prediction Dashboard** — large CHURN / STAY banner, KPI cards, gauge +
  donut charts, risk bar, recommended action, customer snapshot, and global
  feature-importance chart (when the model supports it).
- **Model Insights** — live metrics table, grouped bars, confusion matrix,
  and ROC curves recomputed from the shared test set for every `.pkl` in `models/`.

## Assignment checklist (BMDS2003)

| Requirement | Status |
|-------------|--------|
| CRISP-DM style report structure | `REPORT_DRAFT.md` |
| EDA + visualisations | `shared/eda.py` → `results/eda/` |
| Documented preprocessing | `shared/preprocessing.py` |
| ≥ 3–4 ML models (1 baseline) | 4 models (KNN baseline) |
| Parameter tuning | GridSearchCV in each member script |
| Metrics + comparison | `results/compare_models.py` |
| Deployment prototype | `app.py` Streamlit |
| ≥ 5 APA references (incl. academic) | listed in report draft |

## ZIP submission

Name format from the brief: `GroupX_RSWY1S2_DataScienceProject.zip`

Include Python/notebook files and the Streamlit app. Prefer regenerating large
`.pkl` files with the commands above if the ZIP is size-limited.

## Notes for members

- Always train on data from `shared/preprocessing.py` so comparisons stay fair.
- Do not fit scalers or choose thresholds using the test set.
- Put figures from `results/` and `results/eda/` into the Google Docs report so
  the report is self-contained (code snippets are not required in the doc).
