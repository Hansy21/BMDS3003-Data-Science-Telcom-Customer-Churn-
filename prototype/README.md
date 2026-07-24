# Prototype package (`prototype/`)

Streamlit UI for the Telco Churn deployment demo.  
**Entry point:** still run from project root:

```bash
python -m streamlit run app.py
```

## File map

| File | Responsibility |
|------|----------------|
| `app.py` (project root) | Thin entry: page config → load data → sidebar → tabs |
| `config.py` | Paths, presets (Loyal / At-risk), select options |
| `loaders.py` | Load scaler, models, thresholds, test set (cached) |
| `features.py` | Encode form → feature row; risk bands |
| `charts.py` | All Plotly figures (gauge, donut, ROC, …) |
| `styles.py` | Custom CSS (banner, KPI cards) |
| `sidebar.py` | Sidebar UI (model + form + Predict button) |
| `tabs/predict.py` | Prediction dashboard (results + charts) |
| `tabs/insights.py` | Live model comparison on the test set |

## Data flow

```
sidebar inputs  →  features.build_input_dataframe()
                →  model.predict_proba()
                →  tabs/predict.py dashboard charts

models/*.pkl    →  loaders.load_model() / score_model()
                →  tabs/insights.py metrics + ROC
```

## Editing tips

- Change form fields / presets → `config.py` + `sidebar.py`
- Change chart look → `charts.py`
- Change result layout → `tabs/predict.py`
- Change comparison metrics UI → `tabs/insights.py`
