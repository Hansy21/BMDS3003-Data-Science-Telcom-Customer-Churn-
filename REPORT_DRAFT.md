# BMDS2003 Data Science — Group Assignment Report Draft

**Project title:** Telco Customer Churn Prediction Using Machine Learning  
**Course:** BMDS2003 Data Science (Session 202605)  
**Framework:** CRISP-DM  

> **How to use this file:** Copy each section into your Google Docs report template.  
> Replace `[Group X]`, member names/IDs, and any placeholders.  
> Insert images from `results/eda/` and `results/*_confusion.png`, `*_roc.png`, and `model_comparison.png`.  
> Run plagiarism check before submission. **Rewrite in your own words** where required by academic integrity policy.

---

## 1. Cover Page

| Field | Content |
| :--- | :--- |
| Project title | Telco Customer Churn Prediction Using Machine Learning |
| Course code | BMDS2003 |
| Group number | [Group X] |
| Members | [Name 1, ID] · [Name 2, ID] · [Name 3, ID] · [Name 4, ID] |

---

## 2. Executive Summary (≈ ½ page)

This project applies the CRISP-DM methodology to predict whether a telecommunications customer will **churn** (leave the service) using the public Telco Customer Churn dataset (7,043 customers, 21 raw attributes). After exploratory analysis and careful preprocessing (missing-value handling, encoding, stratified train/test split, and scaling), four supervised classifiers were trained on an identical data split:

1. **K-Nearest Neighbors (KNN)** — project **baseline**  
2. **Logistic Regression** — linear, interpretable probabilistic model  
3. **Random Forest** — bagging ensemble of decision trees  
4. **Histogram Gradient Boosting** — sequential boosting ensemble  

Models were tuned with 5-fold cross-validated grid search (F1 scoring) and evaluated on a held-out 20% test set using accuracy, precision, recall, F1, and ROC-AUC. Because missing a true churner is costly for retention campaigns, **recall** and **F1** were prioritised over raw accuracy.

**Main findings:**  
- Overall churn rate ≈ **26.5%** (class imbalance).  
- Strong risk signals: **month-to-month contracts**, **short tenure**, **fiber optic** internet, **no online security / tech support**, and **electronic check** payment.  
- Best model by F1: **Random Forest** (F1 = 0.6372, accuracy = 0.7608, recall = 0.7914).  
- Gradient Boosting had the highest ROC-AUC (0.8438); KNN baseline remained competitive on accuracy but lower on recall without the same class-weight handling.  
- A **Streamlit** prototype (`app.py`) deploys the trained models for interactive scoring and live model comparison.

**Business implication:** The system can flag high-risk customers so retention teams can offer targeted discounts, contract upgrades, or support outreach before churn occurs.

---

## 3. Business Understanding

### 3.1 Problem definition

Telecommunications providers face high acquisition costs and intense competition. When customers leave, the firm loses recurring revenue and may spend more to replace them. **Churn prediction** is a binary classification problem:

- **Positive class (1):** Customer will churn  
- **Negative class (0):** Customer will stay  

### 3.2 Business objectives

| Objective | Description |
| :--- | :--- |
| Predictive | Identify customers likely to churn before they leave |
| Analytical | Discover which product and billing attributes drive churn |
| Operational | Support a simple decision-support prototype for non-technical users |

### 3.3 Success criteria (analytical)

- Models outperform a naïve majority-class baseline (always “No Churn” → ~73% accuracy but **0% recall** on churn).  
- Prefer higher **recall** on the churn class (catch real leavers) while keeping precision acceptable (avoid flooding staff with false alarms).  
- At least three/four models compared fairly on the same split, with hyperparameter tuning documented.

### 3.4 Significance and impact

Even a modest improvement in early detection can protect monthly recurring revenue. Insights (e.g. month-to-month + fiber + no tech support) also guide product packaging and support investment—not only one-off campaigns.

---

## 4. Data Understanding

### 4.1 Dataset source and description

| Attribute | Detail |
| :--- | :--- |
| Name | Telco Customer Churn |
| Domain | Telecommunications CRM / subscription analytics |
| Size | 7,043 rows × 21 columns |
| Target | `Churn` (`Yes` / `No`) |
| Type | Mixed: categorical service flags + numeric tenure and charges |

**Main feature groups**

| Group | Examples |
| :--- | :--- |
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account | tenure, Contract, PaperlessBilling, PaymentMethod |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, TechSupport, StreamingTV, … |
| Billing | MonthlyCharges, TotalCharges |
| Target | Churn |

### 4.2 Summary statistics (insert tables/figures)

- **Churn balance:** No ≈ 73.5%, Yes ≈ 26.5% → imbalanced classification.  
  *Figure:* `results/eda/01_churn_distribution.png`
- **Numeric means (approx.):** tenure 32.4 months; MonthlyCharges 64.8; TotalCharges 2,280.  
  *Figures:* `02_numeric_by_churn.png`, `03_boxplots_numeric.png`
- **Data quality:** `TotalCharges` stored as text with **11 blank values** (customers with `tenure = 0` who have not yet been billed). Converted to numeric and filled with **0**. No other systematic missingness after that fix.

### 4.3 Key exploratory insights

1. **Contract type is dominant.** Month-to-month customers churn far more than one- or two-year contracts.  
   *Figure:* `04_churn_rate_by_category.png`
2. **Tenure protects against churn.** Short-tenure customers (especially 0–12 months) show elevated risk; interaction with contract is visible in the heatmap.  
   *Figure:* `05_contract_tenure_heatmap.png`
3. **Add-on security/support services** (OnlineSecurity, TechSupport = “Yes”) associate with lower churn than “No”.  
4. **Payment method:** Electronic check has higher churn than automatic bank/credit methods.  
5. **Fiber optic** internet users show higher churn than DSL / no internet (possible price or expectation effects).  
6. **Correlations:** tenure and TotalCharges are positively correlated (expected); churn correlates negatively with tenure.  
   *Figure:* `06_correlation_heatmap.png`

*(Expand each bullet with your own interpretation for “Excellent” CLO2 marks.)*

---

## 5. Data Preparation

All members share one preprocessing pipeline (`shared/preprocessing.py`) so model comparison is fair.

| Step | Action | Rationale |
| :--- | :--- | :--- |
| 1 | Load CSV | Single integrated table |
| 2 | `TotalCharges` → numeric; fill 11 NaNs with 0 | Correct type; new customers have no bill yet |
| 3 | Drop `customerID` | Identifier only; not predictive |
| 4 | Encode target `Churn` Yes→1, No→0 | Binary labels for sklearn |
| 5 | Binary map for gender, Partner, Dependents, PhoneService, PaperlessBilling | Compact 0/1 encoding |
| 6 | One-hot encode multi-category service/contract/payment columns | Avoid ordinal assumptions |
| 7 | Stratified train/test split 80/20, `random_state=42` | Preserve churn rate; reproducibility |
| 8 | `StandardScaler` on tenure, MonthlyCharges, TotalCharges **fit on train only** | Required for KNN/LogReg; avoid leakage |
| 9 | Save `X_train/X_test/y_train/y_test`, `scaler.pkl`, `feature_columns.pkl` | Shared inputs + Streamlit inference |

**Final feature matrix:** 40 columns after encoding (see `shared/processed/`).

**Outliers:** Numeric distributions were inspected via boxplots; extreme values of charges/tenure are plausible for real billing data and were **retained** (no aggressive trimming), to avoid distorting high-value or long-tenure customers.

---

## 6. Modelling

### 6.1 Model inventory (4 models; 1 baseline)

| Member | Model | Family | Role |
| :--- | :--- | :--- | :--- |
| 1 | K-Nearest Neighbors | Instance-based | **Baseline** |
| 2 | Logistic Regression | Linear probabilistic | Interpretable alternative |
| 3 | Random Forest | Bagging ensemble | Non-linear interactions |
| 4 | Hist. Gradient Boosting | Boosting ensemble | Same family as RF, different learning |

### 6.2 Why these models?

- **KNN (baseline):** Non-parametric, simple majority vote of neighbours; classic theoretical foundation (Cover & Hart, 1967). Sensitive to scale → uses shared StandardScaler. No `class_weight`; decision threshold tuned on out-of-fold training probabilities to protect recall.  
- **Logistic Regression:** Industry standard for churn odds; coefficients explain drivers (Hosmer et al., 2013; Huang et al., 2012). Uses `class_weight="balanced"`.  
- **Random Forest:** Reduces variance via bagging and random feature subsets (Breiman, 2001); strong on tabular churn (Idris et al., 2012).  
- **Gradient Boosting:** Sequentially corrects residuals (Friedman, 2001); often competitive with RF on structured data; enables bagging-vs-boosting discussion within tree ensembles.

### 6.3 Hyperparameter tuning

All models: **GridSearchCV**, **5-fold CV**, scoring = **F1**, then final score on the **held-out test set** only once.

| Model | Search space (summary) | Best params (this run) |
| :--- | :--- | :--- |
| KNN | n_neighbors, weights, p; threshold for recall ≥ 0.70 | `n_neighbors=31`, `weights=uniform`, `p=1` (Manhattan); threshold ≈ 0.42 |
| Logistic Regression | C, l1_ratio (L1/L2) | `C=1.0`, `l1_ratio=1.0` (L1), solver `saga` |
| Random Forest | n_estimators, max_depth, min_samples_leaf, max_features | `n_estimators=200`, `max_depth=10`, `min_samples_leaf=5`, `max_features=log2` |
| Gradient Boosting | learning_rate, max_depth, max_iter, min_samples_leaf | `learning_rate=0.05`, `max_depth=5`, `max_iter=100`, `min_samples_leaf=20` |

### 6.4 Class imbalance strategy

- Tree/linear models: `class_weight="balanced"`.  
- KNN: out-of-fold threshold tuning targeting ≥70% training-fold recall (no test leakage).  
- Metric focus: F1 + recall + ROC-AUC rather than accuracy alone.

---

## 7. Evaluation

### 7.1 Metrics used

| Metric | Why it matters here |
| :--- | :--- |
| Accuracy | Overall correctness (misleading alone under imbalance) |
| Precision | Of predicted churners, how many truly churn (campaign efficiency) |
| Recall | Of actual churners, how many we catch (retention priority) |
| F1 | Harmonic mean of precision & recall (ranking metric) |
| ROC-AUC | Ranking quality across thresholds |

### 7.2 Results table

Held-out test set (n = 1,409; stratified 20%). Ranked by F1.

| model | accuracy | precision | recall | f1 | roc_auc |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **RandomForest** | **0.7608** | **0.5333** | **0.7914** | **0.6372** | 0.8417 |
| GradientBoosting | 0.7537 | 0.5242 | 0.7807 | 0.6273 | **0.8438** |
| LogisticRegression | 0.7374 | 0.5034 | 0.7861 | 0.6138 | 0.8420 |
| KNN (baseline) | 0.7594 | 0.5358 | 0.7005 | 0.6072 | 0.8262 |

*Figures to insert:*  
- `results/model_comparison.png`  
- Each model’s `*_confusion.png` and `*_roc.png`  
- Streamlit “Model Insights” screenshots (optional but strong for CLO3 prototype marks)

### 7.3 Discussion

1. **Baseline comparison:** All three advanced models beat the KNN baseline on **F1** and **ROC-AUC**. KNN remained competitive on accuracy (0.759) and precision (0.536) after threshold tuning, but its recall (0.701) lagged the class-weighted models (~0.78–0.79). This shows why accuracy alone is a weak success metric under imbalance.  
2. **Bagging vs boosting:** Random Forest edged Gradient Boosting on F1 (0.637 vs 0.627) and recall, while Gradient Boosting led slightly on ROC-AUC (0.844 vs 0.842). Both tree ensembles outperformed the linear Logistic Regression on F1, supporting the EDA finding that interactions (e.g. Contract × tenure) matter.  
3. **Precision–recall trade-off:** Winning RF precision is only ~0.53: about half of customers flagged as churn risk will not leave. That is acceptable if contact cost is low relative to lost lifetime value; if campaign budget is tight, raise the decision threshold or prioritise only the top probability decile.  
4. **Feature drivers:** RF impurity importance ranked **Contract_Month-to-month**, **tenure**, **TotalCharges**, **MonthlyCharges**, **TechSupport_No**, **OnlineSecurity_No**, and **InternetService_Fiber optic** highest—aligned with EDA churn-rate charts.  
5. **Limitations:**  
   - Single static snapshot (no behavioural time series).  
   - Reason for leaving not recorded.  
   - Demographic fairness not audited.  
   - Finite grid search; further gains possible with richer features or cost-sensitive thresholds.  
6. **Improvements:** Feature engineering (tenure bands, count of add-on services), probability calibration, cost-sensitive learning, and A/B testing of retention offers on model-scored cohorts.

### 7.4 Deployment prototype

`app.py` (Streamlit) provides:

- Interactive customer form → churn probability gauge  
- Model selector across all `.pkl` files  
- Live metrics, confusion matrices, and ROC curves on the shared test set  

```bash
python -m streamlit run app.py
```

---

## 8. Conclusion

This project demonstrated an end-to-end CRISP-DM workflow for **Telco Customer Churn** prediction: business framing, EDA, shared preprocessing, four tuned classifiers (including a KNN baseline), rigorous multi-metric evaluation, and a functional Streamlit prototype.

**Advantages:** Fair multi-model comparison; interpretable business insights; deployable scoring tool.  
**Limitations:** Cross-sectional data; imperfect precision at high recall; no production MLOps monitoring.  
**Contribution:** Supports proactive retention by ranking at-risk customers and explaining risk drivers.  
**Lessons learned:** Shared preprocessing is essential for fair comparison; accuracy alone is insufficient under imbalance; prototype demos strengthen stakeholder communication.

---

## 9. References (APA 7th edition — minimum 5; include ≥2 academic)

1. Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324  

2. Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory, 13*(1), 21–27. https://doi.org/10.1109/TIT.1967.1053964  

3. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232. https://doi.org/10.1214/aos/1013203451  

4. Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied logistic regression* (3rd ed.). Wiley.  

5. Huang, B., Kechadi, M. T., & Buckley, B. (2012). Customer churn prediction in telecommunications. *Expert Systems with Applications, 39*(1), 1414–1425. https://doi.org/10.1016/j.eswa.2011.08.024  

6. Idris, A., Rizwan, M., & Khan, A. (2012). Churn prediction in telecom using Random Forest and PSO based data balancing in favor of minority class. *Applied Soft Computing, 12*(8), 2435–2446.  

7. Natekin, A., & Knoll, A. (2013). Gradient boosting machines, a tutorial. *Frontiers in Neurorobotics, 7*, Article 21. https://doi.org/10.3389/fnbot.2013.00021  

8. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.  

9. Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS Inc.  

*(Add in-text citations in the Google Doc, e.g. (Breiman, 2001); (Cover & Hart, 1967).)*

---

## Appendix A — How to reproduce results

```bash
cd TelcoChurn
pip install -r requirements.txt

python shared/eda.py
python shared/preprocessing.py

python member1_KNN/KNN.py
python member2_logistic_regression/train_logistic_regression.py
python member3_random_forest/train_random_forest.py
python member4_gradient_boosting/train_gradient_boosting.py

python results/compare_models.py
python -m streamlit run app.py
```

ZIP submission name (example): `GroupX_RSWY1S2_DataScienceProject.zip`  
Include all `.py` / `.ipynb` and the Streamlit app; do **not** rely on huge model binaries if file size is limited—document retrain commands in README.

---

## Appendix B — Figure checklist for the Google Doc

| # | File | Section |
| :--- | :--- | :--- |
| 1 | `results/eda/01_churn_distribution.png` | Data Understanding |
| 2 | `results/eda/02_numeric_by_churn.png` | EDA |
| 3 | `results/eda/03_boxplots_numeric.png` | EDA |
| 4 | `results/eda/04_churn_rate_by_category.png` | EDA |
| 5 | `results/eda/05_contract_tenure_heatmap.png` | EDA |
| 6 | `results/eda/06_correlation_heatmap.png` | EDA |
| 7 | `results/eda/07_tenure_vs_monthly.png` | EDA |
| 8 | `results/*_confusion.png` (×4) | Evaluation |
| 9 | `results/*_roc.png` (×4) | Evaluation |
| 10 | `results/model_comparison.png` | Evaluation |
| 11 | Streamlit screenshots | Advanced analytics / prototype |
