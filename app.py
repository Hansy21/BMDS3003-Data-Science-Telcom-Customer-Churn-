import glob
import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# =============================================================================
# PATHS
# =============================================================================
MODELS_DIR = "models"
RESULTS_DIR = "results"
PROCESSED_DIR = os.path.join("shared", "processed")


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_first(paths):
    """Load the first file that exists from a list of candidate paths."""
    for p in paths:
        if os.path.exists(p):
            return _load_pickle(p), p
    raise FileNotFoundError(f"None of these files exist: {paths}")


# =============================================================================
# LOAD SHARED ARTIFACTS
# =============================================================================
scaler, _ = _load_first(
    [os.path.join(PROCESSED_DIR, "scaler.pkl"), os.path.join(MODELS_DIR, "scaler.pkl")]
)
feature_columns, _ = _load_first(
    [
        os.path.join(PROCESSED_DIR, "feature_columns.pkl"),
        os.path.join(MODELS_DIR, "feature_columns.pkl"),
    ]
)

# Which model results/compare_models.py picked as the winner (may be stale —
# the Model Insights tab recomputes everything live so it's always accurate)
BEST_MODEL_NAME = None
_name_file = os.path.join(MODELS_DIR, "best_model_name.txt")
if os.path.exists(_name_file):
    with open(_name_file) as f:
        BEST_MODEL_NAME = f.read().strip()

# Every trained model available in models/, so the user can pick and compare
model_files = sorted(
    p
    for p in glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
    if os.path.basename(p)
    not in ("best_model.pkl", "scaler.pkl", "feature_columns.pkl")
)
model_names = [os.path.splitext(os.path.basename(p))[0] for p in model_files]

if not model_names:
    st.error(
        "No trained models found in models/. Run each member's training "
        "script first (e.g. `python member1_KNN/KNN.py`), then reload this app."
    )
    st.stop()


@st.cache_resource
def load_model(name):
    return _load_pickle(os.path.join(MODELS_DIR, f"{name}.pkl"))


@st.cache_data
def load_model_threshold(name):
    """
    Some members (e.g. KNN) tune a custom decision threshold instead of the
    default 0.5 to protect recall — see results/<name>_metrics.json. Reading
    it here keeps the live app's predictions consistent with the metrics
    that evaluate_and_save() reported. Falls back to 0.5 if a model has no
    saved metrics file or never set a custom threshold.
    """
    import json

    path = os.path.join(RESULTS_DIR, f"{name}_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get("threshold", 0.5)
    return 0.5


@st.cache_data
def load_test_set():
    """The same held-out test set every member evaluated on."""
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()
    return X_test, y_test


HAS_TEST_SET = os.path.exists(os.path.join(PROCESSED_DIR, "X_test.csv"))


@st.cache_data(show_spinner=False)
def score_model(name, _model, X_test, y_test):
    """Compute predictions + metrics for one model, live, from the test set."""
    y_prob = _model.predict_proba(X_test)[:, 1]
    threshold = load_model_threshold(name)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    return y_pred, y_prob, metrics


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="Telco Churn Predictor", page_icon="📡", layout="wide")

st.title("📡 Telco Customer Churn Predictor")
st.write(
    "BMDS2003 Data Science Project — predicting which telecom customers are "
    "likely to churn, and comparing the models each member built."
)

with st.expander("ℹ️ About this project"):
    st.markdown("""
        This prototype is the deployment step of a **CRISP-DM** workflow:
        every member trained a different classifier on the *same*
        preprocessed train/test split (`shared/preprocessing.py`), so the
        comparisons below are apples-to-apples.

        - **Predict tab** — fill in a customer's details and get a live
          churn prediction from whichever model you pick.
        - **Model Insights tab** — every chart is *computed live* from the
          held-out test set each time you open it, not a static image
          copied from the training run, so it always reflects the model
          files currently sitting in `models/`.
        """)

# =============================================================================
# SIDEBAR — choose which of the group's models powers the prediction
# =============================================================================
st.sidebar.header("⚙️ Model Settings")

default_index = (
    model_names.index(BEST_MODEL_NAME) if BEST_MODEL_NAME in model_names else 0
)

selected_model_name = st.sidebar.selectbox(
    "Choose which trained model makes the prediction:",
    model_names,
    index=default_index,
    help="Every .pkl file found in the models/ folder shows up here.",
)
model = load_model(selected_model_name)

if selected_model_name == BEST_MODEL_NAME:
    st.sidebar.success(
        f"⭐ {selected_model_name} was the best model by F1 "
        "when `compare_models.py` last ran."
    )

if HAS_TEST_SET:
    X_test, y_test = load_test_set()
    st.sidebar.caption(
        f"Live-scored on a held-out test set of {len(X_test)} customers."
    )
else:
    st.sidebar.warning(
        "shared/processed/X_test.csv not found — run "
        "`python shared/preprocessing.py` to enable live model insights."
    )

st.sidebar.divider()
st.sidebar.caption(
    "Tip: switch models above, then check the **Model Insights** tab to see "
    "how the choice changes accuracy, recall, and the confusion matrix."
)

# =============================================================================
# TABS
# =============================================================================
predict_tab, insights_tab = st.tabs(["🔮 Predict", "📊 Model Insights"])

# -----------------------------------------------------------------------------
# TAB 1 — PREDICT
# -----------------------------------------------------------------------------
with predict_tab:
    _active_threshold = load_model_threshold(selected_model_name)
    _threshold_note = (
        f" (decision threshold tuned to {_active_threshold:.2f}, not the default 0.50, to protect recall)"
        if abs(_active_threshold - 0.5) > 1e-9
        else ""
    )
    st.caption(
        f"Currently predicting using: **{selected_model_name}**{_threshold_note}"
    )

    # ── Quick example presets so users can try the app in one click ─────────
    DEFAULTS = {
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "No",
        "tenure": 12,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 65.0,
        "total_charges": 1500.0,
    }
    AT_RISK = {
        **DEFAULTS,
        "tenure": 2,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "tech_support": "No",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 95.0,
        "total_charges": 190.0,
    }
    LOYAL = {
        **DEFAULTS,
        "tenure": 60,
        "contract": "Two year",
        "internet_service": "DSL",
        "online_security": "Yes",
        "tech_support": "Yes",
        "paperless_billing": "No",
        "payment_method": "Credit card (automatic)",
        "monthly_charges": 45.0,
        "total_charges": 2700.0,
    }

    if "form_values" not in st.session_state:
        st.session_state.form_values = DEFAULTS.copy()

    st.markdown("**Try an example, or fill in the form yourself:**")
    p1, p2, p3 = st.columns(3)
    if p1.button("🟢 Loyal customer example", use_container_width=True):
        st.session_state.form_values = LOYAL.copy()
    if p2.button("🔴 At-risk customer example", use_container_width=True):
        st.session_state.form_values = AT_RISK.copy()
    if p3.button("↺ Reset form", use_container_width=True):
        st.session_state.form_values = DEFAULTS.copy()

    v = st.session_state.form_values

    with st.form("customer_form"):
        st.subheader("👤 Customer Details")
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
                index=["Male", "Female"].index(v["gender"]),
            )
            senior_citizen = st.selectbox(
                "Senior Citizen",
                ["No", "Yes"],
                index=["No", "Yes"].index(v["senior_citizen"]),
            )
            partner = st.selectbox(
                "Partner",
                ["Yes", "No"],
                index=["Yes", "No"].index(v["partner"]),
                help="Does the customer have a partner?",
            )
            dependents = st.selectbox(
                "Dependents", ["Yes", "No"], index=["Yes", "No"].index(v["dependents"])
            )
            tenure = st.slider(
                "Tenure (months)",
                0,
                72,
                v["tenure"],
                help="How long the customer has been with the company.",
            )
            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"],
                index=["Yes", "No"].index(v["phone_service"]),
            )

        with col2:
            opts = ["No", "Yes", "No phone service"]
            multiple_lines = st.selectbox(
                "Multiple Lines", opts, index=opts.index(v["multiple_lines"])
            )
            opts = ["DSL", "Fiber optic", "No"]
            internet_service = st.selectbox(
                "Internet Service", opts, index=opts.index(v["internet_service"])
            )
            opts = ["No", "Yes", "No internet service"]
            online_security = st.selectbox(
                "Online Security", opts, index=opts.index(v["online_security"])
            )
            online_backup = st.selectbox(
                "Online Backup", opts, index=opts.index(v["online_backup"])
            )
            device_protection = st.selectbox(
                "Device Protection", opts, index=opts.index(v["device_protection"])
            )
            tech_support = st.selectbox(
                "Tech Support",
                opts,
                index=opts.index(v["tech_support"]),
                help="Customers without tech support tend to churn more often.",
            )

        st.divider()
        st.subheader("💳 Billing Details")
        col3, col4 = st.columns(2)

        with col3:
            opts = ["No", "Yes", "No internet service"]
            streaming_tv = st.selectbox(
                "Streaming TV", opts, index=opts.index(v["streaming_tv"])
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", opts, index=opts.index(v["streaming_movies"])
            )
            opts = ["Month-to-month", "One year", "Two year"]
            contract = st.selectbox(
                "Contract",
                opts,
                index=opts.index(v["contract"]),
                help="Month-to-month contracts have the highest churn rate.",
            )

        with col4:
            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"],
                index=["Yes", "No"].index(v["paperless_billing"]),
            )
            opts = [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
            payment_method = st.selectbox(
                "Payment Method", opts, index=opts.index(v["payment_method"])
            )
            monthly_charges = st.slider(
                "Monthly Charges (RM)", 20.0, 120.0, float(v["monthly_charges"])
            )
            total_charges = st.slider(
                "Total Charges (RM)", 0.0, 9000.0, float(v["total_charges"])
            )

        submitted = st.form_submit_button("🔍 Predict Churn", use_container_width=True)

    if submitted:
        input_data = {
            "gender": 1 if gender == "Male" else 0,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": 1 if partner == "Yes" else 0,
            "Dependents": 1 if dependents == "Yes" else 0,
            "tenure": tenure,
            "PhoneService": 1 if phone_service == "Yes" else 0,
            "PaperlessBilling": 1 if paperless_billing == "Yes" else 0,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
        multi_value_cols = {
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaymentMethod": payment_method,
        }
        for col, value in multi_value_cols.items():
            for cat in feature_columns:
                if cat.startswith(col + "_"):
                    input_data[cat] = 1 if cat == f"{col}_{value}" else 0

        input_df = pd.DataFrame([input_data]).reindex(
            columns=feature_columns, fill_value=0
        )
        input_df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
            input_df[["tenure", "MonthlyCharges", "TotalCharges"]]
        )

        with st.spinner("Scoring customer..."):
            probability = model.predict_proba(input_df)[0][1]
            model_threshold = load_model_threshold(selected_model_name)
            prediction = int(probability >= model_threshold)

        st.divider()
        res_col, gauge_col = st.columns([1, 1])

        with res_col:
            st.subheader("📊 Prediction Result")
            st.caption(f"Model used: **{selected_model_name}**")
            if prediction == 1:
                st.error("⚠️ This customer is **likely to CHURN**")
            else:
                st.success("✅ This customer is **likely to STAY**")

            if probability >= 0.7:
                st.warning("🔴 HIGH RISK — consider a discount or contract upgrade")
            elif probability >= 0.4:
                st.warning("🟡 MEDIUM RISK — monitor this customer closely")
            else:
                st.info("🟢 LOW RISK — customer is likely to stay")

        with gauge_col:
            # A dynamically-drawn gauge, not a static image
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%"},
                    title={"text": "Churn Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#1f77b4"},
                        "steps": [
                            {"range": [0, 40], "color": "#d4edda"},
                            {"range": [40, 70], "color": "#fff3cd"},
                            {"range": [70, 100], "color": "#f8d7da"},
                        ],
                    },
                )
            )
            fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

        # ── What's driving this? (global feature importance for the model) ──
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])

        if importances is not None:
            st.subheader("🧭 What drives this model's predictions?")
            imp_df = (
                pd.DataFrame({"feature": feature_columns, "importance": importances})
                .sort_values("importance", ascending=False)
                .head(10)
                .iloc[::-1]
            )
            fig_imp = go.Figure(
                go.Bar(
                    x=imp_df["importance"],
                    y=imp_df["feature"],
                    orientation="h",
                    marker_color="#1f77b4",
                )
            )
            fig_imp.update_layout(
                height=380,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis_title="Relative importance",
                yaxis_title="",
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption(
                "Top 10 features this model relies on most overall (not specific "
                "to the customer above)."
            )
        else:
            st.caption(
                f"{selected_model_name} doesn't expose feature importances "
                "directly (e.g. KNN is instance-based, not rule- or "
                "weight-based), so no driver chart is shown."
            )

# -----------------------------------------------------------------------------
# TAB 2 — MODEL INSIGHTS  (everything here is computed live, not pasted PNGs)
# -----------------------------------------------------------------------------
with insights_tab:
    st.subheader("📊 How each member's model performs")

    if not HAS_TEST_SET:
        st.warning(
            "shared/processed/X_test.csv not found. Run "
            "`python shared/preprocessing.py` first, then reload this app."
        )
    else:
        with st.spinner("Scoring every trained model on the test set..."):
            rows, preds = [], {}
            for name in model_names:
                m = load_model(name)
                y_pred, y_prob, metrics = score_model(name, m, X_test, y_test)
                rows.append(metrics)
                preds[name] = (y_pred, y_prob)

        comparison_df = (
            pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
        )
        display_df = comparison_df.copy()
        display_df.insert(
            0,
            "Best",
            display_df["model"].apply(lambda m: "⭐" if m == BEST_MODEL_NAME else ""),
        )
        for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            display_df[c] = display_df[c].round(4)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption(
            "Ranked by F1 score, recomputed live on the same test set every "
            "member evaluated on — for churn, F1 balances precision (don't "
            "wrongly flag loyal customers) and recall (don't miss real churners)."
        )

        # ── Grouped bar chart, drawn live ────────────────────────────────────
        metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        fig_bar = go.Figure()
        for c in metric_cols:
            fig_bar.add_bar(name=c, x=comparison_df["model"], y=comparison_df[c])
        fig_bar.update_layout(
            barmode="group",
            yaxis_range=[0, 1],
            title="Model comparison across all metrics",
            height=420,
            legend_title="Metric",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()
        st.subheader("🔍 Detail view")
        detail_model = st.selectbox(
            "Inspect one model's confusion matrix and ROC curve:",
            model_names,
            index=model_names.index(selected_model_name),
            key="detail_model_select",
        )
        y_pred, y_prob = preds[detail_model]

        c1, c2 = st.columns(2)

        with c1:
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = go.Figure(
                go.Heatmap(
                    z=cm,
                    x=["No Churn", "Churn"],
                    y=["No Churn", "Churn"],
                    colorscale="Blues",
                    showscale=False,
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 18},
                )
            )
            fig_cm.update_layout(
                title=f"{detail_model} — Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                yaxis=dict(autorange="reversed"),
                height=380,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with c2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_scatter(
                x=fpr, y=tpr, mode="lines", name=f"{detail_model} (AUC={auc:.3f})"
            )
            fig_roc.add_scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(dash="dash", color="gray"),
            )
            fig_roc.update_layout(
                title=f"{detail_model} — ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=380,
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        # ── Overlay every model's ROC curve on one chart ─────────────────────
        with st.expander("Compare ROC curves for every model at once"):
            fig_all = go.Figure()
            for name in model_names:
                _, y_prob_n = preds[name]
                fpr, tpr, _ = roc_curve(y_test, y_prob_n)
                auc = roc_auc_score(y_test, y_prob_n)
                fig_all.add_scatter(
                    x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"
                )
            fig_all.add_scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(dash="dash", color="gray"),
            )
            fig_all.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=450,
            )
            st.plotly_chart(fig_all, use_container_width=True)
