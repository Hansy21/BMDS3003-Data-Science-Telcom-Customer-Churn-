"""
Model Insights tab — live metrics table, bar chart, confusion matrix, ROC.
"""

import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from prototype.charts import (
    make_confusion_heatmap,
    make_metrics_bar,
    make_roc_curve,
    make_roc_overlay,
)
from prototype.loaders import load_model, score_model


def render_insights_tab(
    model_names: list[str],
    best_model_name: str | None,
    selected_model_name: str,
    has_test: bool,
    X_test,
    y_test,
) -> None:
    """Entry point called from app.py inside the Model Insights tab."""
    st.subheader("How each model performs on the held-out test set")

    if not has_test:
        st.warning(
            "shared/processed/X_test.csv not found. Run "
            "`python shared/preprocessing.py` first, then reload this app."
        )
        return

    with st.spinner("Scoring every trained model on the test set..."):
        rows, preds = [], {}
        for name in model_names:
            model = load_model(name)
            y_pred, y_prob, metrics = score_model(name, model, X_test, y_test)
            rows.append(metrics)
            preds[name] = (y_pred, y_prob)

    comparison_df = (
        pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    )
    display_df = comparison_df.copy()
    display_df.insert(
        0,
        "Best",
        display_df["model"].apply(
            lambda m: "⭐" if m == best_model_name else ""
        ),
    )
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        display_df[col] = display_df[col].round(4)

    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption(
        "Ranked by F1. Metrics are recomputed live on the same test set used in training."
    )

    st.plotly_chart(make_metrics_bar(comparison_df), use_container_width=True)

    st.divider()
    st.subheader("Detail view")
    detail_model = st.selectbox(
        "Inspect one model’s confusion matrix and ROC curve",
        model_names,
        index=model_names.index(selected_model_name),
        key="detail_model_select",
    )
    y_pred, y_prob = preds[detail_model]

    c1, c2 = st.columns(2)
    with c1:
        cm = confusion_matrix(y_test, y_pred)
        st.plotly_chart(
            make_confusion_heatmap(cm, detail_model),
            use_container_width=True,
        )
    with c2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        st.plotly_chart(
            make_roc_curve(fpr, tpr, detail_model, auc),
            use_container_width=True,
        )

    with st.expander("Compare ROC curves for every model at once"):
        st.plotly_chart(
            make_roc_overlay(preds, y_test),
            use_container_width=True,
        )
