"""
Plotly chart builders for the prediction dashboard and model insights.
"""

import pandas as pd
import plotly.graph_objects as go


def make_gauge(probability: float, threshold: float, accent: str):
    thr_pct = threshold * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 42}},
            title={"text": "Churn Probability", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": accent, "thickness": 0.35},
                "bgcolor": "#f4f6f7",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, thr_pct], "color": "#d5f5e3"},
                    {"range": [thr_pct, 70], "color": "#fdebd0"},
                    {"range": [70, 100], "color": "#f5b7b1"},
                ],
                "threshold": {
                    "line": {"color": "#2c3e50", "width": 3},
                    "thickness": 0.85,
                    "value": thr_pct,
                },
            },
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(t=50, b=20, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_donut(probability: float, accent: str):
    stay = max(0.0, 1.0 - probability)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Stay", "Churn"],
                values=[stay, probability],
                hole=0.62,
                marker=dict(
                    colors=["#27ae60", accent],
                    line=dict(color="#fff", width=2),
                ),
                textinfo="label+percent",
                textfont_size=14,
                hovertemplate="%{label}: %{percent}<extra></extra>",
                sort=False,
            )
        ]
    )
    fig.update_layout(
        title=dict(text="Stay vs Churn Likelihood", font=dict(size=16)),
        height=300,
        margin=dict(t=50, b=20, l=20, r=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.25),
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                text=f"<b>{probability * 100:.1f}%</b><br>churn",
                x=0.5,
                y=0.5,
                font_size=16,
                showarrow=False,
            )
        ],
    )
    return fig


def make_risk_bar(probability: float, accent: str):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[probability * 100],
            y=["Risk level"],
            orientation="h",
            marker=dict(color=accent),
            text=[f"{probability * 100:.1f}%"],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="Churn probability: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 100], title="Probability (%)"),
        yaxis=dict(showticklabels=False),
        height=120,
        margin=dict(t=10, b=40, l=10, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ecf0f1",
        showlegend=False,
    )
    return fig


def make_importance_chart(feature_columns, importances):
    imp_df = (
        pd.DataFrame({"feature": feature_columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(10)
        .iloc[::-1]
    )
    fig = go.Figure(
        go.Bar(
            x=imp_df["importance"],
            y=imp_df["feature"],
            orientation="h",
            marker=dict(
                color=imp_df["importance"],
                colorscale="Blues",
                showscale=False,
            ),
            hovertemplate="%{y}<br>importance=%{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Top 10 features this model relies on (global)",
        height=380,
        margin=dict(t=50, b=20, l=10, r=20),
        xaxis_title="Relative importance",
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_metrics_bar(comparison_df: pd.DataFrame):
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig = go.Figure()
    for col in metric_cols:
        fig.add_bar(name=col, x=comparison_df["model"], y=comparison_df[col])
    fig.update_layout(
        barmode="group",
        yaxis_range=[0, 1],
        title="Model comparison across all metrics",
        height=420,
        legend_title="Metric",
    )
    return fig


def make_confusion_heatmap(cm, model_name: str):
    fig = go.Figure(
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
    fig.update_layout(
        title=f"{model_name} — Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis=dict(autorange="reversed"),
        height=380,
    )
    return fig


def make_roc_curve(fpr, tpr, model_name: str, auc: float):
    fig = go.Figure()
    fig.add_scatter(
        x=fpr, y=tpr, mode="lines", name=f"{model_name} (AUC={auc:.3f})"
    )
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(dash="dash", color="gray"),
    )
    fig.update_layout(
        title=f"{model_name} — ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=380,
    )
    return fig


def make_roc_overlay(preds: dict, y_test):
    """Overlay every model's ROC on one figure. preds[name] = (y_pred, y_prob)."""
    from sklearn.metrics import roc_auc_score, roc_curve

    fig = go.Figure()
    for name, (_, y_prob) in preds.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})")
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(dash="dash", color="gray"),
    )
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
    )
    return fig


def make_pr_curve(precisions, recalls, model_name: str):
    fig = go.Figure()
    fig.add_scatter(x=recalls, y=precisions, mode="lines", name=model_name, line=dict(color="purple"))
    fig.update_layout(
        title=f"{model_name} — Precision-Recall Curve",
        xaxis_title="Recall (True Positive Rate)",
        yaxis_title="Precision",
        height=380,
    )
    return fig


def make_prob_dist(y_test, y_prob, model_name: str):
    fig = go.Figure()
    # Filter based on numpy arrays or pandas Series
    mask_0 = (y_test == 0)
    mask_1 = (y_test == 1)
    
    fig.add_trace(go.Histogram(x=y_prob[mask_0], name="Actual: No Churn", marker_color="green", opacity=0.5, histnorm='probability density', nbinsx=20))
    fig.add_trace(go.Histogram(x=y_prob[mask_1], name="Actual: Churn", marker_color="red", opacity=0.5, histnorm='probability density', nbinsx=20))
    fig.update_layout(
        title=f"{model_name} — Probability Distribution",
        xaxis_title="Predicted Probability of Churn",
        yaxis_title="Density",
        barmode='overlay',
        height=380,
    )
    return fig


def make_calibration_curve(prob_true, prob_pred, model_name: str):
    fig = go.Figure()
    fig.add_scatter(x=prob_pred, y=prob_true, mode="lines+markers", name=model_name, line=dict(color="darkorange", width=2), marker=dict(size=8))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly Calibrated", line=dict(dash="dash", color="gray"))
    fig.update_layout(
        title=f"{model_name} — Calibration Curve",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives (Actual Churn Rate)",
        height=380,
    )
    return fig
