"""Explore the customer churn data and create charts for the report.

Exploratory data analysis (EDA) means looking at the data before building a
model. This script checks the data quality and creates charts that show which
customer groups are more likely to churn. The charts are saved in results/eda.

Run: python shared/eda.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CSV_PATH = os.path.join(PROJECT_ROOT, "Telco_Cusomer_Churn.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "eda")
# Make the chart folder if this is the first time EDA is run.
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
# Use one consistent chart style so all figures look like part of one report.


def plot_outlier_analysis(df):
    """Check the three number-based columns for unusually high or low values.

    The IQR method uses the middle half of the data to create a normal range.
    Values far outside that range are marked as possible outliers. They are not
    automatically removed; the chart helps us decide whether they are valid.
    """

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    summary_rows = []

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, col in zip(axes, numeric_cols):
        # Q1 and Q3 are the values at 25% and 75% through the sorted data.
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        # IQR is the width of the middle half of the data.
        iqr = q3 - q1
        # Values beyond 1.5 IQR from the middle range are possible outliers.
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        n_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]

        # A boxplot makes the usual range and possible outliers easy to see.
        sns.boxplot(y=df[col], ax=ax, color="#4C78A8")

        data_min, data_max = df[col].min(), df[col].max()
        pad = (data_max - data_min) * 0.08
        ax.set_ylim(data_min - pad, data_max + pad)

        if upper_bound <= data_max + pad:
            ax.axhline(
                upper_bound,
                color="red",
                linestyle="--",
                linewidth=1.2,
                label=f"Upper bound = {upper_bound:.1f}",
            )
            ax.legend(fontsize=8, loc="upper left")

        ax.set_title(
            f"{col}\nIQR bounds: [{lower_bound:.1f}, {upper_bound:.1f}]  |  "
            f"{n_outliers} outlier(s) found",
            fontsize=10,
        )

        # Keep the exact values so they can also be used in a report table.
        summary_rows.append(
            {
                "feature": col,
                "Q1": round(q1, 2),
                "Q3": round(q3, 2),
                "IQR": round(iqr, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "actual_min": round(df[col].min(), 2),
                "actual_max": round(df[col].max(), 2),
                "n_outliers": n_outliers,
            }
        )

    fig.suptitle(
        "Outlier Assessment: Boxplots with IQR Bounds (1.5 x IQR)", fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "13_outlier_boxplots_iqr.png"), dpi=140)
    plt.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "outlier_summary.csv"), index=False)

    print("\nOutlier assessment (IQR method, 1.5 x IQR):")
    print(summary_df.to_string(index=False))

    return summary_df


def plot_additional_figures(df):

    # Figure 8: Compare churn rates for streaming services.
    # This helps us see whether subscribing to TV or movie streaming is linked
    # with customers staying or leaving.
    fig, ax = plt.subplots(figsize=(7, 5))
    stream_df = df.melt(
        id_vars="Churn",
        value_vars=["StreamingTV", "StreamingMovies"],
        var_name="Service",
        value_name="Subscribed",
    )
    # melt changes the two service columns into one "Service" column, making
    # it possible to compare both services in the same chart.
    rates = (
        stream_df.groupby(["Service", "Subscribed"])["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .reset_index(name="ChurnRate")
    )
    # (s == "Yes").mean() is the share of customers in that group who churned.
    sns.barplot(data=rates, x="Subscribed", y="ChurnRate", hue="Service", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Streaming Services")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "08_churn_rate_streaming.png"), dpi=120)
    plt.close()

    # Figure 9: Compare churn rates for protection and backup services.
    # These extra services may indicate whether customers are more connected to
    # the company and therefore less likely to leave.
    fig, ax = plt.subplots(figsize=(7, 5))
    addon_df = df.melt(
        id_vars="Churn",
        value_vars=["DeviceProtection", "OnlineBackup"],
        var_name="Service",
        value_name="Subscribed",
    )
    rates = (
        addon_df.groupby(["Service", "Subscribed"])["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .reset_index(name="ChurnRate")
    )
    sns.barplot(data=rates, x="Subscribed", y="ChurnRate", hue="Service", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Device Protection & Online Backup")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "09_churn_rate_protection_backup.png"), dpi=120)
    plt.close()

    # Figure 10: Compare churn rates by number of extra services.
    # Count how many of six optional services each customer has, then compare
    # the churn percentage for each count.
    addon_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["total_services"] = df[addon_cols].apply(
        lambda row: sum(v == "Yes" for v in row), axis=1
    )
    # For each row, count how many of the six optional services are "Yes".
    rates = (
        df.groupby("total_services")["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .reset_index(name="ChurnRate")
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=rates, x="total_services", y="ChurnRate", ax=ax, color="steelblue")
    ax.set_xlabel("Number of Add-On Services Subscribed")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Number of Add-On Services Subscribed")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "10_churn_rate_total_services.png"), dpi=120)
    plt.close()

    # Figure 11: Show churn rates by payment method and contract type.
    # A heatmap uses colour to make high and low churn-rate combinations easy
    # to spot. Darker red represents a higher churn rate.
    pivot = (
        df.groupby(["PaymentMethod", "Contract"])["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .unstack()
    )
    # unstack turns the two groupings into a grid for the heatmap.
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds", ax=ax)
    ax.set_title("Churn Rate (%) by Payment Method and Contract Type")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "11_churn_rate_payment_contract_heatmap.png"), dpi=120
    )
    plt.close()

    # Figure 12: Compare monthly and total charges, grouped by churn result.
    # Each dot is one customer; colour shows whether that customer churned.
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x="MonthlyCharges",
        y="TotalCharges",
        hue="Churn",
        alpha=0.5,
        ax=ax,
    )
    ax.set_title("Monthly Charges vs Total Charges by Churn Status")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "12_monthly_vs_total_scatter.png"), dpi=120)
    plt.close()


def main():
    df = pd.read_csv(CSV_PATH)
    # Read the raw dataset. This EDA script does not change the saved source file.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Blank total charges mean the customer has not been charged yet. These are
    # new customers with zero months of service, so we use 0 rather than guess.
    n_missing_tc = int(df["TotalCharges"].isna().sum())
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    print("=" * 60)
    print("DATA UNDERSTANDING — Telco Customer Churn")
    print("=" * 60)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"TotalCharges missing (filled with 0): {n_missing_tc}")
    print("\nColumn dtypes:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print("\nChurn distribution:")
    print(df["Churn"].value_counts())
    print(df["Churn"].value_counts(normalize=True).round(4))
    print("\nNumeric summary:")
    print(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2))

    # 1. Show how many customers churned and did not churn.
    # This reveals whether the classes are balanced. If far fewer people churn,
    # accuracy alone can be misleading because predicting "No" often looks good.
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["No", "Yes"]
    # Always show "No" before "Yes" so the order is the same in each chart.
    counts = df["Churn"].value_counts().reindex(order)
    colors = ["#4C78A8", "#E45756"]
    bars = ax.bar(order, counts.values, color=colors, edgecolor="black")
    ax.set_title("Churn Class Distribution", fontweight="bold")
    ax.set_ylabel("Number of customers")
    ax.set_xlabel("Churn")
    for b, c in zip(bars, counts.values):
        # Write both the customer count and percentage above each bar.
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 40,
            f"{c}\n({c / len(df) * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_churn_distribution.png"), dpi=140)
    plt.close()

    # 2. Compare number-based customer details by churn result.
    # The distributions help us see whether tenure or charges differ between
    # people who stayed and people who left.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
        # Draw one distribution for churners and another for non-churners.
        sns.histplot(
            data=df,
            x=col,
            hue="Churn",
            bins=30,
            element="step",
            stat="density",
            common_norm=False,
            ax=ax,
            palette={"No": "#4C78A8", "Yes": "#E45756"},
        )
        ax.set_title(col)
    fig.suptitle("Numeric Features by Churn Status", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_numeric_by_churn.png"), dpi=140)
    plt.close()

    # 3. Use boxplots to compare tenure and charges by churn result.
    # The line inside each box is the middle value, while the box shows where
    # the middle half of customers fall.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
        sns.boxplot(
            data=df,
            x="Churn",
            y=col,
            hue="Churn",
            ax=ax,
            palette={"No": "#4C78A8", "Yes": "#E45756"},
            order=["No", "Yes"],
            legend=False,
        )
        ax.set_title(col)
    fig.suptitle("Boxplots of Numeric Features by Churn", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_boxplots_numeric.png"), dpi=140)
    plt.close()

    # 4. Compare churn rates across important customer groups.
    # For each category, the height of the bar is the share of that group that
    # churned. The dashed line is the churn rate for all customers combined.
    cat_cols = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "TechSupport",
        "OnlineSecurity",
        "PaperlessBilling",
        "SeniorCitizen",
        "Partner",
    ]
    # Replace 0 and 1 with clear labels for the chart.
    plot_df = df.copy()
    # Use a copy so changing the chart label does not alter the original table.
    plot_df["SeniorCitizen"] = plot_df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    for ax, col in zip(axes, cat_cols):
        rates = (
            plot_df.groupby(col)["Churn"]
            .apply(lambda s: (s == "Yes").mean())
            .sort_values(ascending=False)
        )
        # Group customers by one category and calculate the share who churned.
        rates.plot(kind="bar", ax=ax, color="#E45756", edgecolor="black")
        ax.set_title(f"Churn rate by {col}", fontsize=10)
        ax.set_ylabel("Churn rate")
        ax.set_ylim(0, 0.6)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.axhline(0.265, color="gray", ls="--", lw=1, label="overall ~26.5%")
    fig.suptitle("Churn Rate Across Categorical Features", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "04_churn_rate_by_category.png"), dpi=140)
    plt.close()

    # 5. Show how churn changes with contract type and time as a customer.
    # Customers are placed into four tenure bands, making patterns easier to
    # read than looking at every individual month.
    tenure_bins = pd.cut(
        df["tenure"],
        bins=[-0.1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    )
    # Convert individual tenure months into four simple ranges for the heatmap.
    pivot = (
        df.assign(tenure_bin=tenure_bins)
        .groupby(["Contract", "tenure_bin"], observed=True)["Churn"]
        .apply(lambda s: (s == "Yes").mean())
        .unstack()
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Reds", ax=ax, vmin=0, vmax=0.7)
    ax.set_title("Churn Rate: Contract × Tenure Band", fontweight="bold")
    ax.set_xlabel("Tenure (months)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "05_contract_tenure_heatmap.png"), dpi=140)
    plt.close()

    # 6. Show how the number-based columns are related.
    # Correlation near 1 means two values tend to rise together; near -1 means
    # one tends to fall when the other rises; near 0 means little linear link.
    num = df[["tenure", "MonthlyCharges", "TotalCharges"]].copy()
    # Use a numeric 0/1 version of Churn because correlation needs numbers.
    num["Churn"] = (df["Churn"] == "Yes").astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation — Numeric Features & Churn", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "06_correlation_heatmap.png"), dpi=140)
    plt.close()

    # 7. Show the relationship between tenure and monthly charges.
    # Use at most 2,000 customers so the chart stays readable and reproducible.
    fig, ax = plt.subplots(figsize=(8, 5))
    sample = df.sample(n=min(2000, len(df)), random_state=42)
    # random_state keeps the same random sample each time the chart is made.
    sns.scatterplot(
        data=sample,
        x="tenure",
        y="MonthlyCharges",
        hue="Churn",
        alpha=0.5,
        palette={"No": "#4C78A8", "Yes": "#E45756"},
        ax=ax,
    )
    ax.set_title("Tenure vs Monthly Charges (sample of 2000)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "07_tenure_vs_monthly.png"), dpi=140)
    plt.close()

    # Save key figures in a small table for the report. This gives the report
    # exact numbers to use alongside the charts.
    summary = {
        "n_customers": len(df),
        "n_features_raw": df.shape[1] - 1,  # Do not count the Churn answer itself.
        "churn_rate": float((df["Churn"] == "Yes").mean()),
        "missing_TotalCharges_filled": n_missing_tc,
        "mean_tenure": float(df["tenure"].mean()),
        "mean_monthly_charges": float(df["MonthlyCharges"].mean()),
        "mean_total_charges": float(df["TotalCharges"].mean()),
    }
    pd.Series(summary).to_csv(
        os.path.join(OUT_DIR, "summary_stats.csv"), header=["value"]
    )

    # Create the remaining report figures after the main seven charts above.
    plot_additional_figures(df)
    # Check numeric data for unusually high or low values.
    plot_outlier_analysis(df)

    print(f"\n[OK] EDA figures saved to {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
