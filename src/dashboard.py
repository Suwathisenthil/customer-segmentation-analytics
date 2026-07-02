"""
dashboard.py
Executive KPI Dashboard & Interactive Segment Explorer
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.logger import get_logger

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = get_logger(__name__)


def build_executive_dashboard(
    df: pd.DataFrame,
    kpi_df: pd.DataFrame,
    out_path="outputs/reports/executive_dashboard.html",
):
    """
    Build executive KPI dashboard.
    """

    if not HAS_PLOTLY:
        _matplotlib_kpi_dashboard(df, kpi_df, out_path)
        return

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "domain"}],
        ],
        subplot_titles=(
            "Annual Revenue by Segment ($)",
            "Revenue at Risk vs Retention Opportunity ($)",
            "Marketing ROI by Segment (%)",
            "Churn Risk Distribution",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.10,
    )

    colors = px.colors.qualitative.Set2

    kpi_sorted = kpi_df.sort_values("annual_revenue_est")

    fig.add_trace(
        go.Bar(
            x=kpi_sorted["annual_revenue_est"],
            y=kpi_sorted["segment_name"],
            orientation="h",
            marker_color=colors[: len(kpi_sorted)],
            text=[
                f"${v:,.0f}"
                for v in kpi_sorted["annual_revenue_est"]
            ],
            textposition="outside",
            name="Revenue",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=kpi_df["segment_name"],
            y=kpi_df["revenue_at_risk"],
            name="Revenue At Risk",
            marker_color="#E74C3C",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Bar(
            x=kpi_df["segment_name"],
            y=kpi_df["retention_opportunity"],
            name="Retention Opportunity",
            marker_color="#2ECC71",
        ),
        row=1,
        col=2,
    )

    roi_colors = []

    for value in kpi_df["marketing_roi_pct"]:
        if value >= 100:
            roi_colors.append("#27AE60")
        elif value >= 0:
            roi_colors.append("#F39C12")
        else:
            roi_colors.append("#E74C3C")

    fig.add_trace(
        go.Bar(
            x=kpi_df["segment_name"],
            y=kpi_df["marketing_roi_pct"],
            marker_color=roi_colors,
            name="Marketing ROI",
            text=[f"{v:.0f}%" for v in kpi_df["marketing_roi_pct"]],
            textposition="outside",
        ),
        row=2,
        col=1,
    )

    churn = (
        df["churn_risk_band"]
        .value_counts()
        .reset_index()
    )

    churn.columns = ["band", "count"]

    fig.add_trace(
        go.Pie(
            labels=churn["band"],
            values=churn["count"],
            hole=0.4,
            marker=dict(
                colors=[
                    "#27AE60",
                    "#F39C12",
                    "#E74C3C",
                ]
            ),
            showlegend=True,
        ),
        row=2,
        col=2,
    )

    total_customers = int(kpi_df["n_customers"].sum())
    total_revenue = kpi_df["annual_revenue_est"].sum()
    revenue_risk = kpi_df["revenue_at_risk"].sum()

    fig.update_layout(
        template="plotly_white",
        height=850,
        barmode="group",
        title={
            "text":
                "<b>Customer Segmentation Executive Dashboard</b><br>"
                f"<sup>Total Customers: {total_customers:,} | "
                f"Revenue: ${total_revenue:,.0f} | "
                f"Revenue At Risk: ${revenue_risk:,.0f}</sup>",
            "x": 0.5,
        },
    )

    fig.write_html(out_path)

    logger.info(f"Executive dashboard saved -> {out_path}")


def build_interactive_explorer(
    df,
    out_path="outputs/reports/interactive_explorer.html",
):

    if not HAS_PLOTLY:
        _matplotlib_explorer(df, out_path)
        return

    fig = px.scatter(
        df,
        x="annual_income",
        y="spending_score",
        color="segment_name",
        size="clv_proxy",
        hover_data=[
            "customer_id",
            "tenure_months",
            "churn_risk_band",
            "explanation",
        ],
        template="plotly_white",
        title="Interactive Customer Explorer",
        opacity=0.7,
        size_max=18,
    )

    fig.write_html(out_path)

    logger.info(f"Interactive explorer saved -> {out_path}")


def _matplotlib_kpi_dashboard(df, kpi_df, out_path):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    kpi_sorted = kpi_df.sort_values("annual_revenue_est")

    axes[0, 0].barh(
        kpi_sorted["segment_name"],
        kpi_sorted["annual_revenue_est"],
    )

    axes[0, 0].set_title("Revenue")

    x = range(len(kpi_df))

    axes[0, 1].bar(
        [i - 0.2 for i in x],
        kpi_df["revenue_at_risk"],
        width=0.4,
        label="At Risk",
    )

    axes[0, 1].bar(
        [i + 0.2 for i in x],
        kpi_df["retention_opportunity"],
        width=0.4,
        label="Recoverable",
    )

    axes[0, 1].set_xticks(list(x))
    axes[0, 1].set_xticklabels(
        kpi_df["segment_name"],
        rotation=20,
    )

    axes[0, 1].legend()

    axes[1, 0].bar(
        kpi_df["segment_name"],
        kpi_df["marketing_roi_pct"],
    )

    axes[1, 0].set_title("Marketing ROI")

    churn = df["churn_risk_band"].value_counts()

    axes[1, 1].pie(
        churn.values,
        labels=churn.index,
        autopct="%1.1f%%",
    )

    axes[1, 1].set_title("Churn Risk")

    plt.tight_layout()

    png = out_path.replace(".html", ".png")

    plt.savefig(png, dpi=150)

    plt.close()

    logger.info(f"Executive dashboard saved -> {png}")


def _matplotlib_explorer(df, out_path):

    fig, ax = plt.subplots(figsize=(10, 6))

    for segment in df["segment_name"].unique():

        temp = df[df["segment_name"] == segment]

        ax.scatter(
            temp["annual_income"],
            temp["spending_score"],
            s=temp["clv_proxy"] / 100,
            alpha=0.6,
            label=segment,
        )

    ax.legend()

    ax.set_xlabel("Annual Income")

    ax.set_ylabel("Spending Score")

    ax.set_title("Customer Explorer")

    plt.tight_layout()

    png = out_path.replace(".html", ".png")

    plt.savefig(png, dpi=150)

    plt.close()

    logger.info(f"Interactive explorer saved -> {png}")
