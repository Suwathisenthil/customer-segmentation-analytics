"""
insights.py
Automated Insight Generator & Segment Labeller
"""

import os
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


def label_segments(df: pd.DataFrame):
    """
    Label customer clusters based on average CLV.
    """

    cluster_stats = (
        df.groupby("cluster")
        .agg(
            avg_clv=("clv_proxy", "mean"),
            avg_spend=("spending_score", "mean"),
            avg_loyalty=("loyalty_index", "mean"),
            avg_returns=("returns_rate", "mean"),
            count=("customer_id", "count"),
        )
        .reset_index()
        .sort_values("avg_clv", ascending=False)
        .reset_index(drop=True)
    )

    labels = [
        "Champions",
        "Loyal Customers",
        "Potential Loyalists",
        "At-Risk",
        "Need Attention",
        "Hibernating",
    ]

    cluster_stats["segment_name"] = [
        labels[i] if i < len(labels) else f"Segment {i}"
        for i in range(len(cluster_stats))
    ]

    df = df.merge(
        cluster_stats[["cluster", "segment_name"]],
        on="cluster",
        how="left",
    )

    logger.info(
        f"Segments labelled: {cluster_stats['segment_name'].tolist()}"
    )

    return df, cluster_stats


def generate_insights(
    cluster_stats: pd.DataFrame,
    out_path: str = "outputs/reports/insight_report.txt",
):
    """
    Generate automated business insight report.
    """

    lines = [
        "=" * 60,
        "AUTOMATED BUSINESS INSIGHT REPORT",
        "=" * 60,
        "",
    ]

    for _, row in cluster_stats.iterrows():

        lines.extend(
            [
                f"Segment : {row['segment_name']} (n={int(row['count'])})",
                f"Average CLV Proxy : ${row['avg_clv']:,.2f}",
                f"Average Spend     : {row['avg_spend']:.2f}",
                f"Loyalty Index     : {row['avg_loyalty']:.2f}",
                f"Returns Rate      : {row['avg_returns']:.2%}",
            ]
        )

        if (
            row["avg_clv"] > cluster_stats["avg_clv"].median()
            and row["avg_spend"] > 60
        ):
            recommendation = (
                "Action: Offer VIP rewards to retain high-value customers."
            )

        elif (
            row["avg_loyalty"] > 0.60
            and row["avg_spend"] < 50
        ):
            recommendation = (
                "Action: Launch upselling campaigns for loyal customers."
            )

        elif row["avg_returns"] > 0.25:
            recommendation = (
                "Action: Investigate product quality due to high return rate."
            )

        elif row["avg_loyalty"] < 0.20:
            recommendation = (
                "Action: Start onboarding and engagement campaigns."
            )

        else:
            recommendation = (
                "Action: Run customer re-engagement campaigns."
            )

        lines.append(recommendation)
        lines.append("-" * 60)

    report = "\n".join(lines)

    output_dir = os.path.dirname(out_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Insight report saved: {out_path}")

    return report