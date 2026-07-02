"""report.py - Static Plotly report charts (segment profile + radar)."""
import pandas as pd
import os

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import matplotlib.pyplot as plt
from src.logger import get_logger

logger = get_logger(__name__)


def segment_profile_chart(df: pd.DataFrame,
                           out_path: str = "outputs/reports/segment_profile.html"):
    stats = (
        df.groupby("segment_name")
          .agg(CLV=("clv_proxy","mean"), SpendScore=("spending_score","mean"),
               Count=("customer_id","count"))
          .reset_index()
    )
    if HAS_PLOTLY:
        fig = px.scatter(
            stats, x="SpendScore", y="CLV", size="Count", color="segment_name",
            text="segment_name",
            title="Segment Profile: CLV vs Spending Score",
            labels={"SpendScore":"Avg Spending Score","CLV":"Avg CLV Proxy ($)"},
            size_max=60, template="plotly_white",
        )
        fig.update_traces(textposition="top center")
        fig.write_html(out_path)
        logger.info(f"Segment profile chart saved -> {out_path}")
    else:
        fig, ax = plt.subplots(figsize=(8,5))
        ax.scatter(stats["SpendScore"], stats["CLV"], s=stats["Count"]/2, alpha=0.7)
        for _, r in stats.iterrows():
            ax.annotate(r["segment_name"], (r["SpendScore"], r["CLV"]))
        ax.set_xlabel("Avg Spending Score"); ax.set_ylabel("Avg CLV Proxy ($)")
        ax.set_title("Segment Profile")
        plt.tight_layout()
        plt.savefig(out_path.replace(".html", ".png"), dpi=150)
        plt.close()
        logger.info("Segment profile chart saved (matplotlib fallback).")


def radar_chart(cluster_stats: pd.DataFrame,
                out_path: str = "outputs/reports/radar_chart.html"):
    metrics = ["avg_clv","avg_spend","avg_loyalty","avg_returns"]
    labels  = ["CLV","Spend","Loyalty","Returns"]
    norm = cluster_stats[metrics].apply(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    )
    if HAS_PLOTLY:
        import plotly.graph_objects as go
        fig = go.Figure()
        for i, row in cluster_stats.iterrows():
            values = norm.iloc[i].tolist()
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]], theta=labels + [labels[0]],
                fill="toself", name=row["segment_name"],
            ))
        fig.update_layout(title="Segment Radar Chart", template="plotly_white")
        fig.write_html(out_path)
        logger.info(f"Radar chart saved -> {out_path}")
    else:
        logger.warning("Plotly not available - radar chart skipped.")
