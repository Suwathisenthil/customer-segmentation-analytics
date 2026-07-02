"""eda.py - Exploratory Data Analysis plots."""
import matplotlib.pyplot as plt
import os
from src.logger import get_logger

logger = get_logger(__name__)


def run_eda(df, out_dir: str = "outputs/figures"):
    os.makedirs(out_dir, exist_ok=True)

    # Distribution grid
    cols = ["annual_income", "spending_score", "purchase_freq", "avg_order_value"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Customer Feature Distributions", fontsize=14, fontweight="bold")
    for ax, col in zip(axes.flatten(), cols):
        ax.hist(df[col], bins=30, color="#4C72B0", edgecolor="white")
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/distributions.png", dpi=150)
    plt.close()

    # Income vs Spending coloured by CLV
    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(
        df["annual_income"], df["spending_score"],
        c=df["clv_proxy"], cmap="viridis", alpha=0.6, s=15,
    )
    plt.colorbar(scatter, ax=ax, label="CLV Proxy ($)")
    ax.set_xlabel("Annual Income ($)")
    ax.set_ylabel("Spending Score")
    ax.set_title("Income vs Spending Score (coloured by CLV Proxy)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/income_vs_spending.png", dpi=150)
    plt.close()

    logger.info(f"EDA figures saved to {out_dir}/")
