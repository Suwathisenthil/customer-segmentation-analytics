"""
explainability.py - Segment Explainability Engine.
For each customer, generates a plain-English explanation of why they
belong to their assigned segment based on feature deviations from the
population mean (z-score based, no external libraries required).
"""
import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)

FEATURE_DESCRIPTIONS = {
    "annual_income":   ("income",          "high",  "low"),
    "spending_score":  ("spending level",  "high",  "low"),
    "purchase_freq":   ("purchase frequency","frequent","infrequent"),
    "avg_order_value": ("order size",      "large", "small"),
    "clv_proxy":       ("lifetime value",  "high",  "low"),
    "loyalty_index":   ("loyalty",         "strong","weak"),
    "net_spend_score": ("net spend",       "high",  "low"),
    "returns_rate":    ("return rate",     "low",   "high"),   # inverted: low returns = positive
    "churn_risk_score":("churn risk",      "low",   "high"),   # inverted
}
INVERTED = {"returns_rate", "churn_risk_score"}

FEATURES = list(FEATURE_DESCRIPTIONS.keys())


def _explain_customer(row: pd.Series, means: pd.Series, stds: pd.Series, top_n: int = 3) -> str:
    drivers = []
    for feat in FEATURES:
        if feat not in row.index or stds[feat] == 0:
            continue
        z = (row[feat] - means[feat]) / stds[feat]
        label, pos_adj, neg_adj = FEATURE_DESCRIPTIONS[feat]
        inverted = feat in INVERTED
        if abs(z) < 0.5:
            continue
        if (z > 0 and not inverted) or (z < 0 and inverted):
            drivers.append((abs(z), f"{pos_adj} {label}"))
        else:
            drivers.append((abs(z), f"{neg_adj} {label}"))

    drivers.sort(reverse=True)
    top = [d[1] for d in drivers[:top_n]]
    if not top:
        return "Profile is close to the average customer."
    return "Assigned due to: " + ", ".join(top) + "."


def generate_explanations(df: pd.DataFrame, sample_n: int = 500) -> pd.DataFrame:
    """
    Add an 'explanation' column to df.
    For large datasets, compute on a sample and merge back.
    """
    available = [f for f in FEATURES if f in df.columns]
    means = df[available].mean()
    stds  = df[available].std().replace(0, 1)

    # compute on full dataframe (vectorised per row)
    df = df.copy()
    df["explanation"] = df.apply(
        lambda row: _explain_customer(row, means, stds), axis=1
    )

    # Save sample explanations report
    cols_to_save = ["customer_id", "segment_name", "explanation"]
    for opt in ["churn_risk_band", "churn_risk_score"]:
        if opt in df.columns:
            cols_to_save.insert(2, opt)
    sample = df.sample(min(sample_n, len(df)), random_state=42)[
        cols_to_save
    ].reset_index(drop=True)
    sample.to_csv("outputs/reports/customer_explanations_sample.csv", index=False)
    logger.info(f"Explainability complete - sample of {len(sample)} saved.")
    return df
