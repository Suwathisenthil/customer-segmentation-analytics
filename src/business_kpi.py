"""
business_kpi.py - Business KPI Calculator
Computes: CLV, churn risk score, revenue at risk, marketing ROI.
All dollar values driven by config/config.yaml.
"""
import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)


# ── Churn Risk Score ──────────────────────────────────────────────────────────
def compute_churn_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based churn risk score [0–1] per customer.
    Driven by: recency proxy (low spend), high returns, low loyalty.
    """
    df = df.copy()
    score  = np.zeros(len(df))
    score += (df["spending_score"]  < 30).astype(float) * 0.35
    score += (df["returns_rate"]    > 0.25).astype(float) * 0.25
    score += (df["loyalty_index"]   < 0.20).astype(float) * 0.25
    score += (df["purchase_freq"]   < 5).astype(float)  * 0.15
    df["churn_risk_score"] = score.round(3)
    df["churn_risk_band"]  = pd.cut(
        df["churn_risk_score"],
        bins=[-0.01, 0.25, 0.55, 1.01],
        labels=["Low", "Medium", "High"]
    )
    logger.info(f"Churn risk computed | High-risk: {(df['churn_risk_band']=='High').sum():,}")
    return df


# ── Business Impact Estimation ─────────────────────────────────────────────────
def compute_business_impact(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Estimate per-segment annual revenue, revenue at risk, and acquisition cost."""
    biz = cfg["business"]
    segment_kpi = (
        df.groupby("segment_name")
          .agg(
              n_customers=("customer_id", "count"),
              avg_clv=("clv_proxy", "mean"),
              avg_churn_risk=("churn_risk_score", "mean"),
              high_risk_count=("churn_risk_band", lambda x: (x == "High").sum()),
          )
          .reset_index()
    )

    segment_kpi["annual_revenue_est"]   = (segment_kpi["n_customers"] *
                                           biz["avg_monthly_revenue_per_customer"] * 12).round(0)
    segment_kpi["revenue_at_risk"]      = (segment_kpi["high_risk_count"] *
                                           biz["churn_revenue_loss_per_customer"]).round(0)
    segment_kpi["retention_opportunity"]= (segment_kpi["revenue_at_risk"] *
                                           biz["retention_campaign_success_rate"]).round(0)
    segment_kpi["acquisition_cost"]     = (segment_kpi["n_customers"] *
                                           biz["avg_acquisition_cost"]).round(0)

    logger.info(f"Total revenue at risk: ${segment_kpi['revenue_at_risk'].sum():,.0f}")
    return segment_kpi


# ── Marketing ROI Estimation ───────────────────────────────────────────────────
def compute_marketing_roi(segment_kpi: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Estimate ROI per segment for a targeted re-engagement campaign.
    Formula: ROI = (recovered_revenue - campaign_cost) / campaign_cost
    """
    biz = cfg["business"]
    roi = segment_kpi.copy()
    roi["campaign_cost"]       = biz["marketing_campaign_cost"]
    roi["expected_conversions"]= (roi["n_customers"] * biz["campaign_conversion_rate"]).round(0)
    roi["recovered_revenue"]   = (roi["expected_conversions"] *
                                  biz["avg_monthly_revenue_per_customer"] * 3).round(0)   # 3-mo horizon
    roi["marketing_roi_pct"]   = ((roi["recovered_revenue"] - roi["campaign_cost"]) /
                                   roi["campaign_cost"] * 100).round(1)
    roi["roi_band"]            = pd.cut(roi["marketing_roi_pct"],
                                        bins=[-9999, 0, 100, 9999],
                                        labels=["Negative", "Moderate", "Strong"])
    logger.info("Marketing ROI estimation complete.")
    return roi


# ── Consolidated KPI Table ─────────────────────────────────────────────────────
def build_kpi_table(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = compute_churn_risk(df)
    seg_kpi = compute_business_impact(df, cfg)
    kpi_full = compute_marketing_roi(seg_kpi, cfg)
    kpi_full.to_csv("outputs/reports/business_kpi.csv", index=False)
    return df, kpi_full
