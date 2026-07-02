"""
executive_summary.py - AI-Powered Executive Summary Generator.
Calls the Anthropic Claude API to produce a boardroom-ready narrative
from KPI data. Falls back to a template summary if API is unavailable.
"""
import json
import os
import pandas as pd
from src.logger import get_logger

logger = get_logger(__name__)


def _template_summary(kpi_df: pd.DataFrame, model_name: str, sil_score: float) -> str:
    """Deterministic fallback summary when API is unavailable."""
    total_customers   = kpi_df["n_customers"].sum()
    total_revenue     = kpi_df["annual_revenue_est"].sum()
    total_risk        = kpi_df["revenue_at_risk"].sum()
    total_opportunity = kpi_df["retention_opportunity"].sum()
    best_seg          = kpi_df.sort_values("annual_revenue_est", ascending=False).iloc[0]
    riskiest_seg      = kpi_df.sort_values("revenue_at_risk", ascending=False).iloc[0]

    return f"""
EXECUTIVE SUMMARY - CUSTOMER SEGMENTATION ANALYSIS
====================================================
Prepared for: Executive Leadership Team
Classification: Internal - Confidential

OVERVIEW
--------
Our customer base of {total_customers:,} has been segmented into {len(kpi_df)} distinct
behavioral groups using {model_name} clustering (Silhouette Score: {sil_score:.3f}),
enabling precision targeting across marketing, retention, and acquisition functions.

KEY FINDINGS
------------
• Estimated Annual Revenue Base : ${total_revenue:,.0f}
• Revenue Currently at Risk     : ${total_risk:,.0f} ({total_risk/total_revenue*100:.1f}% of base)
• Recoverable via Retention     : ${total_opportunity:,.0f}

TOP SEGMENT - {best_seg['segment_name'].upper()}
  {int(best_seg['n_customers']):,} customers | Est. annual revenue: ${int(best_seg['annual_revenue_est']):,}

HIGHEST RISK - {riskiest_seg['segment_name'].upper()}
  ${int(riskiest_seg['revenue_at_risk']):,} revenue at risk
  Recommended immediate action: targeted win-back campaign.

STRATEGIC PRIORITIES
--------------------
1. Protect Champions revenue with VIP programme expansion.
2. Deploy retention campaign for At-Risk segment (ROI est. positive within 90 days).
3. Accelerate onboarding for New/Potential Loyalists to reduce early churn.

NEXT STEPS
----------
• Integrate segments into CRM for automated campaign triggers.
• Re-run segmentation quarterly; monitor segment migration rate.
• Establish KPI ownership per segment across Marketing and CX teams.
====================================================
""".strip()


def _api_summary(kpi_df: pd.DataFrame, model_name: str, sil_score: float) -> str:
    """Call Claude API for a narrative executive summary."""
    try:
        import anthropic
        client = anthropic.Anthropic()

        kpi_json = kpi_df[["segment_name","n_customers","annual_revenue_est",
                            "revenue_at_risk","marketing_roi_pct"]].to_dict(orient="records")

        prompt = f"""You are a senior business analyst presenting to a Fortune 500 executive team.
Write a concise, professional executive summary (max 350 words) for a customer segmentation project.

Model used: {model_name} | Silhouette Score: {sil_score:.3f}

Segment KPI Data:
{json.dumps(kpi_json, indent=2)}

Structure:
1. Overview (2 sentences)
2. Key Findings (3 bullet points with $ figures)
3. Strategic Priorities (3 numbered actions)
4. Recommended Next Steps (2 sentences)

Tone: boardroom-ready, data-driven, no jargon. Use specific numbers from the data."""

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    except Exception as e:
        logger.warning(f"API call failed ({e}). Using template summary.")
        return None


def generate_executive_summary(kpi_df: pd.DataFrame, model_name: str,
                                sil_score: float,
                                out_path: str = "outputs/reports/executive_summary.txt") -> str:
    summary = _api_summary(kpi_df, model_name, sil_score)
    if not summary:
        summary = _template_summary(kpi_df, model_name, sil_score)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(summary)
    logger.info("Executive summary saved.")
    return summary
