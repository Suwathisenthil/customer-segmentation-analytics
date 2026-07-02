"""
recommendation.py - Customer Recommendation Engine.
For each segment, recommends:
  1. Top products/actions (rule-based on segment profile)
  2. Similar customers within segment (cosine similarity, top-N)
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.logger import get_logger

logger = get_logger(__name__)

# Segment -> curated product/action recommendations
SEGMENT_RECOMMENDATIONS = {
    "Champions": [
        "Invite to VIP loyalty programme",
        "Early access to new product launches",
        "Premium bundle cross-sells",
        "Referral rewards - highest conversion probability",
    ],
    "Loyal Customers": [
        "Tier-upgrade incentives (e.g., Gold -> Platinum)",
        "Category expansion offers based on purchase history",
        "Subscription / auto-replenish upsell",
    ],
    "Potential Loyalists": [
        "Welcome loyalty card with first-purchase bonus points",
        "Personalised 'you might also like' email sequence",
        "Limited-time membership discount",
    ],
    "At-Risk": [
        "Win-back campaign: 20% off next purchase",
        "Personalised re-engagement email based on last category",
        "Customer satisfaction survey + service recovery offer",
    ],
    "Need Attention": [
        "Seasonal reactivation campaign",
        "Bundle deal on previously purchased categories",
        "Push notification with personalised discount",
    ],
    "Hibernating": [
        "Last-chance reactivation: major discount or freebie",
        "Survey to understand departure reason",
        "Consider suppression from paid channels to cut waste",
    ],
}


def add_product_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["recommended_actions"] = df["segment_name"].map(
        lambda s: " | ".join(SEGMENT_RECOMMENDATIONS.get(s, ["General engagement campaign"]))
    )
    logger.info("Product recommendations assigned per segment.")
    return df


def find_similar_customers(df: pd.DataFrame,
                            features: list[str],
                            top_n: int = 5,
                            sample_size: int = 200) -> pd.DataFrame:
    """
    For a random sample of customers, find top-N most similar
    customers within the same segment using cosine similarity.
    Returns a DataFrame of (customer_id, similar_customers[]).
    """
    avail = [f for f in features if f in df.columns]
    sample = df.sample(min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    records = []
    for seg in sample["segment_name"].unique():
        seg_df = sample[sample["segment_name"] == seg].reset_index(drop=True)
        if len(seg_df) < 2:
            continue
        X = seg_df[avail].fillna(0).values
        sims = cosine_similarity(X)
        np.fill_diagonal(sims, -1)           # exclude self

        for i, row in seg_df.iterrows():
            top_idx = np.argsort(sims[i])[::-1][:top_n]
            similar = seg_df.iloc[top_idx]["customer_id"].tolist()
            records.append({
                "customer_id":        row["customer_id"],
                "segment_name":       row["segment_name"],
                "similar_customers":  similar,
            })

    out = pd.DataFrame(records)
    out.to_csv("outputs/reports/similar_customers.csv", index=False)
    logger.info(f"Similar-customer lookup complete - {len(out):,} records.")
    return out
