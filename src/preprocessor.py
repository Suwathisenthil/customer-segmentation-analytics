"""preprocessor.py - Data cleaning, feature engineering, scaling."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.logger import get_logger

logger = get_logger(__name__)

FEATURES = [
    "annual_income", "spending_score", "purchase_freq",
    "avg_order_value", "clv_proxy", "loyalty_index", "net_spend_score",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.copy()
    df.drop_duplicates(subset="customer_id", inplace=True)
    df.dropna(inplace=True)
    df = df[df["annual_income"] > 0]
    df = df[df["spending_score"].between(1, 100)]
    df = df.reset_index(drop=True)
    logger.info(f"Clean: {before:,} -> {len(df):,} rows")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clv_proxy"]       = df["avg_order_value"] * df["purchase_freq"]
    df["loyalty_index"]   = df["tenure_months"] / df["tenure_months"].max()
    df["net_spend_score"] = df["spending_score"] * (1 - df["returns_rate"])
    logger.info("Feature engineering complete: clv_proxy, loyalty_index, net_spend_score")
    return df


def scale(df: pd.DataFrame) -> tuple:
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    logger.info(f"Scaling complete - X shape: {X.shape}")
    return X, scaler
