"""data_loader.py - Load CSV or generate synthetic retail customer dataset."""
import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)


def load_or_generate(path: str = "data/customers.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.info(f"Dataset loaded from {path} - {df.shape[0]:,} rows")
        return df
    except FileNotFoundError:
        logger.warning(f"{path} not found. Generating synthetic dataset.")
        np.random.seed(42)
        n = 2000
        df = pd.DataFrame({
            "customer_id":     range(1, n + 1),
            "age":             np.random.randint(18, 70, n),
            "annual_income":   np.random.normal(60000, 25000, n).clip(15000, 200000),
            "spending_score":  np.random.randint(1, 101, n).astype(float),
            "purchase_freq":   np.random.randint(1, 52, n),
            "avg_order_value": np.random.normal(150, 80, n).clip(10, 800),
            "tenure_months":   np.random.randint(1, 121, n),
            "returns_rate":    np.random.uniform(0, 0.4, n).round(2),
            "gender":          np.random.choice(["M", "F"], n),
            "region":          np.random.choice(["North", "South", "East", "West"], n),
        })
        import os; os.makedirs("data", exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Synthetic dataset generated -> {path} ({n:,} rows)")
        return df
