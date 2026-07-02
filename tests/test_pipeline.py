"""
tests/test_pipeline.py - Unit tests for all core pipeline modules.
Run: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import numpy as np

# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def raw_df():
    np.random.seed(0)
    n = 100
    return pd.DataFrame({
        "customer_id":     range(1, n+1),
        "age":             np.random.randint(18, 65, n),
        "annual_income":   np.random.normal(60000, 20000, n).clip(15000, 180000),
        "spending_score":  np.random.randint(1, 101, n).astype(float),
        "purchase_freq":   np.random.randint(1, 50, n),
        "avg_order_value": np.random.normal(150, 60, n).clip(10, 600),
        "tenure_months":   np.random.randint(1, 100, n),
        "returns_rate":    np.random.uniform(0, 0.35, n).round(2),
        "gender":          np.random.choice(["M","F"], n),
        "region":          np.random.choice(["North","South"], n),
    })


@pytest.fixture
def cfg():
    return {
        "model": {
            "kmeans":       {"k_range": [2,5], "n_init": 5, "random_state": 42},
            "dbscan":       {"eps": 0.8, "min_samples": 5},
            "agglomerative":{"linkage": "ward"},
            "gmm":          {"covariance_type": "full"},
        },
        "business": {
            "avg_acquisition_cost": 45.0,
            "avg_monthly_revenue_per_customer": 120.0,
            "marketing_campaign_cost": 5000.0,
            "campaign_conversion_rate": 0.12,
            "churn_revenue_loss_per_customer": 800.0,
            "retention_campaign_success_rate": 0.30,
        }
    }


# ── Preprocessor ──────────────────────────────────────────────────────────────
class TestPreprocessor:
    def test_clean_removes_duplicates(self, raw_df):
        from src.preprocessor import clean
        dup = pd.concat([raw_df, raw_df.iloc[:5]], ignore_index=True)
        cleaned = clean(dup)
        assert cleaned.duplicated(subset="customer_id").sum() == 0

    def test_clean_filters_invalid_income(self, raw_df):
        from src.preprocessor import clean
        raw_df.loc[0, "annual_income"] = -999
        cleaned = clean(raw_df)
        assert (cleaned["annual_income"] <= 0).sum() == 0

    def test_engineer_features_adds_columns(self, raw_df):
        from src.preprocessor import clean, engineer_features
        df = engineer_features(clean(raw_df))
        for col in ["clv_proxy","loyalty_index","net_spend_score"]:
            assert col in df.columns, f"Missing: {col}"

    def test_clv_proxy_positive(self, raw_df):
        from src.preprocessor import clean, engineer_features
        df = engineer_features(clean(raw_df))
        assert (df["clv_proxy"] >= 0).all()

    def test_loyalty_index_bounded(self, raw_df):
        from src.preprocessor import clean, engineer_features
        df = engineer_features(clean(raw_df))
        assert df["loyalty_index"].between(0, 1).all()

    def test_scale_returns_ndarray(self, raw_df):
        from src.preprocessor import clean, engineer_features, scale
        df = engineer_features(clean(raw_df))
        X, scaler = scale(df)
        assert X.shape[0] == len(df)
        assert abs(X.mean()) < 0.5   # roughly centered


# ── Validator ─────────────────────────────────────────────────────────────────
class TestValidator:
    def test_valid_data_passes(self, raw_df, tmp_path):
        from src.validator import validate
        report = validate(raw_df, out_path=str(tmp_path / "val.json"))
        assert report["duplicate_rows"] == 0

    def test_missing_column_flagged(self, raw_df, tmp_path):
        from src.validator import validate
        bad = raw_df.drop(columns=["spending_score"])
        report = validate(bad, out_path=str(tmp_path / "val2.json"))
        assert any("spending_score" in v for v in report["schema_violations"])

    def test_out_of_range_flagged(self, raw_df, tmp_path):
        from src.validator import validate
        raw_df.loc[0, "spending_score"] = 200   # above max=100
        report = validate(raw_df, out_path=str(tmp_path / "val3.json"))
        assert any("spending_score" in v for v in report["schema_violations"])


# ── Business KPI ──────────────────────────────────────────────────────────────
class TestBusinessKPI:
    def _get_segmented_df(self, raw_df):
        from src.preprocessor import clean, engineer_features
        df = engineer_features(clean(raw_df))
        df["cluster"] = np.random.randint(0, 3, len(df))
        df["segment_name"] = df["cluster"].map({0:"Champions",1:"At-Risk",2:"Hibernating"})
        return df

    def test_churn_risk_score_bounded(self, raw_df):
        from src.business_kpi import compute_churn_risk
        from src.preprocessor import clean, engineer_features
        df = engineer_features(clean(raw_df))
        df = compute_churn_risk(df)
        assert df["churn_risk_score"].between(0, 1).all()

    def test_churn_risk_band_categories(self, raw_df):
        from src.business_kpi import compute_churn_risk
        from src.preprocessor import clean, engineer_features
        df = engineer_features(clean(raw_df))
        df = compute_churn_risk(df)
        assert set(df["churn_risk_band"].dropna().unique()).issubset({"Low","Medium","High"})

    def test_revenue_at_risk_nonnegative(self, raw_df, cfg):
        from src.business_kpi import compute_churn_risk, compute_business_impact
        from src.preprocessor import clean, engineer_features
        df = engineer_features(clean(raw_df))
        df = compute_churn_risk(df)
        df["segment_name"] = "Champions"
        kpi = compute_business_impact(df, cfg)
        assert (kpi["revenue_at_risk"] >= 0).all()

    def test_marketing_roi_computed(self, raw_df, cfg):
        from src.business_kpi import compute_churn_risk, compute_business_impact, compute_marketing_roi
        from src.preprocessor import clean, engineer_features
        df = engineer_features(clean(raw_df))
        df = compute_churn_risk(df)
        df["segment_name"] = "Champions"
        seg_kpi = compute_business_impact(df, cfg)
        roi = compute_marketing_roi(seg_kpi, cfg)
        assert "marketing_roi_pct" in roi.columns


# ── Explainability ────────────────────────────────────────────────────────────
class TestExplainability:
    def test_explanation_column_exists(self, raw_df, tmp_path, monkeypatch):
        from src.preprocessor import clean, engineer_features
        from src.business_kpi import compute_churn_risk
        import src.explainability as exp_mod
        monkeypatch.setattr(exp_mod, "generate_explanations",
                            lambda df, **kw: df.assign(explanation="test"))
        df = engineer_features(clean(raw_df))
        df = compute_churn_risk(df)
        df["segment_name"] = "Champions"
        df = exp_mod.generate_explanations(df, sample_n=10)
        assert "explanation" in df.columns

    def test_no_null_explanations(self, raw_df, tmp_path, monkeypatch):
        import src.explainability as exp_mod
        from src.preprocessor import clean, engineer_features
        from src.business_kpi import compute_churn_risk
        monkeypatch.setattr(exp_mod, "generate_explanations",
                            lambda df, **kw: df.assign(explanation="ok"))
        df = engineer_features(clean(raw_df))
        df = compute_churn_risk(df)
        df["segment_name"] = "Champions"
        df = exp_mod.generate_explanations(df, sample_n=10)
        assert df["explanation"].notnull().all()


# ── Recommendation ────────────────────────────────────────────────────────────
class TestRecommendation:
    def test_recommendations_assigned(self, raw_df):
        from src.recommendation import add_product_recommendations
        from src.preprocessor import clean, engineer_features
        df = engineer_features(clean(raw_df))
        df["segment_name"] = "Champions"
        df = add_product_recommendations(df)
        assert "recommended_actions" in df.columns
        assert df["recommended_actions"].notnull().all()

    def test_unknown_segment_handled(self, raw_df):
        from src.recommendation import add_product_recommendations
        from src.preprocessor import clean, engineer_features
        df = engineer_features(clean(raw_df))
        df["segment_name"] = "UnknownSegment"
        df = add_product_recommendations(df)
        assert df["recommended_actions"].str.len().gt(0).all()
