"""
validator.py - Data quality validation & report generation.
Produces a structured validation report before any modelling.
"""
import pandas as pd
import json
from src.logger import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "customer_id":    {"dtype": "int",   "min": 1},
    "age":            {"dtype": "int",   "min": 18,  "max": 100},
    "annual_income":  {"dtype": "float", "min": 1000},
    "spending_score": {"dtype": "float", "min": 1,   "max": 100},
    "purchase_freq":  {"dtype": "int",   "min": 0},
    "avg_order_value":{"dtype": "float", "min": 0},
    "tenure_months":  {"dtype": "int",   "min": 0},
    "returns_rate":   {"dtype": "float", "min": 0,   "max": 1},
}


def validate(df: pd.DataFrame, out_path: str = "outputs/reports/data_validation.json") -> dict:
    report = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "schema_violations": [],
        "outlier_flags": {},
        "passed": True,
    }

    # Schema checks
    for col, rules in SCHEMA.items():
        if col not in df.columns:
            report["schema_violations"].append(f"MISSING COLUMN: {col}")
            report["passed"] = False
            continue
        if "min" in rules and (df[col] < rules["min"]).any():
            n = int((df[col] < rules["min"]).sum())
            report["schema_violations"].append(f"{col}: {n} values below min={rules['min']}")
        if "max" in rules and (df[col] > rules["max"]).any():
            n = int((df[col] > rules["max"]).sum())
            report["schema_violations"].append(f"{col}: {n} values above max={rules['max']}")

    # IQR outlier flags
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        if n_out > 0:
            report["outlier_flags"][col] = n_out

    # Summary
    total_issues = len(report["schema_violations"]) + int(report["duplicate_rows"]) + \
                   sum(v for v in report["missing_values"].values() if v > 0)
    report["total_issues"] = total_issues

    import os; os.makedirs("outputs/reports", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    status = "PASSED " if report["passed"] and total_issues == 0 else "WARNING ⚠"
    logger.info(f"Data validation complete - {status} | Issues: {total_issues}")
    return report


def print_validation_summary(report: dict):
    print("\n" + "="*55)
    print("  DATA VALIDATION REPORT")
    print("="*55)
    print(f"  Rows × Cols   : {report['shape'][0]:,} × {report['shape'][1]}")
    print(f"  Duplicate Rows: {report['duplicate_rows']}")
    print(f"  Missing Values: {sum(report['missing_values'].values())}")
    print(f"  Schema Issues : {len(report['schema_violations'])}")
    print(f"  Outlier Flags : {len(report['outlier_flags'])} columns")
    print(f"  Total Issues  : {report['total_issues']}")
    status = "PASSED " if report["passed"] and report["total_issues"] == 0 else "WARNING ⚠"
    print(f"  Status        : {status}")
    print("="*55 + "\n")
