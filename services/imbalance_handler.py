"""Section 14 — Class imbalance detection (classification only)."""
import pandas as pd
from config import IMBALANCE_RATIO_THRESHOLD


def detect_class_imbalance(df: pd.DataFrame, target: str) -> dict:
    counts = df[target].value_counts(normalize=True)
    if counts.empty:
        return {"imbalanced": False}

    majority_share = counts.iloc[0]
    imbalanced = majority_share >= IMBALANCE_RATIO_THRESHOLD

    result = {
        "imbalanced": bool(imbalanced),
        "distribution": {str(k): round(v * 100, 2) for k, v in counts.items()},
    }
    if imbalanced:
        result["recommendation"] = (
            "class_weight balancing"
            if len(counts) <= 10
            else "class_weight balancing (SMOTE not recommended for high-cardinality targets)"
        )
        result["smote_eligible"] = True
    return result