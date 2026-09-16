"""Section 38 — Data science safety checks. Warns but does not block unless truly untrainable."""
import pandas as pd
from config import MIN_ROWS_FOR_TRAINING, MIN_ROWS_FOR_STABLE_CV


def run_safety_checks(df: pd.DataFrame, target: str, problem_type: str) -> dict:
    warnings = []
    blocking = False

    n_rows = len(df)
    if n_rows < MIN_ROWS_FOR_TRAINING:
        warnings.append(f"Only {n_rows} rows — dataset is too small to train reliably.")
        blocking = True
    elif n_rows < MIN_ROWS_FOR_STABLE_CV:
        warnings.append(f"Only {n_rows} rows — model evaluation may be unstable.")

    if df[target].isna().all():
        warnings.append("Target column has no valid values.")
        blocking = True
    elif df[target].isna().any():
        warnings.append(f"{df[target].isna().sum()} missing target values will be dropped before training.")

    if problem_type == "classification":
        n_classes = df[target].nunique(dropna=True)
        if n_classes < 2:
            warnings.append("Target has only one class — classification is not possible.")
            blocking = True

    high_card_cats = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c != target and df[c].nunique(dropna=True) / n_rows > 0.9
    ]
    if high_card_cats:
        warnings.append(f"High-cardinality categorical column(s) detected: {high_card_cats}")

    return {"warnings": warnings, "blocking": blocking}