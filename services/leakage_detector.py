"""Section 13 — Data leakage detection (heuristic, not guaranteed)."""
import pandas as pd
import numpy as np
from config import LEAKAGE_CORR_THRESHOLD


def detect_leakage(df: pd.DataFrame, target: str, problem_type: str) -> list[dict]:
    suspicious = []
    y = df[target]

    for col in df.columns:
        if col == target:
            continue

        reason = None
        if problem_type == "regression" and pd.api.types.is_numeric_dtype(df[col]):
            corr = df[[col, target]].dropna().corr().iloc[0, 1]
            if abs(corr) >= LEAKAGE_CORR_THRESHOLD:
                reason = f"Extremely strong correlation with target ({corr:.3f})."

        elif problem_type == "classification":
            # column that near-perfectly determines the target (e.g. duplicate encoding)
            try:
                combined = pd.crosstab(df[col], y)
                if combined.shape[0] > 0:
                    max_purity = (combined.max(axis=1) / combined.sum(axis=1)).mean()
                    if max_purity >= 0.98 and df[col].nunique() > 1:
                        reason = "Column values almost perfectly determine the target class."
            except Exception:
                pass

        # name-based hints for future/derived-outcome columns
        lower = col.lower()
        if reason is None and any(h in lower for h in ("final", "outcome_", "result_after", "post_")):
            reason = "Column name suggests it may be derived from or after the outcome."

        if reason:
            suspicious.append({
                "column": col,
                "reason": reason,
                "recommendation": "Review this feature before training; consider excluding it.",
            })

    return suspicious