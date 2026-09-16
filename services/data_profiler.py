"""Sections 6, 7, 9, 10 — Understanding data, quality score, target & problem type detection."""
import numpy as np
import pandas as pd
import warnings


def profile_dataset(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    date_cols = [c for c in df.columns if _looks_like_date(df[c])]

    missing_by_col = df.isna().sum()
    total_missing = int(missing_by_col.sum())
    missing_pct = round(total_missing / (df.shape[0] * df.shape[1]) * 100, 2) if df.size else 0.0

    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round(duplicate_rows / len(df) * 100, 2) if len(df) else 0.0

    numeric_summary = df[numeric_cols].describe().to_dict() if numeric_cols else {}

    categorical_summary = {}
    for c in categorical_cols:
        vc = df[c].value_counts(dropna=True)
        if len(vc):
            categorical_summary[c] = {
                "n_categories": int(df[c].nunique(dropna=True)),
                "most_frequent": str(vc.index[0]),
                "frequency": int(vc.iloc[0]),
            }

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "bool_cols": bool_cols,
        "date_cols": date_cols,
        "missing_total": total_missing,
        "missing_pct": missing_pct,
        "missing_by_col": missing_by_col[missing_by_col > 0].to_dict(),
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": duplicate_pct,
        "unique_counts": df.nunique(dropna=True).to_dict(),
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
    }


def _looks_like_date(series: pd.Series) -> bool:
    if series.dtype == "datetime64[ns]":
        return True
    if series.dtype != object:
        return False
    sample = series.dropna().head(20)
    if sample.empty:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd.to_datetime(sample, errors="raise")
        return True
    except Exception:
        return False


def compute_data_quality_score(df: pd.DataFrame, profile: dict) -> dict:
    """Simple, transparent 0-100 quality score."""
    score = 100
    issues = []

    if profile["missing_pct"] > 0:
        penalty = min(30, profile["missing_pct"] * 2)
        score -= penalty
        issues.append(f"{profile['missing_pct']}% missing values")

    if profile["duplicate_pct"] > 0:
        penalty = min(15, profile["duplicate_pct"])
        score -= penalty
        issues.append(f"{profile['duplicate_pct']}% duplicate rows")

    constant_cols = [c for c, n in profile["unique_counts"].items() if n <= 1]
    if constant_cols:
        score -= min(15, 5 * len(constant_cols))
        issues.append(f"{len(constant_cols)} constant column(s)")

    n_rows = profile["rows"] or 1
    high_card_cols = [
        c for c in profile["categorical_cols"]
        if profile["unique_counts"].get(c, 0) / n_rows > 0.9
    ]
    if high_card_cols:
        score -= min(10, 5 * len(high_card_cols))
        issues.append(f"{len(high_card_cols)} high-cardinality column(s)")

    # crude outlier signal: numeric columns with extreme skew
    outlier_flag = False
    for c in profile["numeric_cols"]:
        col = df[c].dropna()
        if len(col) > 10:
            skew = col.skew()
            if abs(skew) > 3:
                outlier_flag = True
                break
    if outlier_flag:
        score -= 10
        issues.append("Moderate outliers detected")

    score = max(0, round(score))
    return {"score": score, "issues": issues}


def suggest_target_column(df: pd.DataFrame) -> dict:
    """Heuristic target suggestion: prefer low-cardinality columns near the end,
    or columns literally named like a target."""
    candidates = []
    name_hints = ("target", "label", "class", "churn", "outcome", "y", "result")

    for col in df.columns:
        lower = col.lower()
        n_unique = df[col].nunique(dropna=True)
        ratio = n_unique / len(df) if len(df) else 1

        score = 0
        if any(h in lower for h in name_hints):
            score += 3
        if ratio < 0.5:
            score += 1
        if col == df.columns[-1]:
            score += 1
        candidates.append((col, score, ratio))

    candidates.sort(key=lambda x: (-x[1], x[2]))
    best = candidates[0]
    confidence = "High" if best[1] >= 3 else ("Medium" if best[1] >= 1 else "Low")
    return {"suggested_target": best[0], "confidence": confidence}


def detect_problem_type(df: pd.DataFrame, target: str) -> dict:
    """Sections 10 — classification vs regression, with a plain-language reason."""
    series = df[target].dropna()
    n_unique = series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(series)

    if not is_numeric:
        return {
            "problem_type": "classification",
            "reason": "The selected target contains non-numeric categorical values, "
                      "so this dataset is treated as a classification problem.",
        }

    # numeric target: few distinct integer-like values => classification
    looks_discrete = (
        n_unique <= 20
        and np.all(np.equal(np.mod(series, 1), 0))
    )
    if looks_discrete:
        return {
            "problem_type": "classification",
            "reason": f"The selected target is numeric but has only {n_unique} discrete "
                      "values, so this dataset is treated as a classification problem.",
        }

    return {
        "problem_type": "regression",
        "reason": "The selected target contains continuous numeric values, "
                  "so this dataset is treated as a regression problem.",
    }