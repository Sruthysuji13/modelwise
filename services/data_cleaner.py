"""Section 11 — Structural data cleaning.

Design note: actual missing-value imputation and encoding are handled inside the
sklearn preprocessing Pipeline (services/preprocessing.py) so statistics are fit
on the TRAIN split only and don't leak into test data. This module only handles
things safe to do on the whole dataset before splitting: removing exact duplicate
rows, dropping constant columns, and dropping obvious index columns.
"""
import pandas as pd


def clean_dataset(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict]:
    summary = {"actions": []}
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        summary["actions"].append(f"{removed} duplicate rows removed")

    constant_cols = [
        c for c in df.columns
        if c != target and df[c].nunique(dropna=True) <= 1
    ]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        summary["actions"].append(f"{len(constant_cols)} constant column(s) removed: {constant_cols}")

    index_like = [
        c for c in df.columns
        if c != target and df[c].nunique(dropna=True) == len(df)
        and c.lower() in ("id", "index", "unnamed: 0", "row_id", "rowid")
    ]
    if index_like:
        df = df.drop(columns=index_like)
        summary["actions"].append(f"{len(index_like)} index-like column(s) removed: {index_like}")

    if not summary["actions"]:
        summary["actions"].append("No structural issues found — dataset already clean")

    return df, summary