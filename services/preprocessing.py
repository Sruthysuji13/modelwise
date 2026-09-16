"""Section 12 — Preprocessing pipeline (imputation + scaling + encoding)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(df: pd.DataFrame, target: str) -> tuple[ColumnTransformer, list, list]:
    feature_df = df.drop(columns=[target])
    numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_cols, categorical_cols