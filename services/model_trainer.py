"""Sections 16-19 — Training configuration, training loop, CV, hyperparameter tuning."""
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.special import expit
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold, cross_val_score, RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline

from config import RANDOM_STATE, TEST_SIZE, CV_FOLDS, TUNING_N_ITER_RANDOMSEARCH
from services.preprocessing import build_preprocessor
from services.model_registry import get_models, TUNING_GRIDS
from services.evaluator import compute_classification_metrics, compute_regression_metrics

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# Models with roughly O(n^2)+ training cost — impractical on large datasets,
# especially under repeated hyperparameter-tuning fits.
LARGE_DATASET_ROW_LIMIT_SVM = 20000


@dataclass
class TrainingConfig:
    test_size: float = TEST_SIZE
    cv_folds: int = CV_FOLDS
    tune_hyperparameters: bool = False
    random_state: int = RANDOM_STATE
    selected_models: list = field(default_factory=list)  # empty = all


def _get_classification_scores(pipeline, X_test):
    """Return a predict_proba-shaped (n_samples, n_classes) array for metrics,
    falling back to a sigmoid-squashed decision_function for models like SVM
    that don't expose calibrated probabilities without a heavy cost."""
    if hasattr(pipeline, "predict_proba"):
        try:
            return pipeline.predict_proba(X_test)
        except Exception:
            pass

    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X_test)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            # binary case: squash to (0,1) and build a 2-column pseudo-proba
            pos = expit(scores)
            return np.column_stack([1 - pos, pos])
        # multiclass: softmax-style normalization
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    return None


def train_models(df: pd.DataFrame, target: str, problem_type: str, config: TrainingConfig) -> dict:
    X = df.drop(columns=[target])
    y = df[target]

    stratify = y if problem_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=stratify
    )

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(
        pd.concat([X_train, y_train.rename(target)], axis=1), target
    )

    models = get_models(problem_type, config.selected_models or None)
    results = []

    for name, estimator in models.items():
        result = {"model_name": name, "status": "success"}

        if name in ("SVM", "SVR") and len(X_train) > LARGE_DATASET_ROW_LIMIT_SVM:
            result["status"] = "skipped"
            result["error"] = (
                f"Skipped: {len(X_train)} training rows exceeds the practical limit "
                f"for {name} ({LARGE_DATASET_ROW_LIMIT_SVM}). Consider Random Forest "
                f"or XGBoost instead for datasets this size."
            )
            results.append(result)
            continue

        try:
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

            # cross-validation on TRAIN split only
            cv_splitter = (
                StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
                if problem_type == "classification"
                else KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
            )
            scoring = "f1_weighted" if problem_type == "classification" else "r2"
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring=scoring, n_jobs=-1)
            result["cv"] = {
                "folds": [round(s, 4) for s in cv_scores],
                "mean": round(float(np.mean(cv_scores)), 4),
                "std": round(float(np.std(cv_scores)), 4),
                "scoring": scoring,
            }

            best_params = None
            if config.tune_hyperparameters and name in TUNING_GRIDS:
                pipeline, best_params = _tune(
                    pipeline, name, X_train, y_train, cv_splitter, scoring, problem_type
                )
                tuned_cv_scores = cross_val_score(
                    pipeline, X_train, y_train, cv=cv_splitter, scoring=scoring, n_jobs=-1
                )
                result["cv_after_tuning"] = {
                    "mean": round(float(np.mean(tuned_cv_scores)), 4),
                    "std": round(float(np.std(tuned_cv_scores)), 4),
                }
            result["best_hyperparameters"] = best_params

            t0 = time.time()
            pipeline.fit(X_train, y_train)
            train_time = round(time.time() - t0, 3)

            t0 = time.time()
            y_pred = pipeline.predict(X_test)
            predict_time = round(time.time() - t0, 3)

            if problem_type == "classification":
                y_proba = _get_classification_scores(pipeline, X_test)
                result["metrics"] = compute_classification_metrics(y_test, y_pred, y_proba)
            else:
                result["metrics"] = compute_regression_metrics(y_test, y_pred)

            result["training_time"] = train_time
            result["prediction_time"] = predict_time
            result["fitted_pipeline"] = pipeline  # kept in-memory; Phase 3 will persist via joblib

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)

        results.append(result)

    return {
        "results": results,
        "X_test": X_test,
        "y_test": y_test,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def _tune(pipeline, name, X_train, y_train, cv_splitter, scoring, problem_type):
    grid = TUNING_GRIDS[name]

    if HAS_OPTUNA:
        def objective(trial):
            params = {}
            for param, bounds in grid.items():
                low, high = bounds
                if isinstance(low, int) and isinstance(high, int):
                    params[f"model__{param}"] = trial.suggest_int(param, low, high)
                else:
                    params[f"model__{param}"] = trial.suggest_float(param, low, high)
            pipeline.set_params(**params)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring=scoring, n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=25, show_progress_bar=False)
        best_params = {f"model__{k}": v for k, v in study.best_params.items()}
        pipeline.set_params(**best_params)
        return pipeline, study.best_params

    # Fallback: RandomizedSearchCV
    param_distributions = {}
    for param, bounds in grid.items():
        low, high = bounds
        if isinstance(low, int) and isinstance(high, int):
            param_distributions[f"model__{param}"] = list(range(low, high + 1))
        else:
            param_distributions[f"model__{param}"] = list(np.linspace(low, high, 10))

    search = RandomizedSearchCV(
        pipeline, param_distributions, n_iter=TUNING_N_ITER_RANDOMSEARCH,
        cv=cv_splitter, scoring=scoring, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return search.best_estimator_, best_params