"""Section 15 — Model selection per problem type, plus tuning search spaces."""
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from xgboost import XGBClassifier, XGBRegressor
from config import RANDOM_STATE


def get_classification_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def get_regression_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=RANDOM_STATE),
        "KNN Regressor": KNeighborsRegressor(),
        "SVR": SVR(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost Regressor": XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def get_models(problem_type: str, selected: list[str] | None = None) -> dict:
    models = get_classification_models() if problem_type == "classification" else get_regression_models()
    if selected:
        models = {k: v for k, v in models.items() if k in selected}
    return models


# Hyperparameter search spaces keyed by model name (used by both Optuna & RandomizedSearchCV paths)
TUNING_GRIDS = {
    "Decision Tree": {"max_depth": (3, 20), "min_samples_split": (2, 10)},
    "Decision Tree Regressor": {"max_depth": (3, 20), "min_samples_split": (2, 10)},
    "Random Forest": {"n_estimators": (100, 400), "max_depth": (5, 25), "min_samples_split": (2, 10)},
    "Random Forest Regressor": {"n_estimators": (100, 400), "max_depth": (5, 25), "min_samples_split": (2, 10)},
    "XGBoost": {"n_estimators": (100, 400), "max_depth": (3, 12), "learning_rate": (0.01, 0.3)},
    "XGBoost Regressor": {"n_estimators": (100, 400), "max_depth": (3, 12), "learning_rate": (0.01, 0.3)},
    "Gradient Boosting": {"n_estimators": (100, 300), "max_depth": (2, 8), "learning_rate": (0.01, 0.3)},
    "Gradient Boosting Regressor": {"n_estimators": (100, 300), "max_depth": (2, 8), "learning_rate": (0.01, 0.3)},
    "SVM": {"C": (0.1, 10.0), "gamma": (0.001, 1.0)},
    "SVR": {"C": (0.1, 10.0), "gamma": (0.001, 1.0)},
    "KNN": {"n_neighbors": (3, 15)},
    "KNN Regressor": {"n_neighbors": (3, 15)},
    "Logistic Regression": {"C": (0.01, 10.0)},
    "Ridge Regression": {"alpha": (0.01, 10.0)},
}