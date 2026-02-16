"""Model registry and model-specific hyperparameter spaces."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor

SUPPORTED_MODELS = ("logreg", "rf")


def get_model(model_name: str, random_state: int = 42) -> Any:
    """Return an instantiated sklearn estimator based on name."""
    model_key = model_name.lower()

    if model_key == "logreg":
        return LogisticRegression(max_iter=2000, random_state=random_state)
    if model_key == "rf":
        return RandomForestClassifier(random_state=random_state)

    raise ValueError(
        f"Unsupported model '{model_name}'. Choose one of: {list(SUPPORTED_MODELS)}."
    )


def build_training_pipeline(model_name: str, random_state: int = 42) -> Pipeline:
    """Create end-to-end training pipeline (preprocessing + model)."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", get_model(model_name=model_name, random_state=random_state)),
        ]
    )


def get_param_grid(model_name: str) -> dict[str, list[Any]]:
    """Grid-search space using pipeline parameter names."""
    model_key = model_name.lower()

    if model_key == "logreg":
        return {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["lbfgs", "liblinear"],
        }

    if model_key == "rf":
        return {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
        }

    raise ValueError(
        f"Unsupported model '{model_name}'. Choose one of: {list(SUPPORTED_MODELS)}."
    )
