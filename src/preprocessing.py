"""Data preparation utilities for the PIMA diabetes project.

هذا الملف مسؤول عن تجهيز البيانات بشكل يمنع data leakage:
- التقسيم Train/Test يتم قبل أي `fit` للتحويلات.
- الـimputation والـscaling داخل sklearn Pipeline.
- لا يتم عمل fit على test data إطلاقًا.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "Outcome"

NUMERIC_FEATURES: Sequence[str] = (
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
)


def split_features_target(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract `X, y` with strict column validation."""
    missing_columns = {target_column, *NUMERIC_FEATURES} - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    X = data.loc[:, NUMERIC_FEATURES].copy()
    y = data.loc[:, target_column].copy()
    return X, y


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data before any preprocessing to avoid leakage.

    Important:
    - This function should be called before pipeline fitting.
    - Preprocessing steps are fitted only on `X_train` through the Pipeline.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessor(
    numeric_features: Iterable[str] = NUMERIC_FEATURES,
) -> ColumnTransformer:
    """Build leakage-safe preprocessing graph for numeric features.

    التنفيذ داخل Pipeline يضمن أن:
    1) `SimpleImputer` يعمل fit على train folds فقط.
    2) `StandardScaler` يعمل fit بعد التقسيم (على train فقط).
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, list(numeric_features))],
        remainder="drop",
        verbose_feature_names_out=False,
    )
