"""Train and compare leakage-safe diabetes classifiers."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from models import SUPPORTED_MODELS, build_training_pipeline, get_param_grid
from preprocessing import make_train_test_split, split_features_target

DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/diabetes.csv"))
    parser.add_argument("--model", choices=list(SUPPORTED_MODELS), default="logreg")
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.pkl"))
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("artifacts/model_comparison.csv"),
    )
    parser.add_argument("--random-state", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--metrics-json", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("results/metrics.csv"))
    parser.add_argument("--run-config", type=Path, default=Path("results/run_config.json"))
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable GridSearchCV on train split only.",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help=(
            "Run StratifiedKFold comparison across all supported models "
            "with ROC-AUC and F1, and save a single results table."
        ),
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """Fix global random states for reproducibility (numpy, random, sklearn usage)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # sklearn reproducibility is controlled by passing random_state=seed
    # to splitters/models, which this project does consistently.


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int,
    random_state: int,
) -> pd.DataFrame:
    """Compare all supported models with StratifiedKFold on train split only."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = {"roc_auc": "roc_auc", "f1": "f1"}

    rows: list[dict[str, float | str]] = []
    for model_name in SUPPORTED_MODELS:
        pipeline = build_training_pipeline(model_name, random_state=random_state)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        rows.append(
            {
                "model": model_name,
                "cv_strategy": f"StratifiedKFold(n_splits={cv_splits}, shuffle=True)",
                "roc_auc_mean": float(scores["test_roc_auc"].mean()),
                "roc_auc_std": float(scores["test_roc_auc"].std()),
                "f1_mean": float(scores["test_f1"].mean()),
                "f1_std": float(scores["test_f1"].std()),
            }
        )

    return pd.DataFrame(rows).sort_values(by=["roc_auc_mean", "f1_mean"], ascending=False)


def save_metrics(metrics: dict[str, float | str], json_path: Path, csv_path: Path) -> None:
    """Persist metrics to JSON and single-row CSV."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)


def save_run_config(args: argparse.Namespace, selected_hyperparameters: dict | None) -> None:
    """Save experiment configuration (model + hyperparams + preprocessing choices)."""
    config = {
        "seed": {
            "default": DEFAULT_SEED,
            "active": args.random_state,
            "numpy": args.random_state,
            "random": args.random_state,
            "sklearn_random_state": args.random_state,
        },
        "data_path": str(args.data),
        "model_name": args.model,
        "selected_hyperparameters": selected_hyperparameters,
        "test_size": args.test_size,
        "cv": args.cv,
        "tune": args.tune,
        "compare_models": args.compare_models,
        "preprocessing": {
            "split_first": True,
            "imputation": {"strategy": "median"},
            "scaling": "StandardScaler",
            "feature_columns": [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
            ],
        },
    }

    args.run_config.parent.mkdir(parents=True, exist_ok=True)
    args.run_config.write_text(json.dumps(config, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    set_global_seed(args.random_state)

    data = pd.read_csv(args.data)
    X, y = split_features_target(data)

    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    if args.compare_models:
        comparison_df = compare_models(
            X_train=X_train,
            y_train=y_train,
            cv_splits=args.cv,
            random_state=args.random_state,
        )
        args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(args.comparison_output, index=False)
        print("\nModel comparison (train-only CV):")
        print(comparison_df.to_string(index=False))
        print(f"\nSaved comparison table to: {args.comparison_output}")

    pipeline = build_training_pipeline(args.model, random_state=args.random_state)

    if args.tune:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=get_param_grid(args.model),
            scoring="roc_auc",
            cv=args.cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        trained_pipeline = search.best_estimator_
        selected_hyperparameters = dict(search.best_params_)
        print(f"Best params: {search.best_params_}")
        print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    else:
        trained_pipeline = pipeline.fit(X_train, y_train)
        selected_hyperparameters = trained_pipeline.named_steps["model"].get_params(deep=False)

    y_pred = trained_pipeline.predict(X_test)
    y_prob = trained_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "split": "holdout",
        "model": args.model,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    print(f"Holdout Accuracy: {metrics['accuracy']:.4f}")
    print(f"Holdout F1: {metrics['f1']:.4f}")
    print(f"Holdout ROC-AUC: {metrics['roc_auc']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pickle.dumps(trained_pipeline))
    print(f"Saved trained pipeline to: {args.output}")

    save_metrics(metrics, args.metrics_json, args.metrics_csv)
    print(f"Saved metrics JSON to: {args.metrics_json}")
    print(f"Saved metrics CSV to: {args.metrics_csv}")

    save_run_config(args, selected_hyperparameters=selected_hyperparameters)
    print(f"Saved run config to: {args.run_config}")


if __name__ == "__main__":
    main()
