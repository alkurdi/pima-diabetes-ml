"""Train and compare leakage-safe diabetes classifiers with per-run tracking."""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from experiment_tracking import (
    now_iso,
    resolve_train_run_paths,
    save_json,
    save_single_row_csv,
)
from models import SUPPORTED_MODELS, build_training_pipeline, get_param_grid
from preprocessing import make_train_test_split, split_features_target

DEFAULT_SEED = 42
DEFAULT_DATA_PATH = "data/diabetes.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", choices=list(SUPPORTED_MODELS), default="logreg")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv", type=int, default=5)
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
            "with ROC-AUC and F1."
        ),
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def log_message(message: str, log_file) -> None:
    print(message)
    log_file.write(f"{message}\n")
    log_file.flush()


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int,
    seed: int,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    scoring = {"roc_auc": "roc_auc", "f1": "f1"}

    rows: list[dict[str, float | str]] = []
    for model_name in SUPPORTED_MODELS:
        pipeline = build_training_pipeline(model_name, random_state=seed)
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


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    run_paths = resolve_train_run_paths(args.run_id)

    with run_paths.train_log.open("w", encoding="utf-8") as train_log:
        log_message(f"Starting training for run_id={run_paths.run_id}", train_log)
        log_message(f"data_path={args.data_path}", train_log)

        data = pd.read_csv(args.data_path)
        X, y = split_features_target(data)

        X_train, X_test, y_train, y_test = make_train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.seed,
        )

        if args.compare_models:
            comparison_df = compare_models(
                X_train=X_train,
                y_train=y_train,
                cv_splits=args.cv,
                seed=args.seed,
            )
            comparison_path = run_paths.run_dir / "model_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            log_message(f"Saved model comparison to: {comparison_path}", train_log)

        pipeline = build_training_pipeline(args.model, random_state=args.seed)
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
            log_message(f"Best params: {selected_hyperparameters}", train_log)
            log_message(f"Best CV ROC-AUC: {search.best_score_:.4f}", train_log)
        else:
            trained_pipeline = pipeline.fit(X_train, y_train)
            selected_hyperparameters = trained_pipeline.named_steps["model"].get_params(
                deep=False
            )

        y_pred = trained_pipeline.predict(X_test)
        y_prob = trained_pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "stage": "train_holdout",
            "run_id": run_paths.run_id,
            "model_name": args.model,
            "seed": args.seed,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "timestamp": now_iso(),
        }

        run_config = {
            "run_id": run_paths.run_id,
            "timestamp": now_iso(),
            "data_path": args.data_path,
            "model_name": args.model,
            "seed": args.seed,
            "test_size": args.test_size,
            "cv": args.cv,
            "tune": args.tune,
            "compare_models": args.compare_models,
            "selected_hyperparameters": selected_hyperparameters,
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
            "paths": {
                "run_dir": str(run_paths.run_dir),
                "model_path": str(run_paths.model_path),
                "metrics_json": str(run_paths.metrics_json),
                "metrics_csv": str(run_paths.metrics_csv),
            },
        }

        run_paths.model_path.write_bytes(pickle.dumps(trained_pipeline))
        save_json(metrics, run_paths.metrics_json)
        save_single_row_csv(metrics, run_paths.metrics_csv)
        save_json(run_config, run_paths.run_config_json)

        log_message(f"Saved model to: {run_paths.model_path}", train_log)
        log_message(f"Saved metrics to: {run_paths.metrics_json} / {run_paths.metrics_csv}", train_log)
        log_message(f"Saved run config to: {run_paths.run_config_json}", train_log)

        print(f"RUN_DIR: {run_paths.run_dir}")
        print(f"MODEL_FILE: {run_paths.model_path}")


if __name__ == "__main__":
    main()
