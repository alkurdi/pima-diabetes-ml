"""Evaluate a saved pipeline within a tracked run directory."""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from experiment_tracking import (
    append_summary_row,
    now_iso,
    resolve_eval_run_paths,
    save_json,
    save_single_row_csv,
)
from preprocessing import make_train_test_split, split_features_target

DEFAULT_SEED = 42
DEFAULT_DATA_PATH = "data/diabetes.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def log_message(message: str, log_file) -> None:
    print(message)
    log_file.write(f"{message}\n")
    log_file.flush()


def main() -> None:
    args = parse_args()
    run_paths = resolve_eval_run_paths(args.run_id)

    if not run_paths.run_config_json.exists():
        raise FileNotFoundError(f"Missing run config: {run_paths.run_config_json}")
    run_config = json.loads(run_paths.run_config_json.read_text())

    seed = args.seed if args.seed is not None else int(run_config.get("seed", DEFAULT_SEED))
    data_path = args.data_path or run_config.get("data_path", DEFAULT_DATA_PATH)
    test_size = args.test_size if args.test_size is not None else float(
        run_config.get("test_size", 0.2)
    )

    set_global_seed(seed)

    with run_paths.eval_log.open("w", encoding="utf-8") as eval_log:
        log_message(f"Starting evaluation for run_id={run_paths.run_id}", eval_log)
        log_message(f"Using model={run_paths.model_path}", eval_log)

        if not run_paths.model_path.exists():
            raise FileNotFoundError(f"Missing model file: {run_paths.model_path}")

        model = pickle.loads(run_paths.model_path.read_bytes())
        data = pd.read_csv(data_path)
        X, y = split_features_target(data)

        _, X_test, _, y_test = make_train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
        )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            "stage": "evaluate_holdout",
            "run_id": run_paths.run_id,
            "model_name": str(run_config.get("model_name", "unknown")),
            "seed": int(seed),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "confusion_matrix_tn": int(cm[0, 0]),
            "confusion_matrix_fp": int(cm[0, 1]),
            "confusion_matrix_fn": int(cm[1, 0]),
            "confusion_matrix_tp": int(cm[1, 1]),
            "timestamp": now_iso(),
        }

        save_json(metrics, run_paths.metrics_json)
        save_single_row_csv(metrics, run_paths.metrics_csv)

        append_summary_row(
            {
                "run_id": metrics["run_id"],
                "model_name": metrics["model_name"],
                "seed": metrics["seed"],
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "timestamp": metrics["timestamp"],
            }
        )

        log_message(f"Accuracy: {metrics['accuracy']:.4f}", eval_log)
        log_message(f"F1: {metrics['f1']:.4f}", eval_log)
        log_message(f"ROC-AUC: {metrics['roc_auc']:.4f}", eval_log)
        log_message(f"Confusion Matrix:\n{cm}", eval_log)
        log_message(f"Saved metrics JSON to: {run_paths.metrics_json}", eval_log)
        log_message(f"Saved metrics CSV to: {run_paths.metrics_csv}", eval_log)
        log_message("Updated results/summary.csv", eval_log)
        print(f"RUN_DIR: {run_paths.run_dir}")


if __name__ == "__main__":
    main()
