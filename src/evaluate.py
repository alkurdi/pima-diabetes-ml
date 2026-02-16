"""Evaluate a saved pipeline on a reproducible holdout split."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from preprocessing import make_train_test_split, split_features_target

DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/diabetes.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/model.pkl"))
    parser.add_argument("--random-state", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--metrics-json", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("results/metrics.csv"))
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """Fix global random states for reproducibility (numpy, random, sklearn usage)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_metrics(metrics: dict[str, float | str | int], json_path: Path, csv_path: Path) -> None:
    """Persist metrics to JSON and single-row CSV."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)


def main() -> None:
    args = parse_args()
    set_global_seed(args.random_state)

    model = pickle.loads(args.model_path.read_bytes())
    data = pd.read_csv(args.data)
    X, y = split_features_target(data)

    _, X_test, _, y_test = make_train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "split": "holdout",
        "model_path": str(args.model_path),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix_tn": int(cm[0, 0]),
        "confusion_matrix_fp": int(cm[0, 1]),
        "confusion_matrix_fn": int(cm[1, 0]),
        "confusion_matrix_tp": int(cm[1, 1]),
    }

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    save_metrics(metrics, args.metrics_json, args.metrics_csv)
    print(f"Saved metrics JSON to: {args.metrics_json}")
    print(f"Saved metrics CSV to: {args.metrics_csv}")


if __name__ == "__main__":
    main()
